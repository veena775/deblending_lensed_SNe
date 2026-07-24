#!/usr/bin/env python
"""Reference-informed scarlet2 scene modelling for SN2025wny.

This script fits a joint, forward model to the SN2025wny field using
pre-supernova reference images plus science images from Gemini, Liverpool
Telescope, P60, or any subset listed in ``scene_manifest.json``.

The model follows the scarlet2 vocabulary:

* a ``Frame`` defines the common model grid;
* each FITS image is an ``Observation`` with its own WCS, PSF and noise model;
* static sources describe the pre-SN scene, and are initialized only from
  reference images so that SN flux is not baked into the galaxy/lens model;
* transient ``PointSource`` components carry a separate flux parameter in every
  science epoch and are set to nearly zero in the reference channels.

The implementation is deliberately explicit rather than compact. It is meant
to be readable/reproducible, and mirrors the ideas in the scarlet2
documentation:

* https://scarlet2.readthedocs.io/
* https://scarlet2.readthedocs.io/en/stable/_autosummary/scarlet2.Scene.html

For time-domain scene modelling context, see Ward et al.,
``Disentangling transients and their host galaxies with scarlet2`` and the
scarlet2 JOSS paper:

* https://arxiv.org/abs/2409.15427
* https://doi.org/10.21105/joss.09646

Typical Gemini-only run::

    conda run -n scarlet_env python scene_modeling/run_scarlet2_scene_model.py \
        --telescopes Gemini-North --bands gri \
        --outdir scene_modeling/scarlet2_gemini_gri_science

Typical Gemini + LT run with a comparison to the DIA table::

    conda run -n scarlet_env python scene_modeling/run_scarlet2_scene_model.py \
        --telescopes Gemini-North,Liverpool\\ Telescope --bands gri \
        --lt-dia-csv /Users/suhaildhawan/workspaces/Lensing/SN2025wny/photometry/lt_dia_lcs.csv \
        --outdir scene_modeling/scarlet2_gemini_lt_gri_science
"""

from __future__ import annotations

import argparse
import csv
import sys 
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


SOURCE_LABELS = ["A", "B", "C", "D", "E"]
SOURCE_COORDS = SkyCoord(
    ra=[109.14377825, 109.14399136, 109.14327965, 109.14238996, 109.14311400] * u.deg,
    dec=[38.35226376, 38.35167350, 38.35145402, 38.35219114, 38.35260380] * u.deg,
    frame="icrs",
)

STATIC_LABELS = ["lens_g1", "host_g2"]
STATIC_COORDS = SkyCoord(
    ra=[109.14334799, 109.14300967] * u.deg,
    dec=[38.35190011, 38.35248824] * u.deg,
    frame="icrs",
)

FILTER_TO_BAND = {
    "g": "g",
    "r": "r",
    "i": "i",
    "z": "z",
    "SDSS_G": "g",
    "SDSS_R": "r",
    "SDSS_I": "i",
    "SDSS_Z": "z",
}

# Maps (telescope, instrument) from the manifest to the stem used in
# model_info/PSFs/psfs_{KEY}_{band}.npy and model_info/noisemaps/noisemap_{KEY}_{band}.npy.
# Telescopes absent from this map (CFHT, Pan-STARRS1) have no npy files and
# fall back to the Gaussian PSF / annulus-RMS weight approaches.
TELESCOPE_INSTRUMENT_TO_NPY_KEY: dict[tuple[str, str], str] = {
    ("Gemini-North",        "GMOS-N"):    "Gemini",
    ("Liverpool Telescope", "IOO/Loci"):  None,   # resolved per-instrument in _npy_key_for_row
    ("Liverpool Telescope", "IOO"):       "LTIOO",
    ("Liverpool Telescope", "Loci"):      "LTLoci",
    ("P60",                 "SEDM/P60"):  "P60",
    ("LBT",                 "LBC"):       "LBT",
    ("Wendelstein",         "WWFI"):      "Wendelstein",
}

# MJD tolerance (days) for matching a manifest row to a metadata epoch.
MJD_MATCH_TOL = 0.01


def ab_count_to_nanomaggy_factor(zp: float) -> float:
    """Return the multiplicative conversion from counts to nanomaggies.

    The manifest stores an AB zeropoint for each input image. A source with
    count rate ``C`` has ``mag = zp - 2.5 log10(C)``. Nanomaggies use the SDSS
    convention ``mag = 22.5 - 2.5 log10(flux_nanomaggy)``; equating the two
    expressions gives the conversion used here.
    """

    if not np.isfinite(zp):
        return 1.0
    return float(10 ** (0.4 * (22.5 - zp)))


def gaussian_psf_kernel(sigma_pix: float, size: int = 31) -> np.ndarray:
    """Build a normalized Gaussian PSF image for scarlet2 ``ArrayPSF``.

    The manifest PSF estimates are simple empirical Gaussian widths. A full
    science analysis should replace this with per-image empirical PSFs, but the
    normalized kernel keeps the renderer honest about the image resolution and
    is much better than treating all frames as having the same seeing.
    """

    if not np.isfinite(sigma_pix) or sigma_pix <= 0:
        sigma_pix = 2.0
    size = int(max(size, 8 * sigma_pix))
    if size % 2 == 0:
        size += 1
    yy, xx = np.mgrid[:size, :size]
    cen = (size - 1) / 2.0
    rr2 = (xx - cen) ** 2 + (yy - cen) ** 2
    kernel = np.exp(-0.5 * rr2 / sigma_pix**2)
    return (kernel / np.sum(kernel)).astype("float32")


def read_hdu(path: Path, ext: int | str) -> tuple[np.ndarray, fits.Header]:
    """Read one FITS extension as ``float32`` data plus a copied header."""

    with fits.open(path) as hdul:
        hdu = hdul[ext]
        return np.asarray(hdu.data, dtype="float32"), hdu.header.copy()


def read_weight_cutout(
    row: dict,
    shape: tuple[int, int],
    center: SkyCoord,
    size: u.Quantity,
) -> np.ndarray | None:
    """Read and cut out a native inverse-variance map when one is available.

    LT coadds include weight maps in the manifest. If a weight map is missing,
    has a different shape, or cannot be read, the caller falls back to a robust
    background RMS estimate from the science cutout.
    """

    weight_path = row.get("weight_path")
    if not weight_path:
        return None
    path = Path(weight_path)
    if not path.exists():
        return None
    try:
        weight_data, weight_header = read_hdu(path, 0)
        weight_wcs = WCS(weight_header).celestial
        weight_cut = Cutout2D(weight_data, center, size, wcs=weight_wcs, mode="partial", fill_value=0.0)
        if weight_cut.data.shape == shape:
            return np.asarray(weight_cut.data, dtype="float32")
    except Exception:
        return None
    return None


def robust_weight_from_cutout(image: np.ndarray, row: dict, factor: float, fallback_std: float) -> np.ndarray:
    """Construct a conservative inverse-variance image for one observation.

    scarlet2 optimizes a likelihood weighted by inverse variance. The quick DIA
    scripts used scalar background RMS values; here we keep that scalar model
    but guard against non-finite values and absurdly small variances. This makes
    optimizer failures much easier to interpret because a bad manifest entry
    cannot silently assign infinite weight to one frame.
    """

    sigma_counts = float(row.get("background_rms", fallback_std))
    sigma = sigma_counts * factor if np.isfinite(sigma_counts) and sigma_counts > 0 else float(np.nanstd(image))
    sigma = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
    return np.ones_like(image, dtype="float32") / max(sigma**2, 1e-20)


def fit_moffat_psf(psf: np.ndarray, tag: str = "") -> np.ndarray:
    """Fit a 2-D Gaussian profile to a PSF array and return the smooth model.

    Empirical PSF stamps (especially per-epoch npy PSFs and PS1's stacked
    template PSF) carry substantial pixel-to-pixel noise in their wings. This
    noise gets picked up by the renderer/Starlet decomposition and produces
    ringing/pixellation artefacts. Fitting an analytical Gaussian profile
    (amplitude, x0, y0, x_stddev, y_stddev, theta, with a constant background)
    removes that noise while preserving the overall PSF width and ellipticity.

    The input is clipped at zero, fit on its native pixel grid (so this should
    be called BEFORE any downsampling for oversampled npy PSFs), and the
    returned array has the same shape as the input, normalised to sum=1. The
    fitted constant background is discarded -- only the Gaussian component
    contributes to the returned PSF.
    Falls back to returning the (clipped, normalised) input if the fit fails
    or produces a non-positive model.
    """

    from astropy.modeling import models, fitting

    arr = np.clip(np.asarray(psf, dtype="float64"), 0.0, None)
    h, w = arr.shape
    yy, xx = np.mgrid[:h, :w].astype("float64")

    amp0 = float(arr.max())
    iy0, ix0 = np.unravel_index(np.argmax(arr), arr.shape)

    gaussian = models.Gaussian2D(
        amplitude=amp0, x_mean=float(ix0), y_mean=float(iy0),
        x_stddev=2.0, y_stddev=2.0, theta=0.0,
    ) + models.Const2D(amplitude=0.0)
    gaussian[0].x_stddev.bounds = (0.3, max(h, w))
    gaussian[0].y_stddev.bounds = (0.3, max(h, w))

    fitter = fitting.LevMarLSQFitter()
    try:
        fitted = fitter(gaussian, xx, yy, arr, maxiter=200)
        # Evaluate only the Gaussian component (index 0), discarding the
        # constant background (index 1) which was used to improve the fit
        # but should not contribute flux to the PSF model itself.
        model_arr = np.asarray(fitted[0](xx, yy), dtype="float32")
        model_arr = np.clip(model_arr, 0.0, None)
        total = model_arr.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError(f"Gaussian model sum is {total}")
        model_arr /= total
        sx_fit = float(fitted[0].x_stddev.value)
        sy_fit = float(fitted[0].y_stddev.value)
        bkg_fit = float(fitted[1].amplitude.value)
        print(f"  PSF_GAUSS{' ' + tag if tag else ''}: fit OK "
              f"(x_stddev={sx_fit:.3f}, y_stddev={sy_fit:.3f}, "
              f"bkg={bkg_fit:.3e} discarded)")
        return model_arr
    except Exception as exc:
        print(f"  PSF_GAUSS{' ' + tag if tag else ''}: fit failed ({exc}), "
              f"using clipped/normalised empirical PSF")
        total = arr.sum()
        if total > 0:
            arr = arr / total
        return arr.astype("float32")


def _downsample_psf(psf: np.ndarray, factor: int, tag: str = "") -> np.ndarray:
    """Downsample a 2-D PSF array by an integer ``factor`` using Lanczos
    interpolation.

    Used both for npy PSF models built at 2x the native pixel scale, and for
    PS1's Gaussian PSF model (``OVERSAMP=3`` in its FITS header) which is
    built at 3x the native pixel scale. Resamples using ``resample2d`` from
    ``scarlet2.interpolation`` (Lanczos-3), then renormalises so the PSF sums
    to 1.  Falls back to ``scipy.ndimage.zoom`` if the Lanczos result looks
    degenerate (sum < 0.5).

    The output shape is ``(ceil(H/factor), ceil(W/factor))``.
    """
    import jax.numpy as jnp
    from scarlet2.interpolation import Lanczos, resample2d

    if factor == 1:
        psf = np.clip(np.asarray(psf, dtype="float32"), 0.0, None)
        total = psf.sum()
        if total > 0:
            psf = psf / total
        return psf

    h_in, w_in = psf.shape
    h_out = int(np.ceil(h_in / factor))
    w_out = int(np.ceil(w_in / factor))

    # Input coordinate grid: uniform spacing of 1 pixel.
    # resample2d expects coords[0,:,0] = y-coords, coords[:,0,1] = x-coords.
    ys_in = np.arange(h_in, dtype="float32")
    xs_in = np.arange(w_in, dtype="float32")
    coords = np.stack(np.meshgrid(ys_in, xs_in, indexing="ij"), axis=-1).astype("float32")

    # Output sample positions in input-pixel units (factor-of-N subsampling).
    # Use centres of NxN blocks so each output pixel maps to the centre of an
    # NxN input region rather than the top-left corner.
    half = (factor - 1) / 2.0
    ys_out = np.linspace(half, h_in - 1 - half, h_out, dtype="float32")
    xs_out = np.linspace(half, w_in - 1 - half, w_out, dtype="float32")
    warp = np.stack(np.meshgrid(ys_out, xs_out, indexing="ij"), axis=-1).astype("float32")

    psf_down = np.asarray(
        resample2d(
            jnp.asarray(psf),
            jnp.asarray(coords),
            jnp.asarray(warp),
            interpolant=Lanczos(3),
        )
    ).astype("float32")

    pre_clip_sum = float(psf_down.sum())
    pre_clip_max = float(psf_down.max())
    n_negative = int((psf_down < 0).sum())

    psf_down = np.clip(psf_down, 0.0, None)
    post_clip_sum = float(psf_down.sum())

    print(f"  PSF downsample{' ' + tag if tag else ''}: factor={factor} "
          f"({h_in},{w_in})->({h_out},{w_out}), "
          f"pre_clip sum={pre_clip_sum:.4f} max={pre_clip_max:.4f} n_neg={n_negative}, "
          f"post_clip sum={post_clip_sum:.4f}")

    # If the Lanczos result looks degenerate, fall back to scipy zoom.
    if post_clip_sum < 0.5:
        print(f"  PSF downsample WARNING: Lanczos sum={post_clip_sum:.4f} < 0.5, "
              f"falling back to scipy.ndimage.zoom")
        from scipy.ndimage import zoom as _zoom
        psf_down = _zoom(psf, 1.0 / factor, order=3).astype("float32")
        psf_down = np.clip(psf_down, 0.0, None)
        post_clip_sum = float(psf_down.sum())
        print(f"  PSF downsample scipy fallback: sum={post_clip_sum:.4f}, shape={psf_down.shape}")

    if post_clip_sum > 0:
        psf_down /= post_clip_sum

    return psf_down


def _downsample_psf_2x(psf: np.ndarray, tag: str = "") -> np.ndarray:
    """Downsample a 2-D PSF array by a factor of 2. See ``_downsample_psf``."""
    return _downsample_psf(psf, 2, tag=tag)


def _load_reference_empirical_psf(
    telescope: str,
    band: str,
    model_info_dir: Path,
    tag: str = "",
) -> np.ndarray | None:
    """Load an empirical reference PSF from model_info/reference_psf_models/.

    Covers both CFHT (g, r) and Pan-STARRS1 (i, z) which share the same
    directory and filename convention:
        ``{band}_reference_psf_empirical.fits``

    For Pan-STARRS1, an analytical Gaussian profile is fitted to the
    empirical stamp (via ``fit_moffat_psf``) to remove pixel noise in the
    wings before normalisation.

    Returns a normalised 2-D float32 array, or None if the file is missing.
    """

    from astropy.io import fits as _fits

    psf_path = model_info_dir / "reference_psf_models" / f"{band}_reference_psf_empirical.fits"
    if not psf_path.exists():
        print(f"{tag} empirical PSF not found at {psf_path} -> Gaussian")
        return None

    try:
        with _fits.open(psf_path) as hdul:
            psf = np.asarray(hdul[0].data, dtype="float32")
        while psf.ndim > 2:
            psf = psf[0]
        psf = np.clip(psf, 0.0, None)

        if telescope == "Pan-STARRS1":
            # Fit an analytical Gaussian profile to remove pixel noise in the
            # wings of the stacked empirical PSF (same approach as P60 and
            # Liverpool Telescope npy PSFs).
            psf = fit_moffat_psf(psf, tag=tag)

        total = psf.sum()
        if total <= 0:
            print(f"{tag} empirical PSF sums to zero at {psf_path} -> Gaussian")
            return None
        psf /= total
        print(f"{tag} OK -> empirical PSF ({telescope}) shape={psf.shape} sum={psf.sum():.4f}")
        return psf
    except Exception as exc:
        print(f"{tag} empirical PSF load error ({exc}) -> Gaussian")
        return None


def _npy_key_for_row(row: dict) -> str | None:
    """Return the model_info npy stem for a manifest row, or None if unavailable."""
    telescope = row.get("telescope", "")
    instrument = row.get("instrument", "")
    key = TELESCOPE_INSTRUMENT_TO_NPY_KEY.get((telescope, instrument))
    if key is not None:
        return key
    # Liverpool Telescope rows use instrument "IOO/Loci" in the manifest;
    # distinguish IOO vs Loci by the file path.
    if telescope == "Liverpool Telescope":
        path_str = row.get("path", "")
        if "/Loci/" in path_str or "_Loci_" in path_str:
            return "LTLoci"
        return "LTIOO"
    return None


def load_psf_and_noisemap(
    row: dict,
    model_info_dir: Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load the per-epoch PSF and noisemap arrays for one manifest row.

    Returns ``(psf_2d, noisemap_2d)`` as float32 arrays, or ``(None, None)``
    if no model_info files exist for this telescope/band (e.g. CFHT, PS1).

    PSF npy files have shape ``(N_epochs, H, W)`` with each slice summing to 1.
    Noisemap npy files have shape ``(N_epochs, H, W)`` in flux units matching
    the image after zeropoint conversion and the metadata ``scale`` factor.

    The correct epoch is found by matching ``row["mjd"]`` to the ``mjds`` list
    in the companion metadata JSON within ``MJD_MATCH_TOL`` days.

    Prints a one-line diagnostic for every call so PSF provenance is visible
    in the run log.
    """

    telescope = row.get("telescope", "?")
    band = row.get("band", "?")
    mjd = row.get("mjd", float("nan"))
    tag = f"  PSF [{telescope} {band} MJD={mjd:.2f}]:"

    if model_info_dir is None:
        print(f"{tag} FALLBACK (no model_info_dir) -> Gaussian")
        return None, None

    npy_key = _npy_key_for_row(row)
    if npy_key is None:
        # CFHT (g, r) and Pan-STARRS1 (i, z) share empirical PSF files in
        # reference_psf_models/. Try those before falling back to Gaussian.
        if telescope in ("CFHT", "Pan-STARRS1"):
            ref_psf = _load_reference_empirical_psf(telescope, band, model_info_dir, tag)
            if ref_psf is not None:
                return ref_psf, None
        print(f"{tag} FALLBACK (no npy key for telescope='{telescope}') -> Gaussian")
        return None, None

    psf_path   = model_info_dir / "PSFs"      / f"psfs_{npy_key}_{band}.npy"
    noise_path = model_info_dir / "noisemaps" / f"noisemap_{npy_key}_{band}.npy"
    meta_path  = model_info_dir               / f"metadata_{npy_key}_{band}.json"

    if not psf_path.exists():
        print(f"{tag} FALLBACK (PSF file not found: {psf_path}) -> Gaussian")
        return None, None
    if not meta_path.exists():
        print(f"{tag} FALLBACK (metadata not found: {meta_path}) -> Gaussian")
        return None, None

    import json as _json
    meta = _json.loads(meta_path.read_text())
    mjds = np.asarray(meta["mjds"], dtype=float)
    frame_ids = np.asarray(meta["frame_ids"], dtype=int)

    row_mjd = float(row.get("mjd", np.nan))
    diffs = np.abs(mjds - row_mjd)
    best = int(np.argmin(diffs))
    if diffs[best] > MJD_MATCH_TOL:
        print(f"{tag} FALLBACK (MJD {row_mjd:.4f} unmatched; "
              f"closest={mjds[best]:.4f}, diff={diffs[best]:.4f}d > tol={MJD_MATCH_TOL}) -> Gaussian")
        return None, None

    raw_idx = int(frame_ids[best])
    # Determine if frame_ids are 0-based or 1-based by checking against stack size.
    psf_stack = np.load(psf_path)
    n_epochs = psf_stack.shape[0]
    # If raw_idx is out of bounds for 0-based, try 1-based (subtract 1).
    if raw_idx >= n_epochs and (raw_idx - 1) < n_epochs:
        epoch_idx = raw_idx - 1
    elif raw_idx < n_epochs:
        epoch_idx = raw_idx
    else:
        print(f"{tag} FALLBACK (frame_id={raw_idx} out of range for stack size {n_epochs}) -> Gaussian")
        return None, None

    psf = psf_stack[epoch_idx].astype("float32")
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum

    # For P60 and Liverpool Telescope, fit an analytical Gaussian profile to the
    # oversampled (pre-downsample) PSF to remove pixel noise in the wings
    # before resampling to native resolution.
    if row.get("telescope") in ("P60", "Liverpool Telescope"):
        psf = fit_moffat_psf(psf, tag=tag)

    # Downsample the PSF by a factor of 2: the npy PSF models are built at
    # 2x the native pixel scale and must be resampled to native resolution
    # before being passed to scarlet2.
    psf = _downsample_psf_2x(psf, tag=tag)

    noisemap = None
    noise_tag = "no noisemap"
    if noise_path.exists():
        noise_stack = np.load(noise_path)
        noisemap = noise_stack[epoch_idx].astype("float32")
        noise_tag = f"noisemap shape={noisemap.shape}"

    print(f"{tag} OK -> npy epoch_idx={epoch_idx} (MJD match diff={diffs[best]:.5f}d), "
          f"PSF shape={psf.shape} (downsampled 2x), {noise_tag}")
    return psf, noisemap


def local_rms_weight(image: np.ndarray, factor: float) -> np.ndarray:
    """Build a flat inverse-variance map from a local background annulus.

    Rather than using the global ``background_rms`` from the manifest (which
    for large template cutouts like Pan-STARRS is measured over the full frame
    and may not represent the noise near the source), this function estimates
    the RMS from an annulus in the outer half of the cutout.  This gives a
    noise estimate that is local to the scene region actually being fitted.

    Used in place of ``robust_weight_from_cutout`` for Pan-STARRS observations,
    which have large templates and no native weight map.
    """

    h, w = image.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.hypot(yy - cy, xx - cx)
    r_max = min(cy, cx)
    # Annulus spanning 60-85% of the half-width, away from the central source.
    annulus = (rr > 0.60 * r_max) & (rr <= 0.85 * r_max) & np.isfinite(image)
    if annulus.sum() > 10:
        sigma = float(np.nanstd(image[annulus]))
    else:
        sigma = float(np.nanstd(image[np.isfinite(image)])) if np.any(np.isfinite(image)) else 1.0
    sigma = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
    # image is already in nanomaggies (factor applied); weights are in 1/nmgy^2.
    return np.ones_like(image, dtype="float32") / max(sigma ** 2, 1e-20)


def snr_weight_factor(image: np.ndarray, weights: np.ndarray) -> float:
    """Estimate peak S/N of a stamp for relative downweighting of noisy frames.

    Uses the peak pixel divided by the per-pixel noise estimate (from weights).
    Returns a scalar in (0, 1] — 1.0 for the highest-S/N frame, lower for
    noisier frames.  The caller scales weights by this factor before fitting.
    """
    w = np.asarray(weights)
    pos_w = w[w > 0]
    if len(pos_w) == 0:
        return 1.0
    # Noise per pixel from median weight
    sigma = float(1.0 / np.sqrt(np.nanmedian(pos_w)))
    peak = float(np.nanmax(image[np.isfinite(image)]))
    if sigma <= 0 or not np.isfinite(peak) or peak <= 0:
        return 1.0
    return float(np.clip(peak / sigma, 0.0, None))


def sep_background_subtract(data: np.ndarray, sn: float = 3.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Subtract a spatially-varying SEP background model from a cutout.

    Unlike a single sigma-clipped scalar median, ``sep.Background`` fits a
    2-D background map (and RMS map) using a grid of boxes with median
    filtering. This handles background gradients and large-scale structure
    much better for big template cutouts (e.g. Pan-STARRS1, 130 arcsec),
    where a single scalar median can leave residual gradients that get
    decomposed as spurious low-frequency Starlet coefficients.

    NaNs are replaced with the global median before running SEP (SEP requires
    a contiguous float32 array with no NaNs), and the corresponding pixels are
    zeroed in the output.

    Returns
    -------
    subtracted : np.ndarray
        Background-subtracted data, float32, with original NaN positions set
        to 0.
    rms_map : np.ndarray
        2-D background RMS map from SEP, same shape as ``data``.
    global_rms : float
        SEP's global RMS estimate.
    """

    import sep

    arr = np.asarray(data, dtype="float64")
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        fill = float(np.nanmedian(arr[~nan_mask])) if (~nan_mask).any() else 0.0
        arr = np.where(nan_mask, fill, arr)

    # SEP requires C-contiguous arrays.
    arr = np.ascontiguousarray(arr)

    # Two-pass background estimation: first pass gets a rough RMS, second
    # pass masks pixels brighter than sn * globalrms so source flux doesn't
    # bias the background model. With sn=0.01 essentially no pixels are masked
    # (equivalent to no background subtraction); with sn=3.0 (default) source
    # pixels are masked, preventing over-subtraction on small cutouts.
    bkg0 = sep.Background(arr, bw=32, bh=32, fw=3, fh=3)
    if sn < 3.0:
        # Very low threshold: mask only extreme outliers
        source_mask = arr > (bkg0.globalback + sn * bkg0.globalrms)
        bkg = sep.Background(arr, mask=source_mask.astype(np.uint8), bw=32, bh=32, fw=3, fh=3)
    else:
        bkg = bkg0
    subtracted = (arr - bkg.back()).astype("float32")
    rms_map = bkg.rms().astype("float32")
    global_rms = float(bkg.globalrms)

    subtracted[nan_mask] = 0.0
    rms_map[nan_mask] = 0.0

    return subtracted, rms_map, global_rms


def load_observation(row: dict, channel: tuple[str, int], center: SkyCoord, cutout_arcsec: float, args_model_info_dir: Path | None = None):
    """Load one manifest row as a scarlet2 ``Observation``.

    Steps performed here:

    1. cut out the same sky region from every input frame;
    2. subtract a sigma-clipped scalar sky level;
    3. convert the image and inverse variance to nanomaggies;
    4. normalize the image by its stamp sum so that all observations enter the
       fit on a common scale, removing zeropoint-driven amplitude differences
       that otherwise bias the initialization (e.g. Pan-STARRS appearing too
       bright relative to CFHT).  The scale factor is returned alongside the
       observation so that fitted fluxes can be rescaled to physical units.
    5. attach a Gaussian ``ArrayPSF`` and WCS so scarlet2 can render the common
       model into this particular image.

    Returns
    -------
    obs : scarlet2.Observation
    scale_factor : float
        Always 1.0; retained for API compatibility.
    """

    import jax.numpy as jnp
    import scarlet2

    path = Path(row["path"])
    ext = row.get("ext", 0)
    data, header = read_hdu(path, ext)
    wcs = WCS(header).celestial
    size = (cutout_arcsec * u.arcsec, cutout_arcsec * u.arcsec)
    cutout = Cutout2D(data, center, size, wcs=wcs, mode="partial", fill_value=np.nan)

    _pixscale_arcsec = float(np.mean(proj_plane_pixel_scales(cutout.wcs)) * 3600.0)
    print(f"  WCS_PIXSCALE [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
          f"{_pixscale_arcsec:.4f} arcsec/px  cutout.data.shape={cutout.data.shape}  "
          f"cutout_arcsec={cutout_arcsec}  "
          f"implied_size={cutout_arcsec / _pixscale_arcsec:.1f}px")

    _, median, std = sigma_clipped_stats(cutout.data, sigma=3.0, maxiters=5)
    factor = ab_count_to_nanomaggy_factor(float(row.get("zp", np.nan)))

    # Capture the original NaN/masked-pixel mask BEFORE SEP fills it. These
    # are genuinely missing data (chip gaps, masked regions, partial coverage
    # near cutout edges) -- not just noisy background.
    nan_mask = ~np.isfinite(cutout.data)

    # Run SEP's 2-D background + RMS estimation. For Pan-STARRS1, skip the
    # background subtraction step (use sigma-clipped median instead) since SEP
    # over-subtracts the source flux on small cutouts, leaving large negative
    # residuals that corrupt the likelihood. SEP RMS is still used for weights.
    if row.get("telescope") == "Pan-STARRS1":
        sub, sep_rms_map, sep_global_rms = sep_background_subtract(cutout.data, sn=0.01)
        # Use sigma-clipped median for background subtraction rather than SEP
        # (which over-subtracts on small cutouts) or no subtraction (which
        # leaves a large pedestal that dominates the PS1 flux budget).
        # The sigma-clipped median excludes the bright source pixels, so it
        # is a good estimator of the true sky background level.
        image = np.nan_to_num(cutout.data - median, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        print(f"  SEP_BKG [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
              f"global_rms={sep_global_rms:.4e}  sigma_clipped_median={median:.4e}  "
              f"(PS1: sigma-clipped median background subtraction, SEP RMS for weights)")
    else:
        sub, sep_rms_map, sep_global_rms = sep_background_subtract(cutout.data)
        image = sub.astype("float32")
        print(f"  SEP_BKG [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
              f"global_rms={sep_global_rms:.4e}  "
              f"sigma_clipped_median={median:.4e}  sigma_clipped_std={std:.4e}")
    image *= factor

    # Build inverse-variance weights directly from SEP's per-pixel background
    # RMS map, converted to nanomaggy units. Pixels with non-positive/invalid
    # RMS fall back to the map's median.
    sigma_map = (sep_rms_map * factor).astype("float32")
    valid = sigma_map > 0
    if valid.any():
        fallback_sigma = float(np.nanmedian(sigma_map[valid]))
        sigma_map = np.where(valid, sigma_map, fallback_sigma)
    else:
        sigma_map = np.ones_like(sigma_map)
    weights = (1.0 / np.maximum(sigma_map ** 2, 1e-20)).astype("float32")
    weights[~np.isfinite(weights)] = 0.0

    # Genuinely missing/masked pixels get zero weight regardless of the
    # fallback-sigma logic above, so the fit simply ignores them rather than
    # comparing the model against fabricated zero data with a non-zero weight
    # (which previously produced large spurious residuals e.g. for Pan-STARRS1
    # templates with masked regions at the source position).
    n_masked = int(nan_mask.sum())
    if n_masked > 0:
        weights[nan_mask] = 0.0
        print(f"  MASK [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
              f"{n_masked} masked/NaN pixel(s) -> weight=0")

    # Retain native weight files only as a diagnostic comparison (not used).
    native_weight = read_weight_cutout(row, image.shape, center, size)

    # No per-image normalization: images are in nanomaggies after zeropoint
    # conversion and must remain in consistent physical units across epochs
    # so that StaticArraySpectrum can share flux values across all epochs of
    # a given band.
    scale_factor = 1.0

    # Diagnostic: print image stats so we can verify units are correct.
    _finite = image[np.isfinite(image)]
    _pos = _finite[_finite > 0]
    print(f"  UNITS [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
          f"zp={row.get('zp', float('nan')):.2f}  factor={factor:.4e}  "
          f"image median={float(np.median(_finite)):.4e}  "
          f"image max={float(_finite.max()) if len(_finite) else float('nan'):.4e}  "
          f"pos_sum={float(_pos.sum()) if len(_pos) else 0:.4e}")

    # ── PSF ──────────────────────────────────────────────────────────────────
    # Prefer the per-epoch PSF array from model_info/PSFs/ when available.
    # Fall back to a Gaussian for telescopes without npy files (CFHT, PS1).
    npy_psf, npy_noisemap = load_psf_and_noisemap(row, args_model_info_dir)
    if npy_psf is not None:
        psf_kernel = np.asarray(npy_psf, dtype="float32")
        psf_source = "empirical/npy"
    else:
        psf_kernel = gaussian_psf_kernel(float(row.get("psf_sigma_pix", np.nan)))
        psf_source = "Gaussian"

    # Sanitize: zero out non-finite pixels, clip negatives, renormalize to sum=1.
    n_bad_psf = int((~np.isfinite(psf_kernel)).sum())
    if n_bad_psf > 0:
        psf_kernel = np.nan_to_num(psf_kernel, nan=0.0, posinf=0.0, neginf=0.0)
    psf_kernel = np.clip(psf_kernel, 0.0, None)
    psf_sum = float(psf_kernel.sum())
    if psf_sum > 0:
        psf_kernel = psf_kernel / psf_sum
    else:
        print(f"  PSF [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
              f"WARNING sum<=0 after sanitization, falling back to Gaussian")
        psf_kernel = gaussian_psf_kernel(float(row.get("psf_sigma_pix", np.nan)))
        psf_source = "Gaussian (fallback after bad PSF)"
        psf_sum = float(psf_kernel.sum())
        if psf_sum > 0:
            psf_kernel = psf_kernel / psf_sum

    print(f"  PSF_CHECK [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
          f"source={psf_source}  shape={psf_kernel.shape}  "
          f"n_nonfinite_before={n_bad_psf}  sum_after={float(psf_kernel.sum()):.4f}  "
          f"min={float(psf_kernel.min()):.3e}  max={float(psf_kernel.max()):.3e}")

    psf = scarlet2.ArrayPSF(jnp.asarray([psf_kernel.astype("float32")]))

    # ── Noisemap (diagnostic only) ────────────────────────────────────────────
    # The npy noisemaps were among the unreliable weight sources, so they no
    # longer override the SEP-derived weights computed above. Just report
    # their shape for diagnostic comparison.
    if npy_noisemap is not None:
        if npy_noisemap.shape != image.shape:
            print(f"  Weights [{row.get('telescope')} {row.get('band')} MJD={row.get('mjd', 0):.2f}]: "
                  f"npy noisemap shape={npy_noisemap.shape} (not used; SEP weights used instead)")
        else:
            print(f"  Weights [{row.get('telescope')} {row.get('band')} MJD={row.get('mjd', 0):.2f}]: "
                  f"npy noisemap shape={npy_noisemap.shape} (not used; SEP weights used instead)")

    # Diagnostic: confirm weights were read in and are non-trivial.
    _w = np.asarray(weights)
    _n_pos = int((_w > 0).sum())
    _n_zero = int((_w == 0).sum())
    _w_source = "SEP RMS map"
    print(f"  WEIGHTS [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
          f"source={_w_source}  n_pos={_n_pos}  n_zero={_n_zero}  "
          f"median_pos={float(np.median(_w[_w > 0])):.3e}  max={float(_w.max()):.3e}")

    # Final sanitization: any non-finite pixel in image or weights gets zero
    # weight and zero data value. This prevents NaN/Inf pixels (seen
    # particularly in Pan-STARRS templates, which have ragged coverage edges)
    # from propagating into the fit and corrupting the initialization.
    bad = ~np.isfinite(image) | ~np.isfinite(weights)
    n_bad = int(bad.sum())
    if n_bad > 0:
        image[bad] = 0.0
        weights[bad] = 0.0
        print(f"  SANITIZE [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
              f"zeroed {n_bad} non-finite pixel(s)")

    obs = scarlet2.Observation(
        jnp.asarray(image.reshape(1, *image.shape)),
        weights=jnp.asarray(weights.reshape(1, *weights.shape)),
        psf=psf,
        wcs=cutout.wcs,
        channels=[channel],
    )
    return obs, scale_factor


def gaussian_morphology(size: int, sigma: float) -> np.ndarray:
    """Fallback positive Gaussian morphology with unit integral."""

    yy, xx = np.mgrid[:size, :size]
    cen = (size - 1) / 2.0
    rr2 = (xx - cen) ** 2 + (yy - cen) ** 2
    morph = np.exp(-0.5 * rr2 / sigma**2)
    morph[morph < 1e-8] = 1e-8
    morph /= np.sum(morph)
    return morph.astype("float32")


def cut_square(image: np.ndarray, yx: Iterable[float], size: int, fill_value: float = 0.0) -> np.ndarray:
    """Return a fixed-size square cutout centered on ``(y, x)``.

    ``scarlet2.Frame.get_pixel`` uses array order ``(y, x)``. This helper keeps
    the ordering explicit and pads edges rather than changing stamp size.
    """

    y, x = [int(round(float(v))) for v in yx]
    half = size // 2
    out = np.full((size, size), fill_value, dtype="float32")
    y0, y1 = y - half, y + half + 1
    x0, x1 = x - half, x + half + 1
    src_y0, src_y1 = max(y0, 0), min(y1, image.shape[-2])
    src_x0, src_x1 = max(x0, 0), min(x1, image.shape[-1])
    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def reference_static_initializer(
    all_observations,
    all_rows: list[dict],
    coord: SkyCoord,
    bands: list[str],
    stamp_size: int,
    fallback_sigma: float,
    min_snr: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a static source spectrum and morphology from all available images.

    Previously this function only used the 4 pre-SN reference frames, giving
    poor flux estimates for the lens because only 2 bands were available and
    the 15th-percentile background subtraction clipped the compact core.

    Now it pools stamps from every observation (references and science), using
    a local background annulus rather than a stamp percentile, and only accepts
    frames where the source peak exceeds min_snr * sky_rms.  Science frames
    contribute to both the morphology median and per-band spectrum, giving many
    more stamps and a more robust initialization.

    Parameters
    ----------
    all_observations : list
        All CorrelatedObservations, references first then science.
    all_rows : list[dict]
        Matching manifest rows.
    coord : SkyCoord
        Sky position of this static source.
    bands : list[str]
        Ordered list of band labels for the spectrum vector.
    stamp_size : int
        Side length of the square stamp cut around the source (pixels).
    fallback_sigma : float
        Gaussian sigma used for the morphology if no usable stamp is found.
    min_snr : float
        Minimum peak SNR for a stamp to be accepted. Default 3.0.
    """

    half = stamp_size // 2
    # Annulus for background estimation: ring just outside the central half.
    yy_full, xx_full = np.mgrid[:stamp_size, :stamp_size]
    cen_full = (stamp_size - 1) / 2.0
    rr_full = np.hypot(yy_full - cen_full, xx_full - cen_full)
    annulus_mask = (rr_full > half * 0.65) & (rr_full <= half * 0.90)

    stamps = []
    flux_by_band: dict[str, list[float]] = defaultdict(list)

    for obs, row in zip(all_observations, all_rows, strict=True):
        band = row["band"]
        image = np.asarray(obs.data[0], dtype="float32")
        weights = np.asarray(obs.weights[0], dtype="float32")

        yx = np.asarray(obs.frame.get_pixel(coord), dtype=float)
        stamp = cut_square(image, yx, stamp_size)
        wstamp = cut_square(weights, yx, stamp_size)

        finite = np.isfinite(stamp)
        if finite.sum() < stamp_size:
            continue

        # Background from annulus so the compact lens core is not subtracted away.
        ann_vals = stamp[annulus_mask & finite]
        sky = float(np.nanmedian(ann_vals)) if ann_vals.size > 0 else 0.0
        sky_rms = float(np.nanstd(ann_vals)) if ann_vals.size > 1 else np.nan

        source = np.nan_to_num(stamp - sky, nan=0.0, posinf=0.0, neginf=0.0)
        source[source < 0] = 0.0

        # SNR check: peak pixel must exceed min_snr * sky_rms.
        if np.isfinite(sky_rms) and sky_rms > 0:
            peak_snr = float(np.nanmax(source)) / sky_rms
            if peak_snr < min_snr:
                continue
        elif np.nanmax(wstamp) > 0:
            noise = float(1.0 / np.sqrt(np.nanmax(wstamp)))
            if float(np.nanmax(source)) < min_snr * noise:
                continue

        total = float(np.sum(source))
        if total <= 0 or not np.isfinite(total):
            continue

        if row["telescope"] == "Pan-STARRS1":
            cen = stamp_size // 2
            img_argmax = np.unravel_index(np.argmax(image), image.shape)
            print(f"  PS1_POS [{band} MJD={row.get('mjd',0):.1f}]: "
                  f"pixel=({yx[0]:.2f}, {yx[1]:.2f})  "
                  f"image_shape={image.shape}  image_argmax={img_argmax}  "
                  f"image_argmax_value={float(image[img_argmax]):.4e}  "
                  f"stamp_center_value={float(stamp[cen, cen]):.4e}  "
                  f"stamp_max={float(np.nanmax(stamp)):.4e}  "
                  f"sky={sky:.4e}  aperture_total={total:.4e}")

        # Pan-STARRS1 templates (130 arcsec coadds) have background/PSF
        # characteristics that don't match the science instruments and were
        # producing noise-dominated, pixellated morphology initializations.
        # Exclude PS1 from the morphology stamp median entirely -- it still
        # contributes to flux_by_band below for spectrum estimation (used
        # only as a cross-band fallback per the non-PS1 median logic).
        if row["telescope"] != "Pan-STARRS1":
            stamps.append(source / total)
        flux_by_band[band].append((row["telescope"], total))

    n_used = len(stamps)
    n_total_passed = sum(len(v) for v in flux_by_band.values())
    print(f"  static initializer ({coord.ra.deg:.4f}, {coord.dec.deg:.4f}): "
          f"{n_used}/{len(all_observations)} frames used for morphology "
          f"(PS1 excluded), {n_total_passed} passed SNR>{min_snr:.1f} cut overall")

    if stamps:
        morph = np.nanmedian(np.stack(stamps), axis=0).astype("float32")
        morph[morph < 1e-8] = 1e-8
        morph /= np.sum(morph)
    else:
        print(f"  WARNING: no usable stamps found, falling back to Gaussian morphology")
        morph = gaussian_morphology(stamp_size, fallback_sigma)

    # Build per-band non-PS1 medians first; PS1 bands will use these instead
    # of their own (unreliable) aperture flux estimates.
    non_ps1_by_band: dict[str, list[float]] = defaultdict(list)
    for band, entries in flux_by_band.items():
        for telescope, total in entries:
            if telescope != "Pan-STARRS1":
                non_ps1_by_band[band].append(total)

    all_non_ps1 = [v for vals in non_ps1_by_band.values() for v in vals]
    global_non_ps1_median = float(np.nanmedian(all_non_ps1)) if all_non_ps1 else None

    spectrum = []
    fallback = np.nanmedian([total for entries in flux_by_band.values() for _, total in entries]) if flux_by_band else 1e-3
    fallback = float(fallback) if np.isfinite(fallback) and fallback > 0 else 1e-3
    for band in bands:
        entries = flux_by_band.get(band, [])
        non_ps1_vals = np.asarray(non_ps1_by_band.get(band, []), dtype=float)
        non_ps1_vals = non_ps1_vals[np.isfinite(non_ps1_vals) & (non_ps1_vals > 0)]
        ps1_vals = [total for tel, total in entries if tel == "Pan-STARRS1"]
        print(f"  band {band}: non_ps1_vals={[f'{v:.4e}' for v in non_ps1_vals]}  "
              f"ps1_vals={[f'{v:.4e}' for v in ps1_vals]}")

        if non_ps1_vals.size > 0:
            # Use non-PS1 median directly for this band.
            band_val = float(np.nanmedian(non_ps1_vals))
        elif global_non_ps1_median is not None:
            # No non-PS1 data for this band; use the cross-band non-PS1 median.
            band_val = global_non_ps1_median
            print(f"  band {band}: no non-PS1 flux; using cross-band median {band_val:.4f}")
        else:
            # No non-PS1 data at all; fall back to all-telescope median.
            all_vals = np.asarray([total for _, total in entries], dtype=float)
            all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
            band_val = float(np.nanmedian(all_vals)) if all_vals.size else fallback

        spectrum.append(band_val)
    return np.asarray(spectrum, dtype="float32"), morph




def aperture_initial_fluxes(observations, rows: list[dict], channels: list[tuple[str, int]], coord: SkyCoord) -> np.ndarray:
    """Initialize transient epoch fluxes with small positive aperture sums.

    ``scarlet2.init.pixel_spectrum`` is convenient but can underestimate point
    sources when the image is undersampled or the WCS alignment is imperfect.
    A compact aperture sum gives the optimizer a more realistic starting point,
    while reference channels are still forced near zero later in ``build_scene``.
    """

    fluxes = []
    for obs, row in zip(observations, rows, strict=True):
        image = np.asarray(obs.data[0], dtype="float32")
        yx = np.asarray(obs.frame.get_pixel(coord), dtype=float)
        fwhm_pix = float(row.get("psf_fwhm_arcsec", np.nan)) / max(float(row.get("pixel_scale_arcsec", np.nan)), 1e-6)
        radius = int(max(2, min(8, round(0.75 * fwhm_pix if np.isfinite(fwhm_pix) else 3))))
        stamp = cut_square(image, yx, 2 * radius + 5)
        yy, xx = np.mgrid[: stamp.shape[0], : stamp.shape[1]]
        cen = (stamp.shape[0] - 1) / 2.0
        rr = np.hypot(yy - cen, xx - cen)
        aperture = rr <= radius
        annulus = (rr > radius + 1) & (rr <= radius + 3)
        sky = float(np.nanmedian(stamp[annulus])) if np.any(annulus) else 0.0
        flux = float(np.nansum(stamp[aperture] - sky))
        fluxes.append(flux if np.isfinite(flux) and flux > 0 else np.nan)

    fluxes = np.asarray(fluxes, dtype="float32")
    if np.all(~np.isfinite(fluxes)):
        fluxes[:] = 1e-3
    else:
        fallback = float(np.nanmedian(fluxes[np.isfinite(fluxes) & (fluxes > 0)]))
        fluxes[~np.isfinite(fluxes) | (fluxes <= 0)] = fallback if fallback > 0 else 1e-3
    return fluxes


def _p60_centering_ok(row: dict, threshold_fraction: float = 0.25) -> bool:
    """Return True if the brightest pixel in a P60 stamp is near the centre.

    Reads the image, cuts a stamp around the expected source position (from the
    manifest scene_center), and checks whether the peak pixel is within
    ``threshold_fraction`` of the half-stamp width from the centre.  Frames
    where the source has wandered outside this radius are dropped.

    Parameters
    ----------
    row : dict
        Manifest row for a P60 science frame.
    threshold_fraction : float
        Maximum allowed fractional offset from the stamp centre (default 0.25,
        i.e. 25% of the half-stamp width).
    """
    try:
        path = Path(row["path"])
        if not path.exists():
            return True  # missing file will fail later with a clearer error
        ext = row.get("ext", 0)
        data, header = read_hdu(path, ext)
        wcs = WCS(header).celestial
        center = SkyCoord(row["scene_ra_deg"], row["scene_dec_deg"], unit="deg")
        size_arcsec = float(row.get("stamp_arcsec", 30.0))
        size = (size_arcsec * u.arcsec, size_arcsec * u.arcsec)
        cutout = Cutout2D(data, center, size, wcs=wcs, mode="partial", fill_value=np.nan)
        stamp = np.nan_to_num(cutout.data, nan=0.0)
        h, w = stamp.shape
        if h < 4 or w < 4:
            return True
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        peak_y, peak_x = np.unravel_index(np.argmax(stamp), stamp.shape)
        dy = abs(peak_y - cy) / (h / 2.0)
        dx = abs(peak_x - cx) / (w / 2.0)
        ok = dy <= threshold_fraction and dx <= threshold_fraction
        if not ok:
            fname = path.name
            print(f"  P60 centering REJECT: {fname}  peak offset ({dy:.2f}, {dx:.2f}) > {threshold_fraction:.2f}")
        return ok
    except Exception as exc:
        # If we can't read the file, let it through and fail later with context.
        print(f"  P60 centering check failed for {row.get('path','?')}: {exc}")
        return True


def select_rows(
    payload: dict,
    telescopes: set[str] | None,
    bands: set[str] | None,
    max_images: int | None,
    quality_top_per_band_telescope: int | None,
    exclude_telescopes: set[str] | None = None,
    skip_mjds: set[float] | None = None,
) -> list[dict]:
    """Select references plus a science subset from the manifest.

    References are always included for requested bands unless their telescope
    appears in ``exclude_telescopes``, which applies to both science and
    reference rows unconditionally. Science rows are additionally filtered by
    ``telescopes`` (allowlist) and ``bands``. If
    ``quality_top_per_band_telescope`` is provided, each telescope/band group
    keeps the frames with the smallest heuristic score based on background RMS,
    PSF FWHM, zeropoint scatter and calibration-star count. This is useful for
    fast smoke tests and for LT, where the full joint fit can be large.
    """

    selected = []
    for row in payload["observations"]:
        if bands and row["band"] not in bands:
            continue
        if exclude_telescopes and row["telescope"] in exclude_telescopes:
            continue
        if telescopes and row["telescope"] not in telescopes:
            continue
        if skip_mjds and any(abs(float(row.get("mjd", float("nan"))) - m) < 0.1 for m in skip_mjds):
            print(f"  Skipping frame MJD={row.get('mjd'):.2f} {Path(row['path']).name} (--skip-frames)")
            continue
        # For P60, reject frames where the source peak is far from the stamp
        # centre (bad telescope pointing / guide-star failure).
        if row.get("telescope") == "P60" and row.get("kind") == "science":
            if not _p60_centering_ok(row, threshold_fraction=0.25):
                continue
        selected.append(row)

    references = [row for row in selected if row["kind"] == "reference"]
    science = [row for row in selected if row["kind"] == "science"]

    def score(row: dict) -> float:
        rms = float(row.get("background_rms", np.inf))
        fwhm = float(row.get("psf_fwhm_arcsec", np.inf))
        zp_scatter = float(row.get("zp_scatter", 1.0))
        nstars = float(row.get("n_cal_stars_fit", 1.0))
        terms = [
            math.log10(max(rms, 1e-6)),
            0.35 * max(fwhm, 0.0),
            2.0 * max(zp_scatter, 0.0),
            -0.05 * min(max(nstars, 0.0), 30.0),
        ]
        return float(np.nansum(terms))

    if quality_top_per_band_telescope:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in science:
            grouped[(row["telescope"], row["band"])].append(row)
        science = []
        for rows in grouped.values():
            science.extend(sorted(rows, key=score)[:quality_top_per_band_telescope])

    science = sorted(science, key=lambda r: (r["telescope"], r["band"], r["mjd"], Path(r["path"]).name))
    if max_images is not None:
        science = science[:max_images]
    return references + science


def register_astrometric_shifts(corr_obs: list, model_frame, stepsize_arcsec: float = 1.6e-2) -> int:
    """Register a per-observation astrometric shift parameter for every observation.

    Follows the approach in the example notebook: call ``check_set_renderer`` to
    ensure the renderer exists, then access ``obs.renderer[1].shift`` directly.
    This is the pattern that works reliably with scarlet2 >= 0.5 where the
    renderer is a two-element composite (resampler + shifter).

    The default stepsize of 1.6e-2 arcsec is 1/5 of the notebook's 8e-2 arcsec
    value, appropriate for this real-data field where WCS residuals are modest.
    Increase toward 8e-2 if the optimizer is slow to converge on large shifts;
    decrease further if shifts are oscillating.

    Returns the number of observations for which a shift parameter was registered.
    """

    import scarlet2

    n_registered = 0
    for idx, obs in enumerate(corr_obs):
        obs.check_set_renderer(model_frame)
        try:
            shift = obs.renderer[1].shift
        except (IndexError, AttributeError, TypeError):
            # Renderer layout differs in this scarlet2 build -- skip silently.
            continue
        with scarlet2.Parameters(obs):
            scarlet2.Parameter(
                shift,
                name=f"shift_{idx}",
                stepsize=stepsize_arcsec * u.arcsec,
            )
        n_registered += 1
    return n_registered


def wavelet_static_initializer(
    corr_obs: list,
    rows: list[dict],
    static_coords: list,
    static_labels: list[str],
    bands: list[str],
    stamp_size: int,
    min_snr: float = 3.0,
) -> list[tuple]:
    """Initialize static source spectra and morphologies using Starlet wavelets.

    Rather than taking median stamps per source, this function:

    1. Builds a detection image by summing all reference observations after
       normalising each by its stamp sum.
    2. Decomposes the detection image with a Starlet wavelet transform.
    3. Initialises the **lens** (compact) from the inner wavelet scales (1-2)
       and the **host** (extended) from the outer scales (2-3), ensuring the
       two sources start with genuinely distinct morphologies even though they
       overlap spatially.
    4. Derives spectra by aperture-summing each source's wavelet morphology
       stamp against each observation, using the same non-PS1 median logic as
       the stamp-based initializer.

    Returns a list of ``(spectrum_array, morphology_array)`` tuples in the
    same order as ``static_coords``.
    """

    import scarlet2

    # ── Select best Gemini science frame for wavelet detection ───────────────
    # Using a single high-quality science image rather than summed references
    # gives a sharper detection image and tighter source boxes.
    # "Best" = lowest psf_fwhm_arcsec among Gemini science rows; fall back to
    # any science row, then any reference row if no Gemini frames are present.
    gemini_science = [
        (i, row) for i, row in enumerate(rows)
        if row["telescope"] == "Gemini-North" and row["kind"] == "science"
        and np.isfinite(float(row.get("psf_fwhm_arcsec", np.nan)))
    ]
    if gemini_science:
        best_idx, best_row = min(gemini_science, key=lambda x: float(x[1]["psf_fwhm_arcsec"]))
        print(f"  wavelet_static_initializer: using best Gemini frame "
              f"{Path(best_row['path']).name} "
              f"(FWHM={best_row['psf_fwhm_arcsec']:.3f} arcsec) for detection")
        detect_obs = corr_obs[best_idx]
    else:
        # Fall back to first science frame, then first reference
        science_idx = next((i for i, r in enumerate(rows) if r["kind"] == "science"), None)
        fallback_idx = science_idx if science_idx is not None else 0
        best_row = rows[fallback_idx]
        detect_obs = corr_obs[fallback_idx]
        print(f"  wavelet_static_initializer: no Gemini frame found, "
              f"falling back to {Path(best_row['path']).name}")

    detect_image = np.asarray(detect_obs.data[0], dtype="float32")
    # Normalise so Starlet scale thresholds are data-independent
    img_max = float(np.nanmax(detect_image))
    if img_max > 0:
        detect_image = detect_image / img_max

    # ── Starlet decomposition per source on a tight stamp ────────────────────
    # Run Starlet on a stamp cut around each source position rather than the
    # full detection image — this prevents wavelet coefficients from the
    # surrounding field bleeding into the morphology and inflating the box size.
    import jax.numpy as jnp

    morph_by_label = {}
    for label, coord in zip(static_labels, static_coords):
        # Get pixel position of this source in the detection observation frame.
        yx = np.asarray(detect_obs.frame.get_pixel(coord), dtype=float)
        stamp = cut_square(detect_image, yx, stamp_size).astype("float32")

        # Run Starlet on the stamp.
        starlet = scarlet2.Starlet.from_image(stamp)
        coeffs = np.asarray(starlet.coefficients)  # (n_scales, H, W)
        n_scales = coeffs.shape[0]
        print(f"  wavelet_static_initializer [{label}]: stamp shape={stamp.shape}, "
              f"n_scales={n_scales}")

        if label == static_labels[0]:
            # lens (compact): inner scales only
            scale_slice = slice(0, min(2, n_scales - 1))
        else:
            # host (extended): outer scales
            scale_slice = slice(max(1, 1), min(3, n_scales - 1))

        morph = np.clip(coeffs[scale_slice].sum(axis=0), 0, None).astype("float32")
        morph[morph < 1e-8] = 1e-8
        morph /= morph.sum()
        morph_by_label[label] = morph

    # Decorrelate: if the two morphologies overlap heavily, subtract some of
    # the compact from the extended so they don't start as near-identical.
    if len(static_labels) == 2:
        m0 = morph_by_label[static_labels[0]]
        m1 = morph_by_label[static_labels[1]]
        overlap_frac = float(np.minimum(m0, m1).sum())
        if overlap_frac > 0.3:
            m1 = np.clip(m1 - 0.5 * m0, 0, None)
            s = m1.sum()
            m1 = m1 / s if s > 0 else m1
            morph_by_label[static_labels[1]] = m1
            print(f"  wavelet_static_initializer: decorrelated extended morph "
                  f"(overlap was {overlap_frac:.2f})")

    # ── Per-source spectra from aperture sums ─────────────────────────────────
    results = []
    for label, coord in zip(static_labels, static_coords):
        morph = morph_by_label[label]
        flux_by_band: dict[str, list[tuple[str, float]]] = defaultdict(list)
        half = stamp_size // 2

        for obs, row in zip(corr_obs, rows, strict=True):
            band = row["band"]
            image = np.asarray(obs.data[0], dtype="float32")
            yx = np.asarray(obs.frame.get_pixel(coord), dtype=float)
            stamp = cut_square(image, yx, stamp_size)
            finite = np.isfinite(stamp)
            if finite.sum() < stamp_size:
                continue

            yy, xx = np.mgrid[:stamp_size, :stamp_size]
            cen = (stamp_size - 1) / 2.0
            rr = np.hypot(yy - cen, xx - cen)
            annulus = (rr > half * 0.65) & (rr <= half * 0.90) & finite
            sky = float(np.nanmedian(stamp[annulus])) if annulus.any() else 0.0
            sky_rms = float(np.nanstd(stamp[annulus])) if annulus.sum() > 1 else np.nan

            source = np.nan_to_num(stamp - sky, nan=0.0, posinf=0.0, neginf=0.0)
            source[source < 0] = 0.0

            if np.isfinite(sky_rms) and sky_rms > 0:
                if float(np.nanmax(source)) / sky_rms < min_snr:
                    continue
            total = float(np.sum(source))
            if total <= 0 or not np.isfinite(total):
                continue
            flux_by_band[band].append((row["telescope"], total))

        # Non-PS1 spectrum logic (same as reference_static_initializer)
        non_ps1_by_band: dict[str, list[float]] = defaultdict(list)
        for band, entries in flux_by_band.items():
            for telescope, total in entries:
                if telescope != "Pan-STARRS1":
                    non_ps1_by_band[band].append(total)
        all_non_ps1 = [v for vals in non_ps1_by_band.values() for v in vals]
        global_non_ps1_median = float(np.nanmedian(all_non_ps1)) if all_non_ps1 else None
        fallback = float(np.nanmedian(all_non_ps1)) if all_non_ps1 else 1e-3
        fallback = fallback if np.isfinite(fallback) and fallback > 0 else 1e-3

        spectrum = []
        for band in bands:
            non_ps1_vals = np.asarray(non_ps1_by_band.get(band, []), dtype=float)
            non_ps1_vals = non_ps1_vals[np.isfinite(non_ps1_vals) & (non_ps1_vals > 0)]
            if non_ps1_vals.size > 0:
                band_val = float(np.nanmedian(non_ps1_vals))
            elif global_non_ps1_median is not None:
                band_val = global_non_ps1_median
            else:
                all_vals_raw = [t for _, t in flux_by_band.get(band, [])]
                all_vals = np.asarray(all_vals_raw, dtype=float)
                all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
                band_val = float(np.nanmedian(all_vals)) if all_vals.size else fallback
            spectrum.append(band_val)

        spectrum_arr = np.asarray(spectrum, dtype="float32")
        print(f"  wavelet_static_initializer [{label}]: "
              f"morph_sum={morph.sum():.4f}, spectrum={spectrum_arr}")
        results.append((spectrum_arr, morph))

    return results


def subtract_gaussian_at(
    morph: np.ndarray,
    center_yx: tuple[float, float],
    sigma: float,
    amplitude_fraction: float = 0.2,
) -> np.ndarray:
    """Subtract a scaled Gaussian profile from ``morph`` at ``center_yx``.

    The Gaussian is scaled so its peak equals ``amplitude_fraction`` times
    ``morph``'s value at the nearest pixel to ``center_yx``, then subtracted.
    The result is clipped at zero and renormalised to unit sum.

    A Gaussian with sigma in pixels has total volume ``peak * 2*pi*sigma^2``;
    for sigma=8px that's ~400x the peak value, which can vastly exceed the
    total flux in a unit-sum stamp and over-subtract almost everything.
    ``amplitude_fraction`` (default 0.3) tempers this so only a modest,
    partial central component is removed -- enough to stop the host
    morphology re-modelling the lens core, without wiping out the
    surrounding structure the host needs to initialise from.

    Used to remove (most of) the central lens flux from the host stamp before
    Starlet decomposition.
    """

    h, w = morph.shape
    yy, xx = np.mgrid[:h, :w].astype("float32")
    cy, cx = center_yx
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    gauss = np.exp(-0.5 * rr2 / sigma**2).astype("float32")

    iy = int(np.clip(round(cy), 0, h - 1))
    ix = int(np.clip(round(cx), 0, w - 1))
    peak = float(morph[iy, ix])
    if peak <= 0 or not np.isfinite(peak):
        return morph

    subtracted = morph - gauss * peak * amplitude_fraction
    subtracted = np.clip(subtracted, 0.0, None)
    subtracted[subtracted < 1e-8] = 1e-8
    subtracted /= subtracted.sum()
    return subtracted.astype("float32")


def _get_obs_psf_array(obs) -> np.ndarray | None:
    """Best-effort retrieval of an Observation's PSF as a 2-D array.

    Different scarlet2 versions expose the per-observation PSF under
    different attribute names/calling conventions. Try several and return
    None if none work (diagnostic-only; never raises).
    """

    candidates = []
    for attr in ("psf", "_psf"):
        obj = getattr(obs, attr, None)
        if obj is not None:
            candidates.append(obj)
    frame = getattr(obs, "frame", None)
    if frame is not None:
        for attr in ("psf", "_psf"):
            obj = getattr(frame, attr, None)
            if obj is not None:
                candidates.append(obj)

    for obj in candidates:
        for getter in (lambda o: o(), lambda o: o):
            try:
                arr = np.asarray(getter(obj), dtype="float64")
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.ndim == 2:
                    return arr
            except Exception:
                continue
    return None


def gaussian_sigma_from_psf(psf_2d: np.ndarray) -> float:
    """Estimate the Gaussian sigma (in pixels) of a normalised 2-D PSF via
    second moments.

    ``sigma = sqrt((sigma_y^2 + sigma_x^2) / 2)`` using intensity-weighted
    second moments about the centroid. Assumes ``psf_2d`` sums to ~1 and is
    non-negative.
    """

    psf = np.clip(np.asarray(psf_2d, dtype="float64"), 0.0, None)
    total = psf.sum()
    if total <= 0:
        return np.nan
    psf = psf / total

    h, w = psf.shape
    yy, xx = np.mgrid[:h, :w].astype("float64")
    y0 = float((psf * yy).sum())
    x0 = float((psf * xx).sum())
    var_y = float((psf * (yy - y0) ** 2).sum())
    var_x = float((psf * (xx - x0) ** 2).sum())
    return float(np.sqrt(max(0.0, (var_y + var_x) / 2.0)))


def build_scene(
    observations,
    rows: list[dict],
    channels: list[tuple[str, int]],
    science_epoch_ids: set[int],
    bands: list[str],
    static_stamp_size: int,
    fit_centers: bool,
    fit_astrometric_shifts: bool,
    shift_stepsize: float = 1.6e-2,
    static_init_min_snr: float = 3.0,
    morphology_init_path: Path | None = None,
    morph_stepsize_absolute: float | None = None,
    morph_stepsize_relative_factor: float = 2e-4,
):
    """Build and parameterize the scarlet2 scene.

    The static components are initialized from reference images only. The
    transient components are point sources at the measured lensed-image
    coordinates with one independent flux parameter per science epoch. By
    default their centers are fixed, which is usually the safer choice for a
    lensed SN field: otherwise a faint component can walk toward residual host
    structure or steal flux from image A.

    If ``morphology_init_path`` is given, it must point to an ``.npz`` file
    (as written by ``fit_reference_morphology.py``) containing pre-fitted
    Starlet coefficients for the lens and host, keyed ``lens_g1_coeffs`` and
    ``host_g2_coeffs``. These are used directly as the initial Starlet
    coefficients instead of the Gaussian/wavelet initializers, and
    ``morph_stepsize_absolute`` (if given) is used as a fixed absolute
    stepsize for the morphology Parameter instead of ``relative_step`` -- this
    is intended for small refinement steps around an already-good reference
    morphology rather than free initialization from scratch.
    """

    import jax.numpy as jnp
    import equinox as eqx
    import scarlet2
    from numpyro.distributions import constraints

    morphology_init = None
    if morphology_init_path is not None:
        morphology_init = dict(np.load(morphology_init_path))
        print(f"Loaded reference morphology init from {morphology_init_path}: "
              f"keys={list(morphology_init.keys())}")

    band_selector = lambda ch: ch[0]
    epoch_selector = lambda ch: ch[1]

    # Split observations into two groups:
    # - correlated (non-PS1): will be wrapped in CorrelatedObservation and
    #   pre-resampled onto the model frame
    # - native (PS1): will be matched directly to the model frame as plain
    #   Observations without pre-resampling (CorrelatedObservation.from_observation
    #   produces a severe resampling malfunction for PS1 -- flux loss ~2e8 and
    #   weights blown to 2e20)
    # The model frame is built from ALL observations (both groups) so its WCS
    # and PSF envelope properly covers PS1's footprint too.
    ps1_obs = [obs for obs, row in zip(observations, rows) if row.get("telescope") == "Pan-STARRS1"]
    other_obs = [obs for obs, row in zip(observations, rows) if row.get("telescope") != "Pan-STARRS1"]

    # Build the model frame from ALL native observations (before any resampling).
    # Frame.from_observations selects the finest pixel scale as the model
    # resolution. Using corr_other (CorrelatedObservations without
    # resample_to_frame) averaged pixel scales and produced a wrong 0.162"/px
    # model frame instead of Gemini's native 0.0807"/px.
    model_frame = scarlet2.Frame.from_observations(
        observations=observations,
        coverage="union",
    )
    try:
        mf_pixscale = float(np.mean(proj_plane_pixel_scales(model_frame.wcs)) * 3600.0)
        print(f"  model_frame pixel scale: {mf_pixscale:.6f} arcsec/px  "
              f"shape={model_frame.bbox.shape}")
    except Exception as exc:
        print(f"  model_frame pixel scale diagnostic failed: {exc}")
    print(f"  model_frame channels: {model_frame.channels}")
    for ps1_o, row in [(o, r) for o, r in zip(observations, rows) if r.get("telescope") == "Pan-STARRS1"]:
        print(f"  PS1 [{row['band']}] obs channels: {ps1_o.frame.channels}  "
              f"model_frame has band: {row['band'] in [ch if isinstance(ch, str) else ch[0] for ch in model_frame.channels]}")

    # Now match everything to the unified model frame.
    corr_obs = []
    for obs, row in zip(observations, rows, strict=True):
        if row.get("telescope") == "Pan-STARRS1":
            # PS1: use plain Observation matched directly to model frame.
            # CorrelatedObservation.from_observation with resample_to_frame
            # produces a severe resampling malfunction for PS1 (flux loss ~2e8,
            # weights ~2e20). Instead, match the native Observation directly --
            # scarlet2 sets up an on-the-fly renderer via .match().
            obs.match(model_frame)
            obs_corr = obs
            corr_obs.append(obs_corr)
        else:
            obs_corr = scarlet2.CorrelatedObservation.from_observation(
                obs,
                resample_to_frame=model_frame,
                resample_psf=True,
            )
            obs_corr.match(model_frame)
            corr_obs.append(obs_corr)

        if row.get("telescope") == "Pan-STARRS1":
            matched = corr_obs[-1]  # the PS1 obs just appended
            native = np.asarray(obs.data[0], dtype="float64")
            native_w = np.asarray(obs.weights[0], dtype="float64")
            native_psf = _get_obs_psf_array(obs)
            resampled = np.asarray(matched.data[0], dtype="float64")
            resampled_w = np.asarray(matched.weights[0], dtype="float64")
            resampled_psf = _get_obs_psf_array(matched)

            native_pos = native[native > 0]
            resampled_pos = resampled[resampled > 0]
            print(f"  RESAMPLE_CHECK [Pan-STARRS1 {row['band']} MJD={row.get('mjd',0):.1f}]: "
                  f"native shape={native.shape} sum={native.sum():.4e} "
                  f"max={native.max():.4e} pos_sum={float(native_pos.sum()) if native_pos.size else 0:.4e} | "
                  f"resampled shape={resampled.shape} sum={resampled.sum():.4e} "
                  f"max={resampled.max():.4e} pos_sum={float(resampled_pos.sum()) if resampled_pos.size else 0:.4e}")

            # NaN/Inf checks on every array involved in the resampling.
            def _nan_report(name, arr):
                if arr is None:
                    print(f"    NAN_CHECK {name}: array unavailable")
                    return
                n_nan = int(np.isnan(arr).sum())
                n_inf = int(np.isinf(arr).sum())
                print(f"    NAN_CHECK {name}: shape={arr.shape} n_nan={n_nan} n_inf={n_inf} "
                      f"min={np.nanmin(arr):.4e} max={np.nanmax(arr):.4e}")

            _nan_report("native_data", native)
            _nan_report("native_weights", native_w)
            _nan_report("native_psf", native_psf)
            _nan_report("resampled_data", resampled)
            _nan_report("resampled_weights", resampled_w)
            _nan_report("resampled_psf", resampled_psf)

    # Initial Gaussian sigma (pixels) used both for the lens's parametric
    # profile and for the central subtraction applied to the host stamp.
    LENS_SIGMA_PIX = 4.0

    with scarlet2.Scene(model_frame) as scene:
        for idx, (label, coord) in enumerate(zip(STATIC_LABELS, STATIC_COORDS, strict=True)):
            if label == "lens_g1":
                source_stamp_size = static_stamp_size
            else:
                # Host: use the same stamp size and the same centre as the
                # lens, since it models an extended/ring-like shape around
                # the lens rather than a separate galaxy at its own position.
                source_stamp_size = static_stamp_size
                coord = STATIC_COORDS[0]

            spectrum_data, morph = reference_static_initializer(
                corr_obs, rows, coord, bands, source_stamp_size,
                fallback_sigma=LENS_SIGMA_PIX if label == "lens_g1" else 12.0,
                min_snr=static_init_min_snr,
            )
            print(f"  [{label}] spectrum_data by band: "
                  + ", ".join(f"{b}={v:.4e}" for b, v in zip(bands, spectrum_data, strict=True)))

            coeffs_key = f"{label}_coeffs"
            if morphology_init is not None and coeffs_key in morphology_init:
                # Use the pre-fitted reference morphology directly. We still
                # need a StarletMorphology object of the right shape; the
                # simplest way is to build one from a placeholder image of the
                # saved shape, then overwrite its coefficients via eqx.tree_at.
                ref_coeffs = jnp.asarray(morphology_init[coeffs_key])
                placeholder = np.zeros(ref_coeffs.shape[-2:], dtype="float32")
                placeholder[placeholder.shape[0] // 2, placeholder.shape[1] // 2] = 1.0
                morphology = scarlet2.StarletMorphology.from_image(jnp.asarray(placeholder))
                morphology = eqx.tree_at(lambda m: m.coeffs, morphology, ref_coeffs)
                print(f"  [{label}] morphology initialised from reference fit "
                      f"({morphology_init_path.name}), coeffs shape={ref_coeffs.shape}")
            elif label == "lens_g1":
                # Lens: initialise from a smooth Gaussian profile (rather than
                # the noisy stamp median), but still fit as a StarletMorphology
                # so the optimizer can refine substructure away from the
                # initial Gaussian if the data require it. The same
                # scale-dependent Cauchy prior as the host applies (registered
                # below in the shared StarletMorphology branch).
                gaussian_init = gaussian_morphology(morph.shape[0], LENS_SIGMA_PIX)
                morphology = scarlet2.StarletMorphology.from_image(jnp.asarray(gaussian_init))
            else:
                # Host: subtract a Gaussian matching the lens profile at the
                # stamp centre (since the stamp is now centred on the lens
                # coordinate too) before Starlet decomposition.
                stamp_center = (source_stamp_size - 1) / 2.0
                lens_pos_in_stamp = (stamp_center, stamp_center)
                morph = subtract_gaussian_at(morph, lens_pos_in_stamp, LENS_SIGMA_PIX)
                print(f"  [{label}] center set to lens coordinate; subtracted central "
                      f"Gaussian (sigma={LENS_SIGMA_PIX} px) at stamp centre "
                      f"({lens_pos_in_stamp[0]:.1f}, {lens_pos_in_stamp[1]:.1f})")
                morphology = scarlet2.StarletMorphology.from_image(jnp.asarray(morph))

            spectrum = scarlet2.StaticArraySpectrum(
                jnp.asarray(spectrum_data),
                bands=bands,
                band_selector=band_selector,
            )

            scarlet2.Source(coord, spectrum, morphology)

        for label, coord in zip(SOURCE_LABELS, SOURCE_COORDS, strict=True):
            initial_flux = aperture_initial_fluxes(corr_obs, rows, channels, coord)
            finite_positive = initial_flux[np.isfinite(initial_flux) & (initial_flux > 0)]
            fallback = float(np.nanmedian(finite_positive)) if len(finite_positive) else 1e-3
            for idx, channel in enumerate(channels):
                if channel[1] not in science_epoch_ids:
                    # Hard-zero reference epochs: TransientArraySpectrum will
                    # keep these fixed at zero via the epochs= allowlist.
                    initial_flux[idx] = 0.0
                elif not np.isfinite(initial_flux[idx]) or initial_flux[idx] <= 0:
                    initial_flux[idx] = fallback
            # epochs= defines which epoch IDs are free to vary; all others are
            # forced to zero (pre-explosion / reference frames).
            spectrum = scarlet2.TransientArraySpectrum(
                jnp.asarray(initial_flux),
                epochs=list(science_epoch_ids),
                epoch_selector=epoch_selector,
            )
            scarlet2.PointSource(coord, spectrum)

    with scarlet2.Parameters(scene):
        for idx, source in enumerate(scene.sources):
            if isinstance(source, scarlet2.PointSource):
                # Transient point source: no positivity constraint on spectrum
                # because reference epochs are forced to zero and the flux can
                # fluctuate freely around zero during fitting (per the scarlet2
                # transient tutorial).
                scarlet2.Parameter(
                    source.spectrum.data,
                    name=f"spectrum.{idx}",
                    stepsize=lambda p: scarlet2.relative_step(p, factor=1e-3),
                )
            else:
                # Static extended source (lens / host): must stay positive.
                scarlet2.Parameter(
                    source.spectrum.data,
                    name=f"spectrum.{idx}",
                    stepsize=lambda p: scarlet2.relative_step(p, factor=1e-3),
                    constraint=constraints.positive,
                )
            if hasattr(source, "morphology") and not isinstance(source, scarlet2.PointSource):
                if isinstance(source.morphology, scarlet2.SersicMorphology):
                    # Lens: parametric Gaussian via Sersic with n fixed at 0.5
                    # (a true Gaussian profile). n is NOT registered as a free
                    # parameter, so the lens stays a single smooth Gaussian
                    # galaxy and cannot develop Sersic-index substructure.
                    # size and ellipticity are free so the lens can adjust its
                    # width and shape to the data.
                    scarlet2.Parameter(
                        source.morphology.size,
                        name=f"morphology.{idx}.size",
                        constraint=constraints.positive,
                        stepsize=0.1,
                    )
                    scarlet2.Parameter(
                        source.morphology.ellipticity,
                        name=f"morphology.{idx}.eps",
                        constraint=scarlet2.constraint.unit_disk,
                        stepsize=1e-2,
                    )
                else:
                    import numpyro.distributions as dist
                    coeffs = source.morphology.coeffs
                    n_scales = coeffs.shape[0]
                    # ── Best configuration (as of 2026-06) ────────────────────────
                    # Validated on Gemini single-band (gri) fits for SN2025wny.
                    # StarletMorphology with scale-dependent Cauchy prior:
                    #   fine scale 0:        1e-5  (very tight)
                    #   intermediate 1..n-2: 1e-5  (very tight, suppresses noise)
                    #   coarsest n-1:        1.0   (loose, allows extended emission)
                    # Spectrum + morphology stepsize: factor=1e-3 / 2e-4
                    # Astrometric shift stepsize: 1.6e-1 arcsec
                    # Max iterations: 5000
                    # Adding Liverpool/P60 with these settings causes correlated-noise
                    # ripples in the extended morphology due to resampling artefacts.
                    # ──────────────────────────────────────────────────────────────
                    scale_list = []
                    for s in range(n_scales):
                        if s == 0:
                            scale_list.append(1e-6)       # finest: extremely tight
                        elif s == n_scales - 1:
                            scale_list.append(1.0)        # coarsest: loose
                        else:
                            scale_list.append(1e-5)       # intermediate: very tight
                    scale_values = jnp.array(scale_list, dtype=jnp.float32)
                    stdev = scale_values[:, None, None] * jnp.ones_like(coeffs)
                    prior = dist.Cauchy(scale=stdev).to_event(coeffs.ndim)
                    if morph_stepsize_absolute is not None:
                        # Small, fixed absolute stepsize for refining around an
                        # already-good reference morphology (rather than
                        # relative_step, which scales with the current
                        # coefficient value and can take large jumps away from
                        # a good initialization).
                        morph_step = morph_stepsize_absolute
                        print(f"  [morphology.{idx}] using absolute stepsize={morph_stepsize_absolute:.1e}")
                    else:
                        morph_step = lambda p: scarlet2.relative_step(p, factor=morph_stepsize_relative_factor)
                        print(f"  [morphology.{idx}] using relative stepsize factor={morph_stepsize_relative_factor:.1e}")
                    scarlet2.Parameter(
                        coeffs,
                        name=f"morphology.{idx}",
                        stepsize=morph_step,
                        prior=prior,
                    )
            if fit_centers and isinstance(source, scarlet2.PointSource):
                scarlet2.Parameter(source.center, name=f"center.{idx}", stepsize=5e-3)

    if fit_astrometric_shifts:
        n = register_astrometric_shifts(corr_obs, model_frame, stepsize_arcsec=shift_stepsize)
        print(f"Registered astrometric shift parameters for {n}/{len(corr_obs)} observations.")

    # NaN check on the rendered model and log-likelihood for every PS1
    # observation, using the initial (pre-fit) scene.
    try:
        model = scene()
        for idx, (obs_corr, row) in enumerate(zip(corr_obs, rows, strict=True)):
            try:
                rendered = np.asarray(obs_corr.render(model), dtype="float64")
                n_nan_r = int(np.isnan(rendered).sum())
                n_inf_r = int(np.isinf(rendered).sum())
                print(f"    RENDER_CHECK [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
                      f"rendered shape={rendered.shape} n_nan={n_nan_r} n_inf={n_inf_r} "
                      f"min={np.nanmin(rendered):.4e} max={np.nanmax(rendered):.4e} sum={np.nansum(rendered):.4e}")
            except Exception as exc:
                print(f"    RENDER_CHECK [{row['telescope']} {row['band']}]: render failed ({exc})")
            try:
                logl = float(obs_corr.log_likelihood(model))
                if np.isnan(logl) or np.isinf(logl):
                    print(f"    LOGL_CHECK [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
                          f"log_likelihood={logl:.4e} *** BAD ***")
            except Exception as exc:
                print(f"    LOGL_CHECK [{row['telescope']} {row['band']}]: log_likelihood failed ({exc})")
    except Exception as exc:
        print(f"  RENDER_CHECK: scene() evaluation failed ({exc})")

    return scene, corr_obs, model_frame


def write_flux_csv(
    scene,
    channels: list[tuple[str, int]],
    rows: list[dict],
    scale_factors: list[float],
    output_path: Path,
) -> None:
    """Write transient-source fluxes measured from the fitted scarlet2 model.

    The model was fitted on normalized images (each divided by its stamp sum).
    Fluxes are rescaled back to physical nanomaggies by multiplying by the
    per-observation ``scale_factors`` before computing AB magnitudes.
    """

    import scarlet2

    point_sources = scene.sources[-len(SOURCE_LABELS) :]
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "source",
            "label",
            "channel_index",
            "band",
            "epoch_id",
            "kind",
            "telescope",
            "mjd",
            "filename",
            "flux_nanomaggy",
            "mag_ab",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for src_label, source in zip(SOURCE_LABELS, point_sources, strict=True):
            fluxes = np.asarray(scarlet2.measure.flux(source), dtype=float)
            for idx, flux in enumerate(fluxes):
                band, epoch_id = channels[idx]
                row = rows[idx]
                # Fluxes are already in nanomaggies (no normalization applied).
                flux_physical = flux
                if np.isfinite(flux_physical) and flux_physical > 0:
                    mag_check = 22.5 - 2.5 * np.log10(flux_physical)
                    if abs(mag_check) < 30:  # only print plausible values
                        print(f"  FLUX {src_label} [{rows[idx]['telescope']} {rows[idx]['band']} "
                              f"MJD={rows[idx]['mjd']:.1f}]: "
                              f"flux={flux_physical:.4e} nmgy  mag={mag_check:.3f}")
                mag = 22.5 - 2.5 * np.log10(flux_physical) if flux_physical > 0 else np.nan
                writer.writerow(
                    {
                        "source": src_label,
                        "label": src_label,
                        "channel_index": idx,
                        "band": band,
                        "epoch_id": epoch_id,
                        "kind": row["kind"],
                        "telescope": row["telescope"],
                        "mjd": row["mjd"],
                        "filename": Path(row["path"]).name,
                        "flux_nanomaggy": flux_physical,
                        "mag_ab": mag,
                    }
                )


def capture_scene_validation(scene, observations) -> dict[int, dict]:
    """Run scarlet2 scene validation and capture chi2 results per observation.

    Captures stdout during validation calls and parses the printed chi2_in
    and chi2_border values per source/observation index.

    Returns a dict mapping obs_index -> {
        "chi2_in_flag":      "ok" | "warn" | "error",
        "chi2_border_flag":  "ok" | "warn" | "error",
        "chi2_in_worst":     float,
        "chi2_border_worst": float,
    }
    The flag reflects the worst level across all sources for that observation.
    """

    import io
    import re
    import sys

    # Capture stdout where scarlet2 prints its validation messages.
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        model = scene()
        for obs in observations:
            try:
                obs.validate(model)
            except Exception:
                pass
        try:
            scene.check(observations)
        except Exception:
            pass
    finally:
        sys.stdout = old_stdout

    captured = buf.getvalue()
    # Also print so the user still sees the messages in the console.
    print(captured, end="")

    # Parse lines like:
    #   [005]   ERROR   The chi-square in the box for source 2 is poor.
    #                   | Context={'chi2_in': Array(23.049225, ...), 'source': 2}
    #   [007]   WARN    The chi-square in the border for source 3 is acceptable
    #                   | Context={'chi2_border': Array(3.41...), 'source': 3}
    # We extract level, metric type, chi2 value, and source index from each line.

    level_rank = {"ok": 0, "warn": 1, "error": 2}

    # Per-source results first, then aggregate to per-observation below.
    # Since we don't have a direct source->obs mapping here, we aggregate
    # across all sources per observation using the worst flag seen in each call.
    # obs_results keyed by obs_index (from the loop order in observations).
    obs_results: dict[int, dict] = {}

    # scarlet2 prints one block per observation; track current obs index by
    # counting "Observation validation results:" headers.
    current_obs = -1
    for line in captured.splitlines():
        if "Observation validation results" in line or "observation validation" in line.lower():
            current_obs += 1
            obs_results.setdefault(current_obs, {
                "chi2_in_flag": "ok", "chi2_border_flag": "ok",
                "chi2_in_worst": np.nan, "chi2_border_worst": np.nan,
            })
            continue

        if current_obs < 0:
            continue

        # Determine level from line prefix
        line_upper = line.upper()
        if "ERROR" in line_upper:
            level = "error"
        elif "WARN" in line_upper:
            level = "warn"
        else:
            continue

        entry = obs_results.setdefault(current_obs, {
            "chi2_in_flag": "ok", "chi2_border_flag": "ok",
            "chi2_in_worst": np.nan, "chi2_border_worst": np.nan,
        })

        # chi2_in: "chi-square in the box" or chi2_in in context
        if "in the box" in line.lower() or "chi2_in" in line:
            m = re.search(r"chi2_in[^0-9.]*([0-9]+\.[0-9]+)", line)
            if not m:
                m = re.search(r"Array\(([0-9]+\.[0-9]+)", line)
            if m:
                val = float(m.group(1))
                if level_rank[level] > level_rank[entry["chi2_in_flag"]]:
                    entry["chi2_in_flag"] = level
                if not np.isfinite(entry["chi2_in_worst"]) or val > entry["chi2_in_worst"]:
                    entry["chi2_in_worst"] = val

        # chi2_border: "chi-square in the border" or chi2_border in context
        if "in the border" in line.lower() or "chi2_border" in line:
            m = re.search(r"chi2_border[^0-9.]*([0-9]+\.[0-9]+)", line)
            if not m:
                m = re.search(r"Array\(([0-9]+\.[0-9]+)", line)
            if m:
                val = float(m.group(1))
                if level_rank[level] > level_rank[entry["chi2_border_flag"]]:
                    entry["chi2_border_flag"] = level
                if not np.isfinite(entry["chi2_border_worst"]) or val > entry["chi2_border_worst"]:
                    entry["chi2_border_worst"] = val

    return obs_results


def write_fit_quality_csv(scene, observations, rows: list[dict], output_path: Path) -> None:
    """Write per-observation fit diagnostics when supported by scarlet2.

    Columns:
    - goodness_of_fit: scarlet2 average weighted chi-square-like metric
    - chi2_in_flag / chi2_border_flag: worst validation flag across all sources
      for this observation ("ok", "warn", or "error")
    - chi2_in_worst / chi2_border_worst: worst chi2 value seen across sources
    """

    model = scene()
    validation_results = capture_scene_validation(scene, observations)

    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "index", "kind", "telescope", "band", "mjd", "filename",
            "goodness_of_fit",
            "chi2_in_flag", "chi2_in_worst",
            "chi2_border_flag", "chi2_border_worst",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (obs, row) in enumerate(zip(observations, rows, strict=True)):
            try:
                gof = float(np.asarray(obs.goodness_of_fit(model)))
            except Exception:
                gof = np.nan
            val = validation_results.get(idx, {})
            writer.writerow(
                {
                    "index": idx,
                    "kind": row["kind"],
                    "telescope": row["telescope"],
                    "band": row["band"],
                    "mjd": row["mjd"],
                    "filename": Path(row["path"]).name,
                    "goodness_of_fit": gof,
                    "chi2_in_flag":      val.get("chi2_in_flag",      ""),
                    "chi2_in_worst":     val.get("chi2_in_worst",     np.nan),
                    "chi2_border_flag":  val.get("chi2_border_flag",  ""),
                    "chi2_border_worst": val.get("chi2_border_worst", np.nan),
                }
            )


def make_lightcurve_products(outdir: Path, raw_flux_csv: Path, lt_dia_csv: Path | None) -> None:
    """Create science-only CSVs, summary tables, plots, and optional LT DIA comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    raw = pd.read_csv(raw_flux_csv)
    science = raw[(raw["kind"] == "science") & (raw["epoch_id"] >= 0)].copy()
    science = science.sort_values(["telescope", "source", "band", "mjd"])
    science["finite_mag"] = np.isfinite(science["mag_ab"])
    science["suspect_near_zero_flux"] = science["flux_nanomaggy"] < 0.01
    science_csv = outdir / "sn2025wny_scarlet2_science_lightcurve.csv"
    science.to_csv(science_csv, index=False)

    summary = (
        science.groupby(["telescope", "source", "band"], dropna=False)
        .agg(
            n=("mag_ab", "size"),
            median_mag=("mag_ab", "median"),
            min_flux=("flux_nanomaggy", "min"),
            max_flux=("flux_nanomaggy", "max"),
            n_near_zero=("suspect_near_zero_flux", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "sn2025wny_scarlet2_science_summary.csv", index=False)

    for telescope, tel_df in science.groupby("telescope"):
        bands = [b for b in ["g", "r", "i"] if b in set(tel_df["band"])]
        if not bands:
            continue
        fig, axes = plt.subplots(1, len(bands), figsize=(4.4 * len(bands), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, band in zip(axes, bands, strict=True):
            for src, sub in tel_df[(tel_df["band"] == band) & np.isfinite(tel_df["mag_ab"])].groupby("source"):
                ax.plot(sub["mjd"], sub["mag_ab"], marker="o", lw=1.2, label=src)
            ax.set_title(f"{band}-band")
            ax.set_xlabel("MJD")
            ax.grid(alpha=0.25)
            ax.invert_yaxis()
        axes[0].set_ylabel("AB mag from scarlet2 flux")
        axes[-1].legend(title="Image", fontsize=8)
        fig.suptitle(f"SN2025wny {telescope} scarlet2 scene-model light curves", fontsize=9)
        fig.tight_layout()
        safe_name = telescope.replace(" ", "_").replace("/", "_")
        fig.savefig(outdir / f"sn2025wny_scarlet2_{safe_name}_gri_lightcurve.png", dpi=180)
        plt.close(fig)

    if lt_dia_csv and lt_dia_csv.exists():
        dia = pd.read_csv(lt_dia_csv)
        dia["band"] = dia["filter"].map(FILTER_TO_BAND).fillna(dia["filter"])
        dia["source"] = dia["label"]
        dia = dia[np.isfinite(dia["mag"]) & dia["band"].isin(["g", "r", "i"])].copy()
        lt = science[(science["telescope"] == "Liverpool Telescope") & np.isfinite(science["mag_ab"])].copy()
        rows = []
        for _, row in lt.iterrows():
            cand = dia[(dia["source"] == row["source"]) & (dia["band"] == row["band"])]
            if cand.empty:
                continue
            nearest_idx = (cand["mjd"] - row["mjd"]).abs().idxmin()
            nearest = cand.loc[nearest_idx]
            if abs(float(nearest["mjd"]) - float(row["mjd"])) > 0.25:
                continue
            rows.append(
                {
                    "source": row["source"],
                    "band": row["band"],
                    "mjd": row["mjd"],
                    "filename": row["filename"],
                    "scarlet_mag_ab": row["mag_ab"],
                    "dia_mag": nearest["mag"],
                    "delta_mag_scarlet_minus_dia": row["mag_ab"] - nearest["mag"],
                    "dia_usable": nearest.get("usable", np.nan),
                    "dia_quality_score": nearest.get("quality_score", np.nan),
                }
            )
        comp = pd.DataFrame(rows)
        comp.to_csv(outdir / "sn2025wny_lt_scarlet2_vs_dia.csv", index=False)

        if not comp.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for src, sub in comp.groupby("source"):
                axes[0].scatter(sub["dia_mag"], sub["scarlet_mag_ab"], s=24, label=src)
                axes[1].scatter(sub["mjd"], sub["delta_mag_scarlet_minus_dia"], s=24, label=src)
            lo = float(np.nanmin([comp["dia_mag"].min(), comp["scarlet_mag_ab"].min()]))
            hi = float(np.nanmax([comp["dia_mag"].max(), comp["scarlet_mag_ab"].max()]))
            axes[0].plot([lo, hi], [lo, hi], color="0.3", lw=1)
            axes[0].invert_xaxis()
            axes[0].invert_yaxis()
            axes[0].set_xlabel("LT DIA mag")
            axes[0].set_ylabel("scarlet2 mag")
            axes[0].grid(alpha=0.25)
            axes[1].axhline(0, color="0.3", lw=1)
            axes[1].set_xlabel("MJD")
            axes[1].set_ylabel("scarlet2 - DIA mag")
            axes[1].grid(alpha=0.25)
            axes[1].legend(title="Image", fontsize=8)
            fig.suptitle("Liverpool Telescope: scarlet2 vs DIA")
            fig.tight_layout()
            fig.savefig(outdir / "sn2025wny_lt_scarlet2_vs_dia.png", dpi=180)
            plt.close(fig)


def write_observation_plots(observations, rows: list[dict], outdir: Path) -> None:
    """Save a per-observation data + PSF panel for every loaded observation.

    Uses ``scarlet2.plot.observation`` with ``show_psf=True`` so you can
    immediately see whether the Gaussian PSF estimate is a plausible match to
    the stellar profiles in each frame.  Also saves a companion weight map PNG
    for each observation so masked/zero-weight regions (stripes, bad columns,
    edge effects) are immediately visible.  Files go to ``outdir/obs_plots/``
    named by index, kind, telescope, band, and MJD.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scarlet2

    obs_dir = outdir / "obs_plots"
    obs_dir.mkdir(parents=True, exist_ok=True)

    for idx, (obs, row) in enumerate(zip(observations, rows, strict=True)):
        kind = row["kind"]
        telescope = row["telescope"].replace(" ", "_").replace("/", "_")
        band = row["band"]
        mjd = row["mjd"]
        stem = f"{idx:03d}_{kind}_{telescope}_{band}_mjd{mjd:.1f}"
        save_path = obs_dir / f"{stem}.png"
        weight_path = obs_dir / f"{stem}_weights.png"

        try:
            norm = scarlet2.plot.AsinhAutomaticNorm(obs)
            fig = scarlet2.plot.observation(
                observation=obs,
                norm=norm,
                channel_map=None,
                show_psf=True,
                add_labels=False,
            )
            if fig is None:
                fig = plt.gcf()
            fig.suptitle(
                f"{kind} | {row['telescope']} | {band}-band | MJD {mjd:.2f}",
                fontsize=9,
                y=1.01,
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            save_path.with_suffix(".error.txt").write_text(
                f"write_observation_plots failed for index {idx} ({stem}):\n{exc}\n"
            )
            plt.close("all")

        # ── Weight map plot ───────────────────────────────────────────────────
        try:
            weights = np.asarray(obs.weights[0], dtype="float32")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(weights, origin="lower", cmap="viridis",
                           vmin=0, vmax=float(np.nanpercentile(weights[weights > 0], 99)) if (weights > 0).any() else 1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight (1/σ²)")
            ax.set_title(
                f"{kind} | {row['telescope']} | {band}-band | MJD {mjd:.2f}\n"
                f"min={weights.min():.3e}  max={weights.max():.3e}  "
                f"frac_zero={float((weights == 0).mean()):.2%}",
                fontsize=7,
            )
            fig.tight_layout()
            fig.savefig(weight_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            weight_path.with_suffix(".error.txt").write_text(
                f"weight map plot failed for index {idx} ({stem}):\n{exc}\n"
            )
            plt.close("all")

    print(f"Wrote observation+PSF plots to {obs_dir}")


def write_init_scene_plots(scene, observations, rows: list[dict], outdir: Path) -> None:
    """Save per-observation model/rendered/residual panels for the *initialized*
    (pre-fit) scene.

    Calling this before ``scarlet2.fit`` lets you catch bad source initializations
    — wrong positions, wildly off spectra, morphology bleed — before committing
    to a full fit.  Uses the same ``scarlet2.plot.scene`` call as the post-fit
    plots so comparisons are direct.  Files go to ``outdir/init_scene_plots/``.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scarlet2

    init_dir = outdir / "init_scene_plots"
    init_dir.mkdir(parents=True, exist_ok=True)

    for idx, (obs, row) in enumerate(zip(observations, rows, strict=True)):
        kind = row["kind"]
        telescope = row["telescope"].replace(" ", "_").replace("/", "_")
        band = row["band"]
        mjd = row["mjd"]
        fname = f"{idx:03d}_{kind}_{telescope}_{band}_mjd{mjd:.1f}.png"
        save_path = init_dir / fname

        try:
            norm = scarlet2.plot.AsinhAutomaticNorm(obs)
            fig = scarlet2.plot.scene(
                scene,
                observation=obs,
                norm=norm,
                show_model=True,
                show_observed=True,
                show_rendered=True,
                show_residual=True,
                add_labels=True,
                add_boxes=True,
                split_channels=False,
                box_kwargs={"edgecolor": "red", "facecolor": "none"},
                label_kwargs={"color": "red"},
            )
            if fig is None:
                fig = plt.gcf()
            fig.suptitle(
                f"INIT | {kind} | {row['telescope']} | {band}-band | MJD {mjd:.2f}",
                fontsize=9,
                y=1.01,
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            save_path.with_suffix(".error.txt").write_text(
                f"write_init_scene_plots failed for index {idx} ({fname}):\n{exc}\n"
            )
            plt.close("all")

    print(f"Wrote initialized scene plots to {init_dir}")


def write_morphology_plots(scene, outdir: Path, prefix: str = "") -> None:
    """Save a PNG of the rendered morphology for each extended source.

    Shows the morphology model in image space (after StarletMorphology renders
    it) alongside a log-stretch version to reveal faint outer structure.
    Files go to ``outdir/morph_plots/`` named by source label, optionally
    prefixed (e.g. ``init_`` for pre-fit, ``fit_`` for post-fit) so both can
    coexist in the same directory for direct comparison.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import scarlet2

    morph_dir = outdir / "morph_plots"
    morph_dir.mkdir(parents=True, exist_ok=True)

    static_sources = [
        (label, src)
        for label, src in zip(STATIC_LABELS, scene.sources[:len(STATIC_LABELS)], strict=True)
        if hasattr(src, "morphology")
    ]

    if not static_sources:
        print("  write_morphology_plots: no extended sources with morphology found.")
        return

    fname_prefix = f"{prefix}_" if prefix else ""

    for label, source in static_sources:
        try:
            # Render morphology to image space
            morph_image = np.asarray(source.morphology(), dtype="float32")

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            # Linear stretch
            vmax = float(np.nanpercentile(morph_image, 99.5))
            vmin = 0.0
            im0 = axes[0].imshow(morph_image, origin="lower", cmap="inferno",
                                  vmin=vmin, vmax=vmax)
            axes[0].set_title("Linear", fontsize=9)
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # Log stretch — clip to positive before taking log
            pos = np.clip(morph_image, 1e-8 * vmax, None) if vmax > 0 else morph_image + 1e-20
            im1 = axes[1].imshow(np.log10(pos), origin="lower", cmap="inferno")
            axes[1].set_title("Log₁₀", fontsize=9)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            title_tag = f"{prefix.upper()} | " if prefix else ""
            fig.suptitle(f"{title_tag}Morphology model: {label}", fontsize=9)
            fig.tight_layout()

            save_path = morph_dir / f"{fname_prefix}morphology_{label}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved morphology plot: {save_path.name}  "
                  f"(shape={morph_image.shape}, max={float(morph_image.max()):.3e})")

        except Exception as exc:
            (morph_dir / f"{fname_prefix}morphology_{label}.error.txt").write_text(
                f"write_morphology_plots failed for {label}:\n{exc}\n"
            )
            plt.close("all")

    print(f"Wrote morphology plots to {morph_dir}")


def write_fit_images(scene, observations, rows: list[dict], outdir: Path) -> None:
    """Save per-observation data / rendered model / residual panels to PNG.

    For each observation a three-row figure is saved:
        row 1 – observed data cutout
        row 2 – scarlet2 rendered model in the observation frame
        row 3 – residual (data − model)

    The norm is derived automatically from the observed data using
    ``scarlet2.plot.AsinhAutomaticNorm`` so faint structure is visible.
    Files are written to ``outdir/fit_images/`` named by index, kind,
    telescope, band, and MJD so they sort sensibly on disk.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scarlet2

    img_dir = outdir / "fit_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for idx, (obs, row) in enumerate(zip(observations, rows, strict=True)):
        kind = row["kind"]
        telescope = row["telescope"].replace(" ", "_").replace("/", "_")
        band = row["band"]
        mjd = row["mjd"]
        fname = f"{idx:03d}_{kind}_{telescope}_{band}_mjd{mjd:.1f}.png"
        save_path = img_dir / fname

        try:
            norm = scarlet2.plot.AsinhAutomaticNorm(obs)
            fig = scarlet2.plot.scene(
                scene,
                observation=obs,
                norm=norm,
                show_model=True,
                show_observed=True,
                show_rendered=True,
                show_residual=True,
                add_labels=True,
                add_boxes=True,
                split_channels=False,
                box_kwargs={"edgecolor": "red", "facecolor": "none"},
                label_kwargs={"color": "red"},
            )
            if fig is None:
                fig = plt.gcf()
            fig.suptitle(
                f"{kind} | {row['telescope']} | {band}-band | MJD {mjd:.2f}",
                fontsize=9,
                y=1.01,
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            # Write a plain-text error file so a failed frame is visible without
            # crashing the whole batch.
            save_path.with_suffix(".error.txt").write_text(
                f"write_fit_images failed for index {idx} ({fname}):\n{exc}\n"
            )
            plt.close("all")

    print(f"Wrote fit images to {img_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible scene-model runs."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("scene_modeling/scene_manifest.json"))
    parser.add_argument("--outdir", type=Path, default=Path("scene_modeling/scarlet2_outputs"))
    parser.add_argument("--bands", default="griz", help="Filter string, for example gri.")
    parser.add_argument("--telescopes", default="", help="Comma-separated telescope names to include (all kinds); empty means all.")
    parser.add_argument("--exclude-telescopes", default="", help="Comma-separated telescope names to exclude from both science and reference rows (default: none).")
    parser.add_argument("--cutout-arcsec", type=float, default=26.0)
    parser.add_argument("--static-stamp-size", type=int, default=63, help="Odd pixel size for reference-derived static stamps.")
    parser.add_argument("--quality-top-per-band-telescope", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--e-rel", type=float, default=1e-7)
    parser.add_argument("--fit-centers", action="store_true", help="Allow transient point-source centers to move.")
    parser.add_argument("--fit-astrometric-shifts", action=argparse.BooleanOptionalAction, default=True, help="Fit per-observation astrometric shifts (on by default). Use --no-fit-astrometric-shifts to disable.")
    parser.add_argument(
        "--skip-frames",
        default="",
        help="Comma-separated MJDs to skip (e.g. '61135.3,60924.0'). "
             "Frames within 0.1 days of any listed value are excluded.",
    )
    parser.add_argument("--shift-stepsize", type=float, default=1.6e-1, help="Astrometric shift stepsize in arcsec (default 0.04). Increase if convergence is slow, decrease if shifts oscillate.")
    parser.add_argument("--static-init-min-snr", type=float, default=3.0, help="Minimum peak SNR for a stamp to contribute to static source initialization (default 3.0).")
    parser.add_argument("--morphology-init-path", type=Path, default=None, help="Path to an .npz file (from fit_reference_morphology.py) with pre-fitted Starlet coefficients for lens_g1/host_g2. If given, these are used as the initial morphology instead of the Gaussian/wavelet initializer.")
    parser.add_argument("--morph-stepsize-absolute", type=float, default=None, help="If given, use this fixed absolute stepsize for the morphology Parameter instead of relative_step. Intended for small refinement around a --morphology-init-path reference morphology (e.g. 1e-4).")
    parser.add_argument("--morph-stepsize-relative-factor", type=float, default=2e-4, help="Factor passed to scarlet2.relative_step() for the morphology Parameter, used when --morph-stepsize-absolute is NOT given (default: 2e-4).")
    parser.add_argument("--lt-dia-csv", type=Path, default=None, help="Optional LT DIA light-curve table for comparison plots.")
    parser.add_argument(
        "--model-info-dir",
        type=Path,
        default=None,
        help="Path to model_info/ directory containing PSFs/ and noisemaps/ subdirectories. "
             "Defaults to <manifest_dir>/../model_info if not set.",
    )
    parser.add_argument("--save-obs-plots", action="store_true", help="Save per-observation data+PSF PNG panels to outdir/obs_plots/ before fitting.")
    parser.add_argument("--save-init-plots", action="store_true", help="Save per-observation initialized scene PNG panels to outdir/init_scene_plots/ before fitting.")
    parser.add_argument("--save-morph-plots", action="store_true", help="Save morphology model PNG panels for extended sources to outdir/morph_plots/.")
    parser.add_argument("--save-scene-plots", action="store_true", help="Save per-observation data/model/residual PNG panels to outdir/fit_images/.")
    return parser.parse_args()


def apply_fitted_shifts(
    obs_native: list,
    obs_corr_fit: list,
    model_frame,
) -> None:
    """Apply optimized astrometric shifts from fitted CorrelatedObservations
    back to the native (non-resampled) Observation objects.

    After fitting, the optimized shift lives in
    ``obs_corr.renderer[-1].shift`` in the resampled (correlated) frame.
    This function transforms each shift back to native pixels using the
    Jacobian between the correlated and native frames, then sets it on the
    native observation's renderer so that any subsequent rendering into the
    native frame uses the corrected astrometry.

    Follows the pattern from the scarlet2 example:
        jacobian, _ = sc2.frame.get_relative_jacobian_shift(corr.frame, native.frame)
        shift_native = jacobian @ shift
        native.check_set_renderer(model_frame)
        object.__setattr__(native.renderer[-1], "shift", shift_native)
    """

    import scarlet2

    n_applied = 0
    for idx, (obs_nat, obs_corr) in enumerate(zip(obs_native, obs_corr_fit, strict=True)):
        try:
            shift_corr = obs_corr.renderer[-1].shift
        except (IndexError, AttributeError):
            continue

        try:
            jacobian, _ = scarlet2.frame.get_relative_jacobian_shift(
                obs_corr.frame, obs_nat.frame
            )
            shift_native = jacobian @ shift_corr
        except Exception as exc:
            print(f"  apply_fitted_shifts [{idx}]: could not compute Jacobian: {exc}")
            continue

        obs_nat.check_set_renderer(model_frame)
        try:
            object.__setattr__(obs_nat.renderer[-1], "shift", shift_native)
            n_applied += 1
        except Exception as exc:
            print(f"  apply_fitted_shifts [{idx}]: could not set shift: {exc}")

    print(f"Applied fitted astrometric shifts to {n_applied}/{len(obs_native)} native observations.")


def main() -> None:
    """Run the complete scarlet2 fit and write science-ready inspection products."""

    import scarlet2
    import scarlet2.io as scio

    args = parse_args()
    payload = json.loads(args.manifest.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    #print(args.outdir)
    #sys.exit()
    bands = sorted(set(args.bands))
    telescopes = {name.strip() for name in args.telescopes.split(",") if name.strip()} or None
    exclude_telescopes = {name.strip() for name in args.exclude_telescopes.split(",") if name.strip()} or None
    skip_mjds = {float(v.strip()) for v in args.skip_frames.split(",") if v.strip()} or None
    # Inject scene centre into each row so _p60_centering_ok can use it.
    _scene_ra  = payload["scene_center"]["ra"]
    _scene_dec = payload["scene_center"]["dec"]
    for _row in payload["observations"]:
        _row.setdefault("scene_ra_deg",  _scene_ra)
        _row.setdefault("scene_dec_deg", _scene_dec)
        _row.setdefault("stamp_arcsec",  args.cutout_arcsec)

    rows = select_rows(payload, telescopes, set(bands), args.max_images, args.quality_top_per_band_telescope, exclude_telescopes=exclude_telescopes, skip_mjds=skip_mjds)
    center = SkyCoord(payload["scene_center"]["ra"], payload["scene_center"]["dec"], unit="deg", frame="icrs")

    # Resolve model_info directory.
    # Default search order:
    #   1. --model-info-dir if explicitly passed
    #   2. <manifest_dir>/../../model_info  (images_25wny/model_info — correct layout)
    #   3. <manifest_dir>/../model_info     (one level shallower, fallback)
    model_info_dir = args.model_info_dir
    if model_info_dir is None:
        manifest_path = Path(args.manifest).resolve()
        candidates = [
            manifest_path.parent.parent.parent / "model_info",
            manifest_path.parent.parent / "model_info",
        ]
        for candidate in candidates:
            if candidate.exists():
                model_info_dir = candidate
                break
    if model_info_dir is None or not model_info_dir.exists():
        print(f"WARNING: model_info_dir not found; tried: {[str(c) for c in candidates]}. "
              "Falling back to Gaussian PSFs and manifest noise estimates. "
              "Pass --model-info-dir explicitly to override.")
        model_info_dir = None
    else:
        print(f"Using model_info_dir: {model_info_dir}")

    channels = []
    science_epoch_ids = []
    observations = []
    scale_factors = []
    for idx, row in enumerate(rows):
        epoch_id = -1 if row["kind"] == "reference" else idx
        channel = (row["band"], epoch_id)
        channels.append(channel)
        if row["kind"] == "science":
            science_epoch_ids.append(epoch_id)
        obs, scale = load_observation(row, channel, center, args.cutout_arcsec, args_model_info_dir=model_info_dir)
        observations.append(obs)
        scale_factors.append(scale)
        if row.get("telescope") == "Pan-STARRS1":
            try:
                pixscale = float(np.mean(proj_plane_pixel_scales(obs.frame.wcs)) * 3600.0)
                print(f"  PS1 obs.frame.wcs pixel scale: {pixscale:.6f} arcsec/px  "
                      f"CDELT1={obs.frame.wcs.wcs.cdelt[0]:.6e}  "
                      f"shape={obs.data.shape}")
            except Exception as exc:
                print(f"  PS1 obs.frame.wcs diagnostic failed: {exc}")



    selected_manifest = args.outdir / "selected_scene_manifest_rows.json"
    selected_manifest.write_text(json.dumps({"observations": rows}, indent=2, allow_nan=True))

    # Diagnostic only: report each observation's PSF sigma so it's easy to
    # spot a mismatch. The model PSF itself is now chosen automatically by
    # scarlet2.Frame.from_observations.
    n_psf_unavailable = 0
    for obs, row in zip(observations, rows, strict=True):
        psf_arr = _get_obs_psf_array(obs)
        if psf_arr is None:
            n_psf_unavailable += 1
            continue
        try:
            sigma_pix = gaussian_sigma_from_psf(psf_arr)
            pixscale_arcsec = float(
                np.mean(proj_plane_pixel_scales(obs.frame.wcs)) * 3600.0
            )
            sigma_arcsec = sigma_pix * pixscale_arcsec
            print(f"  PSF_SIGMA [{row['telescope']} {row['band']} MJD={row.get('mjd',0):.1f}]: "
                  f"sigma={sigma_pix:.2f} px = {sigma_arcsec:.4f} arcsec "
                  f"(pixscale={pixscale_arcsec:.4f} arcsec/px)")
        except Exception as exc:
            print(f"  PSF_SIGMA [{row['telescope']} {row['band']}]: could not estimate ({exc})")
    if n_psf_unavailable:
        print(f"  PSF_SIGMA: PSF array not accessible for {n_psf_unavailable}/{len(observations)} "
              f"observations (diagnostic only, does not affect the fit)")

    scene, matched_observations, model_frame = build_scene(
        observations,
        rows,
        channels,
        set(science_epoch_ids),
        bands,
        static_stamp_size=args.static_stamp_size,
        fit_centers=args.fit_centers,
        fit_astrometric_shifts=args.fit_astrometric_shifts,
        shift_stepsize=args.shift_stepsize,
        static_init_min_snr=args.static_init_min_snr,
        morphology_init_path=args.morphology_init_path,
        morph_stepsize_absolute=args.morph_stepsize_absolute,
        morph_stepsize_relative_factor=args.morph_stepsize_relative_factor,
    )

    if args.save_obs_plots:
        write_observation_plots(matched_observations, rows, args.outdir)

    if args.save_init_plots:
        write_init_scene_plots(scene, matched_observations, rows, args.outdir)

    if args.save_morph_plots:
        write_morphology_plots(scene, args.outdir, prefix="init")

    scene_fit, obs_fit = scarlet2.fit(
        scene,
        matched_observations,
        max_iter=args.max_iter,
        e_rel=args.e_rel,
        progress_bar=True,
    )

    # Apply optimized astrometric shifts back to the native observations so
    # that any rendering into native pixel frames uses the corrected WCS.
    if args.fit_astrometric_shifts:
        apply_fitted_shifts(observations, obs_fit, model_frame)
        print("Final fitted astrometric shifts (in resampled frame pixels):")
        for idx, (obs_corr, row) in enumerate(zip(obs_fit, rows, strict=True)):
            try:
                shift = obs_corr.renderer[-1].shift
                print(f"  [{idx:03d}] {row['telescope']:20s} {row['band']} "
                      f"MJD={row['mjd']:.2f}  shift=({float(shift[0]):.4f}, {float(shift[1]):.4f}) px")
            except (IndexError, AttributeError):
                pass

    scene_path = args.outdir / "sn2025wny_scarlet2_scene.h5"
    raw_flux_path = args.outdir / "sn2025wny_scarlet2_transient_fluxes.csv"
    quality_path = args.outdir / "sn2025wny_scarlet2_fit_quality.csv"
    try:
        scio.model_to_h5(scene_fit, scene_path, id=1, overwrite=True)
    except OSError as exc:
        warning_path = args.outdir / "sn2025wny_scarlet2_scene_h5_warning.txt"
        warning_path.write_text(
            "scarlet2 finished the optimization, but model_to_h5 could not "
            "serialize this scene. This can happen for large joint runs because "
            f"the serialized model exceeds an HDF5 attribute-size limit.\n\n{exc}\n"
        )
        print(f"Warning: could not write {scene_path}; wrote {warning_path}")
    write_flux_csv(scene_fit, channels, rows, scale_factors, raw_flux_path)
    write_fit_quality_csv(scene_fit, obs_fit, rows, quality_path)
    make_lightcurve_products(args.outdir, raw_flux_path, args.lt_dia_csv)

    if args.save_scene_plots:
        write_fit_images(scene_fit, obs_fit, rows, args.outdir)

    if args.save_morph_plots:
        write_morphology_plots(scene_fit, args.outdir, prefix="fit")

    print(f"Wrote {scene_path}")
    print(f"Wrote {raw_flux_path}")
    print(f"Wrote {quality_path}")
    print(f"Wrote {args.outdir / 'sn2025wny_scarlet2_science_lightcurve.csv'}")


if __name__ == "__main__":
    main()
