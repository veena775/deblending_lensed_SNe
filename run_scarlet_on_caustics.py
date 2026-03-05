import sys
import os

sys.path.insert(0, os.path.abspath('/home/vk9342/scarlet2/src'))

import scarlet2
print(scarlet2.__file__) # This confirms it's loading from your home directory
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
matplotlib.rc('image', cmap='gray', interpolation='none', origin='lower')

import jax


import jax.numpy as jnp
from jax import random, jit
import jax
import copy

import pandas as pd
from scipy.stats import gaussian_kde as kde
from scipy.stats import norm

# import scarlet
import scarlet2
# from scarlet.display import AsinhMapping,AsinhPercentileNorm,show_scarlet2_scene,LinearPercentileNorm
# from scarlet.source import StaticSource,MultiExtendedSource, StaticMultiExtendedSource
from scarlet2 import *
from scarlet2 import relative_step
from scarlet2 import (
    Observation, Frame, Scene, Source, PointSource,
    init, relative_step, Parameter, StaticArraySpectrum, 
    TransientArraySpectrum
)
from scarlet2 import Starlet
from scarlet2.plot import AsinhPercentileNorm


import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits
import astropy.io.fits as fits

import sys
import copy
import corner
import h5py
import os
import glob
import sep
from sep import Background, extract
import tarfile
import equinox as eqx
# import distrax
import optax
from tqdm.auto import tqdm
# import cmasher as cmr


from functools import partial
from skimage import draw
%matplotlib inline
# import astrophot as ap
pixelscale=0.17

# turn on automatic validation checks
from scarlet2.validation_utils import set_validation

# turn on automatic validation checks
set_validation(False)

# ============================================================================
# 1. General File
# ============================================================================
# img_index = 1952
img_index = 17462 # 12486 #1952  #17462 #96679

t_start = 20
t_end = 140
# times = np.arange(56150,56350,2)
t_off   = 0
t_on    = 50
n_roman_epochs = 2


times = [0, 10, 20, 30, 36, 40, 44, 48, 50, 56, 60, 70, 80, 100, 130]


epochs = [int(i) for i in range(len(times))] # basically just list of index keys correspond to each time?

path = '/scratch/network/vk9342/lenstronomy-tutorials/Notebooks/JP_spring_2025_modified_code/outputs/caustics/SNsims/'
data_path  = path + 'img_' + str(img_index) + '/'
psf_path   = '/scratch/network/vk9342/lenstronomy-tutorials/Notebooks/JP_spring_2025_modified_code/'

bands     = ["g","r","i"]
BANDS = ['G', 'R', 'I']
bandall = ['g','r','i','G','R','I']

channels = []
channels_on = []
observations = []
epochs_on = []

n_lsst  = len(bands) * len(times)   # e.g. 3 bands × (len(times))
n_roman  = len(BANDS) * n_roman_epochs   # 3 bands × 2 epochs = 6


# ============================================================================
# Get weights: copied function from caustics (which is based on lenstronomy)
# ============================================================================
telescope_params = {
    'LSST': {
        'read_noise': 10.0,
        'exposure_time': 15.0,
        'pixel_scale': 0.2, 
        'ccd_gain': 2.3,
        'sky_brightness_mags': {'g': 22.26, 'r': 21.2, 'i': 20.48},
        'zero_points_mags': {'g': 28.3, 'r': 28.13, 'i': 27.79},
        'num_exposures': {'g': 200, 'r': 460, 'i': 460}
    },
    'Roman': {
        'read_noise': 15.5,
        'exposure_time': 146.0,
        'pixel_scale': 0.11, 
        'ccd_gain': 1,
        'sky_brightness_mags': {'g': 23.19, 'r': 22.99, 'i': 23.22}, # mapped to F062, F106, F184
        'zero_points_mags': {'g': 26.56, 'r': 26.44, 'i': 25.95},
        'num_exposures': {'g': 96, 'r': 96, 'i': 96}
    }
}

def gaussian_noise_sigma(
    image, 
    readout_noise,   # have read noise param value,     in counts (e.g. e-)?
    exposure_time,   # have exposure time param value,  in seconds?
    sky_brightness,  # have sky brightness param value, in magnitudes/arcsec^2
    pixel_scale,     # have pixel scale param value,    in arcsec
    num_exposures,   # have num exposures param value,  unitless
    magnitude_zero_point
):
    '''
    background_noise: Gaussian sigma of noise level per pixel in counts (e- or ADU) per second.
    Im assuming we're using e- and not ADU???
    '''
    
    # Convert Sky Magnitude to Counts/sec/arcsec^2 (conversion from Lenstronomy)
    sky_brightness_counts_arcsec2 = 10 ** (-(sky_brightness - magnitude_zero_point) / 2.5)

    exposure_time_tot = num_exposures * exposure_time
    readout_noise_tot = num_exposures * readout_noise**2  # square of readout noise
    sky_per_pixel = sky_brightness_counts_arcsec2 * pixel_scale**2
    sky_brightness_tot = exposure_time_tot * sky_per_pixel
    
    sigma_bkg = np.sqrt(readout_noise_tot + sky_brightness_tot) / exposure_time_tot

    return sigma_bkg


# ============================================================================
# 3. Load Observations for each Rubin epoch
# ============================================================================
for ind_b, band in enumerate(bands):
    for ind_t, epoch in enumerate(times):
        channel_sc2 = (band, int(ind_t)) 

        if epoch == t_on:
            ind_t_on = str(ind_t)
        elif epoch == t_off:
            ind_t_off = str(ind_t)
        
        # get channels for only 'on' times
        if epoch >= t_start and epoch <t_end:  
            channels_on.append(channel_sc2)
            epochs_on.append(int(ind_t))
        
        # Caustics image simulation files (fits)
        data_file_name = 'image_LSST_'+band+'_'+str(img_index)+'_SNnoise_'+str(epoch)+'.0.fits' 
        img = data_path+ data_file_name

        obs_hdu = fits.open(img)
        data_lsst = obs_hdu[0].data
        data_lsst = data_lsst.astype(data_lsst.dtype.newbyteorder('=')) ## Do I sill need this?
        N1, N2 = data_lsst.shape
        data_lsst = data_lsst.reshape(1, N1, N2) #reshape because?

        # WCS - saved as headerof fits files, made manually in Caustics
        wcs_lsst = WCS(obs_hdu[0].header)

        # PSF data - npy files with gaussians, made manually in Caustics
        psf_data = np.load(psf_path + f'caustics_gaussian_psf_LSST_{band}_size_17.npy')
        psf_jax = jnp.asarray([psf_data.astype('float32')])
        psf_lsst = scarlet2.ArrayPSF(psf_jax)

        
        ## Recalculate added noise (added on top of Caustics image before saving .fits files)
        params = telescope_params['LSST']
        noise_sigma = gaussian_noise_sigma(
            image = data_lsst,
            readout_noise = params['read_noise'],
            exposure_time = params['exposure_time'],
            sky_brightness = params['sky_brightness_mags'][band],
            pixel_scale = params['pixel_scale'],
            num_exposures = params['num_exposures'][band],
            magnitude_zero_point = params['zero_points_mags'][band]
        )
        weights_lsst = jnp.ones_like(data_lsst) / (noise_sigma**2)
        # print(noise_sigma)

        # Make observation
        obs_lsst = scarlet2.Observation(jnp.asarray(data_lsst),
                                      wcs=wcs_lsst,
                                      psf=psf_lsst,
                                      channels=[channel_sc2],
                                      weights=weights_lsst)
        observations.append(obs_lsst)
        channels.append(channel_sc2)


print(wcs_lsst)

# ============================================================================
# 5. Load high-res off/on images for Roman
# ============================================================================

#Read in a single epoch of high resolution imaging WITHOUT the SN
for ind_b, (band, BAND) in enumerate(zip(bands, BANDS)):

    channel_sc2 = (BAND, int(ind_t_off))
    epoch = t_off
    
    data_file_name = 'image_Roman_'+band+'_'+str(img_index)+'_SNnoise_'+str(t_off)+'.0.fits' 
    img = data_path+ data_file_name

    obs_hdu = fits.open(img)
    data_roman = obs_hdu[0].data
    data_roman = data_roman.astype(data_roman.dtype.newbyteorder('=')) ## Do I sill need this?
    N1, N2 = data_roman.shape
    data_roman = data_roman.reshape(1, N1, N2) #reshape because?
    wcs_roman = WCS(obs_hdu[0].header)

    psf_data = np.load(psf_path + f'caustics_gaussian_psf_Roman_{band}_size_17.npy')
    psf_jax = jnp.asarray([psf_data.astype('float32')])
    psf_roman = scarlet2.ArrayPSF(psf_jax)

    ## Recalculate added noise
    params = telescope_params['Roman']
    noise_sigma = gaussian_noise_sigma(
        image = data_roman,
        readout_noise = params['read_noise'],
        exposure_time = params['exposure_time'],
        sky_brightness = params['sky_brightness_mags'][band],
        pixel_scale = params['pixel_scale'],
        num_exposures = params['num_exposures'][band],
        magnitude_zero_point = params['zero_points_mags'][band]
    )
    weights_roman = jnp.ones_like(data_roman) / (noise_sigma**2)
    print(noise_sigma)

    # define two observation packages and match to frame
    obs_roman_offSN = scarlet2.Observation(data_roman,
                                   wcs=wcs_roman,
                                   psf=psf_roman,
                                   channels=[channel_sc2],
                                   weights= weights_roman) #should weights be none?    
    
    channels.append(channel_sc2)
    observations.append(obs_roman_offSN)



#Read in a single epoch of high resolution imaging WITH the SN
for ind_b, (band, BAND) in enumerate(zip(bands, BANDS)):
    
    channel_sc2 = (BAND, int(ind_t_on))
    epoch = t_on
    
    data_file_name = 'image_Roman_'+band+'_'+str(img_index)+'_SNnoise_'+str(t_on)+'.0.fits' 
    img = data_path+ data_file_name

    obs_hdu = fits.open(img)
    data_roman = obs_hdu[0].data
    data_roman = data_roman.astype(data_roman.dtype.newbyteorder('=')) ## Do I sill need this?
    N1, N2 = data_roman.shape
    data_roman = data_roman.reshape(1, N1, N2) #reshape because?
    wcs_roman = WCS(obs_hdu[0].header)
    print(wcs_roman)
    # data_roman_offSN = jnp.array(np.asarray(data_roman,dtype=float), jnp.float32)

    psf_data = np.load(psf_path + f'caustics_gaussian_psf_Roman_{band}_size_17.npy')
    psf_jax = jnp.asarray([psf_data.astype('float32')])
    psf_roman = scarlet2.ArrayPSF(psf_jax)

    ## Recalculate added noise
    params = telescope_params['Roman']
    noise_sigma = gaussian_noise_sigma(
        image = data_roman,
        readout_noise = params['read_noise'],
        exposure_time = params['exposure_time'],
        sky_brightness = params['sky_brightness_mags'][band],
        pixel_scale = params['pixel_scale'],
        num_exposures = params['num_exposures'][band],
        magnitude_zero_point = params['zero_points_mags'][band]
    )
    weights_roman = jnp.ones_like(data_roman) / (noise_sigma**2)
    print(noise_sigma)
    

    channels_on.append(channel_sc2)
    epochs_on.append(int(ind_t_on))
    
    # define two observation packages and match to frame
    obs_roman_postSN = scarlet2.Observation(jnp.asarray(data_roman),
                                   wcs=wcs_roman,
                                   psf=psf_roman,
                                   channels=[channel_sc2],
                                   weights=weights_roman) #should weights be none? 
    
    channels.append(channel_sc2)
    observations.append(obs_roman_postSN)


#How many observations did we end up reading in? Should be at least a few
print(len(observations))

# ============================================================================
# 6. Make Frame and match observations
# ============================================================================
# model_frame_psf = scarlet2.GaussianPSF(0.7)
model_frame = scarlet2.Frame.from_observations(
    observations=observations,
    coverage="union"  # or "intersection"
    )

for obs in observations:
    obs.match(model_frame)

# ============================================================================
# 7. Find sourced in image - from RubinRoman_TD notebook
# ============================================================================

def makeCatalog(observations, lvl=3, wave=True, SNR=2.0):
    
    normed_images = np.asarray([obs.data[0] for obs in observations])
    interps = normed_images
    
    interps = np.asarray(interps/np.sum(interps))
    detect_image = np.sum(interps,axis=(0))
    # Wavelet transform
    # wave_detect = scarlet.Starlet.from_image(detect_image).coefficients
    wave_detect = Starlet.from_image(detect_image).coefficients

    if wave:
        # Creates detection from the first 3 wavelet levels
        detect = wave_detect[:lvl,:,:].sum(axis = 0)
    else:
        detect = detect_image

    detect = np.array(detect)

    # Runs SEP detection
    bkg = sep.Background(detect)
    catalog = sep.extract(detect-bkg.globalback, SNR, err=bkg.globalrms)
    background=[]
    bg_rms=[]
    for ind in range(len(observations)):
        img = copy.copy(np.asarray(observations[ind].data))
        if np.size(img.shape) == 3:
            bg_rms.append(np.array([sep.Background(band).globalrms for band in img]))
            background.append(np.array([sep.Background(band).globalback for band in img]))
        else:
            bg_rms.append(sep.Background(img).globalrms)
            background.append(sep.Background(img).globalback)
    return catalog, bg_rms, detect_image, background

# ============================================================================
# 8. Find sources in image - from RubinRoman_TD notebook
# USE ROMAN IMAGES ONLY
# ============================================================================

#If the source detection is not finding all the sources, you can first try to adjust these three parameters to help it find the sources
lvl=2
wave=2
SNR=5.0  # was 2

lsst_obs = observations[:-n_roman]
roman_obs = observations[-n_roman:]

observations_sc2=[]
normsingle=[]

# ============================================================================   
# DETECT SOURCES BASED ONLY ON ROMAN IMGS
catalog_single_roman, bgsingle_roman, detectsingle_roman, globalback_roman = makeCatalog(roman_obs, lvl, wave, SNR)
pixel = np.stack((catalog_single_roman['y'], catalog_single_roman['x']), axis=1)
ra_dec = pixel
print(ra_dec.shape)
# ============================================================================

#################################################################################
### MAKE observations_sc2 ###########################
bgarr_roman     = np.asarray(bgsingle_roman)
bgarrall_roman  = np.hstack((bgarr_roman.flatten(),np.asarray(bgsingle_roman).flatten()))
bgarr_roman     = np.asarray(bgsingle_roman)
print(bgarrall_roman.shape)

catalog_single_lsst, bgsingle_lsst, detectsingle_lsst, globalback_lsst = makeCatalog(lsst_obs, lvl, wave, SNR)
bgarr_lsst    = np.asarray(bgsingle_lsst)
bgarrall_lsst = np.hstack((bgarr_lsst.flatten(),np.asarray(bgsingle_lsst).flatten()))
bgarr_lsst    = np.asarray(bgsingle_lsst)
print(bgarrall_lsst.shape)


bgr_lsst = np.array([b[0] for b in bgarr_lsst])
bgr_roman = np.array([b[0] for b in bgarr_roman])
bgarr_all = np.concatenate([bgr_lsst, bgr_roman], axis=0)

gb_lsst = np.array([b[0] for b in globalback_lsst])
gb_roman = np.array([b[0] for b in globalback_roman])
globalback_all = np.concatenate([gb_lsst, gb_roman], axis=0)

print(len(bgarr_all))
print(len(globalback_all))


for ind,(obs2, bg, back) in enumerate(zip(observations, bgarr_all, globalback_all)):

    print(ind)
    
    w = obs2.frame.wcs
    psf = obs2.frame.psf
    weights2=np.ones(obs2.data.shape) / (bg**2)#[:, None, None]  #??in RubinRoman notebook insteads of bg it uses bgarr_all, is this a typo?
    data = obs2.data-back   #Subtract background flux

    print(channels[ind])
    
    obs_sc2 = scarlet2.Observation(jnp.asarray(data), jnp.asarray(weights2), psf=psf,channels=[channels[ind]],wcs=w)
    observations_sc2.append(obs_sc2)
    # print("obs2.data shape:", obs2.data.shape, '\n')
    # print('obs2.data: ', obs2.data, '\n')
    # print('obs2.data[:,10:-10,10:-10]:', obs2.data[:,10:-10,10:-10], '\n')

    
    #Store norm based on observation data
    normsingle.append(
        AsinhPercentileNorm(
            jnp.asarray(np.asarray((obs2.data[:,10:-10,10:-10]),dtype=np.float32)),
            percentiles=[0.001, 50, 80])
    )



print(observations_sc2[0].data.shape)

model_frame = scarlet2.Frame.from_observations(
    observations=observations_sc2,
    coverage="union",  # or "intersection"
)


for i, obs in enumerate(observations_sc2):
    obs_corr = scarlet2.CorrelatedObservation.from_observation(obs,resample_to_frame=model_frame)
    obs_corr.match(model_frame)
    observations_sc2[i] = obs_corr


# ============================================================================
# 10. Source initialization
# ============================================================================
lsst_obs_sc2 = observations_sc2[:-n_roman]
roman_obs_sc2 = observations_sc2[-n_roman:]

band_selector  = lambda ch: ch[0]
epoch_selector = lambda ch: ch[1]

# ============================================================================
# 9. Get and visualize sources in image
# ============================================================================
world_coords = []
for y, x in ra_dec:
    sky = wcs_roman.pixel_to_world(x, y)   # note: pixel_to_world(x, y)
    world_coords.append(SkyCoord(sky.ra, sky.dec))

roman_pixel_coords = [wcs_roman.world_to_pixel(sky) for sky in world_coords]
lsst_pixel_coords = [wcs_lsst.world_to_pixel(sky) for sky in world_coords]

## Ra and dec centers from SEP catalog
sky_coords_ = world_coords

print("Detected source pixel coordinates:")
for i, (y, x) in enumerate(ra_dec, start=1):
    print(f"  Source {i:>2d}: x = {x:.2f}, y = {y:.2f}")

print("\nDetected source sky coordinates:")
for i, sky in enumerate(world_coords, start=1):
    # in decimal degrees:
    print(f"  Source {i:>2d}: RA = {sky.ra.deg:.6f}°,  Dec = {sky.dec.deg:.6f}°")

print("\nBack to LSST pixel coords:")
for i, (x_lsst, y_lsst) in enumerate(lsst_pixel_coords, start=1):
    print(f" Source {i}:  x = {x_lsst:.2f},  y = {y_lsst:.2f}")

print("\nBack to Roman pixel coords:")
for i, (x_roman, y_roman) in enumerate(roman_pixel_coords, start=1):
    print(f" Source {i}:  x = {x_roman:.2f},  y = {y_roman:.2f}")


# draw the observation into the current axes
lsst_obs_sc2 = observations_sc2[:-n_roman]
roman_obs_sc2 = observations_sc2[-n_roman:]

scarlet2.plot.observation(
    observation=observations[48],
    norm=None,
    channel_map=None,
    show_psf=False,
    add_labels=False,
)

# get current axis
ax = plt.gca()

# loop over sources and draw both red circle and index label
for i, (x0, y0) in enumerate(lsst_pixel_coords):
    x0, y0 = float(x0), float(y0)

    # Draw red circle
    ax.scatter(
        x0, y0,
        s=80,
        edgecolors='red',
        facecolors='none',
        linewidths=2
    )

    # Add index label
    ax.text(
        x0, y0,
        str(i),
        color='red',
        fontsize=5,
        ha='center',
        va='center',
        weight='bold'
    )

plt.show()


# ============================================================================
# 10.2 Get coords of source in center of image (assume it is the lens)
# ============================================================================

'''
Modify sources based on system
--> adapt list sky_coords
--> adapt is extended
'''

#### Check if in center of image:
roman_obs_sc2_on = roman_obs_sc2[-1]
_, H, W   = roman_obs_sc2_on.data.shape               # data.shape = (C, H, WT
center_pix = np.array([W//2, H//2], dtype=int)        # [x_center, y_center] in Roman coords
tol_pix    = 3                                        # how close (in px) counts as “at centre”
is_center = []


center_pix_sky = wcs_roman.pixel_to_world(center_pix[0], center_pix[1])
center_pix_sky_coords = SkyCoord(center_pix_sky.ra, center_pix_sky.dec)
# print(center_pix_sky_coords, '\n')
# print(sky_coords_, '\n')


for i, roman_pix_coords in enumerate(roman_pixel_coords):
    is_center.append(np.all(np.abs(roman_pix_coords - center_pix) <= tol_pix))
    print(f"Source {i} at pixel {roman_pix_coords}, centre is {center_pix}, is_center={is_center[i]}")
print('is_center: ', is_center, '\n')


sky_coords = sky_coords_ #+ [center_pix_sky_coords]

#17462 #96679 #1952
if img_index == 96679:
    sky_coords = sky_coords + [sky_coords[0]] # add PS at same location as 3rd source ie lens
    del sky_coords[1]   # delete second source on host to the left
    is_extended = [True, True, False, False]

elif img_index == 17462:
    new_source_pixel_x = 24.97
    new_source_pixel_y = 24.97
    new_sky_coord = wcs_roman.pixel_to_world(new_source_pixel_x, new_source_pixel_y)
    sky_coords = sky_coords + [new_sky_coord]
    is_extended = [True, True, False, False]

    


print(sky_coords, '\n')
for i, sky_coord in enumerate(sky_coords):
    print(f"Source {i} at pixel {wcs_roman.world_to_pixel(sky_coord)}, centre is {center_pix}")

print(is_extended)


scarlet2.plot.observation(
    observation=observations[48],
    norm=None,
    channel_map=None,
    show_psf=False,
    add_labels=False,
)

# get current axis
ax = plt.gca()

# # loop over sources and draw both red circle and index label
for i, sky_coord_i in enumerate(sky_coords):
    x0, y0 = wcs_roman.world_to_pixel(sky_coord_i)
    x0, y0 = float(x0), float(y0)

    # Draw red circle
    if i ==3:
        ax.scatter(
            x0, y0,
            s=80,
            edgecolors='red',
            facecolors='none',
            linewidths=1
    )
    else:  
        ax.scatter(
            x0, y0,
            s=80,
            edgecolors='red',
            facecolors='none',
            linewidths=1.5
        )

    # Add index label
    if i ==3:
        ax.text(
            x0, y0,
            str(i),
            color='red',
            fontsize=5,
            ha='center',
            va='center',
            weight='bold'
        )
    else:
        ax.text(
            x0, y0,
            str(i),
            color='red',
            fontsize=5,
            ha='center',
            va='center',
            weight='bold'
        )

# plt.savefig(str(img_index)+'_source_detection_roman.png')
plt.show()

print(channels)
# print(epochs)
print(channels_on)

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

# ============================================================================
# 10.3 Source initialization (point vs extended + Gaussian‐moment sizes)
# ============================================================================
## Build the Scarlet Scene with one Source per detection

lens_indx = 1

with Scene(model_frame) as scene:
    for i, coords  in enumerate(sky_coords):

        ##############################################################        
        ##############################################################
        ## EXTENDED SOURCE (e.g. host galaxy) 
        if is_extended[i]:
            print(f"Source {i}: EXTENDED ")
            if i == lens_indx: 
                print('lens')
            else:
                print('host')
            try: 
                print('trying from_gaussian_moments')
                x0_roman, y0_roman = wcs_roman.world_to_pixel(coords)
                x0_roman, y0_roman = float(x0_roman), float(y0_roman)
                spectrum_roman, morphology_roman = init.from_gaussian_moments(
                    roman_obs_sc2, 
                    (x0_roman, y0_roman),
                    min_snr = 0.001
                )
                x0_lsst, y0_lsst = wcs_lsst.world_to_pixel(coords)
                x0_lsst, y0_lsst = float(x0_lsst), float(y0_lsst)
                spectrum_lsst, morphology_lsst = init.from_gaussian_moments(
                    lsst_obs_sc2, 
                    (x0_lsst, y0_lsst),
                    min_snr = 0.001
                )
                morphology = morphology_roman
                
                
            ## Use compact_morphology if gaussian moments fails
            except:# IndexError:
                print('Using gaus2d')
                if i == lens_indx:
                    if img_index == 96679:  #added lens box
                        x = np.linspace(-10, 10, 20)
                        y = np.linspace(-10, 10, 20)
                        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
                        morphology = gaus2d(x, y, sx=1.5,sy=1.5)
                    elif img_index == 17462:
                        x = np.linspace(-50, 50, 50)     #physical size on image, number of pixels
                        y = np.linspace(-50, 50, 50)
                        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
                        morphology = gaus2d(x, y, sx=10,sy=10)   #sx,sy = spread i.e std of gaussian
                        
                else:
                    if img_index == 96679:
                        x = np.linspace(-20, 20, 40)
                        y = np.linspace(-20, 20, 40)
                        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
                        morphology = gaus2d(x, y, sx=5, sy=5)
                    elif img_index == 17462:
                        x = np.linspace(-10, 10, 20)
                        y = np.linspace(-10, 10, 20)
                        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
                        morphology = gaus2d(x, y, sx=2,sy=2)
                        
                spectrum_lsst = init.pixel_spectrum(lsst_obs_sc2, coords)
                spectrum_roman = init.pixel_spectrum(roman_obs_sc2, coords)

            
            print('morphology.shape before padding: ', morphology.shape)
            # initialize with  Starlet Array Morphology
            if i == lens_indx:                      #lens
                pad = int((40 - morphology.shape[0])/2)
            else:                                           #host
                pad = int((20 - morphology.shape[0])/2)
            pad = np.clip(pad, a_min=0, a_max = 100)
            morph = np.pad(morphology,(pad, pad), 'constant')
           
            # morph = scarlet2.StarletMorphology.from_image(morph)
            morph = jnp.asarray(morph)
            
            print('morphology.shape after padding: ', morph.shape)
            
            ## join lsst and roman fluxes
            fluxes = 3*np.hstack((spectrum_lsst,spectrum_roman))
            fluxes[fluxes<1e-6]=1e-1
            
            ## Fit StaticArraySpectrum to source
            static_spectrum = scarlet2.StaticArraySpectrum(
                    fluxes,
                    bands=bandall,   #[g, r, i, G, R, I]
                    band_selector=band_selector
                )  
            scarlet2.Source(coords, static_spectrum, morph)
        
        ########################################################################
        # POINT SOURCE (transient/SN) —
        else:
            print(f"Source {i}: POINT SOURCE")
            spectrum_lsst = init.pixel_spectrum(lsst_obs_sc2, coords)  #coords in sky coords ?, include correct_psf ?
            spectrum_roman = init.pixel_spectrum(roman_obs_sc2, coords)  #coords in sky coords ?, include correct_psf ?
            # the above returns arrays containing fluxes for ALL images, but sets the roman/lsst image fluxes to 0 in spectrum_lsst/roman
            spectrum_lsst = spectrum_lsst[:n_lsst]
            spectrum_roman = spectrum_roman[n_lsst:]
            flux_var = np.hstack((spectrum_lsst, spectrum_roman))
            flux_var[flux_var<1e-6]=1e-1  #from RubinRoman_TD
            flux_var[np.isnan(flux_var)]=1e-1 #from RubinRoman_TD

            # Set off epoch flux to 0 (small value)
            for j, (band, epoch) in enumerate(channels):
                if epoch not in epochs_on:
                    flux_var[j] = 1e-10  # Or 0.0 
    

            # Build mask of al the “off‐SN” channels
            # Set flux for off channels to 1e-20 (almost 0, avoid division w 0 later on)
            # off_mask = [i for i,ch in enumerate(channels) if ch not in channels_on]
            flux = jnp.asarray(3*flux_var)
            print(flux)
            # off_idx = jnp.array(off_mask, dtype=jnp.int32)
            # flux = flux.at[off_idx].set(1e-20)
            
            transient_spectrum = scarlet2.TransientArraySpectrum(
                    flux,
                    epochs=epochs_on,#epochs, # all epochs or channels_zeroed?
                    epoch_selector=epoch_selector
                )
            scarlet2.PointSource(coords, transient_spectrum)
            

# By now `scene.sources` contains your point/transient sources and extended source(s).
print(scene)

import cmasher as cmr
cmap = cmr.lilac
int_method='none'
fig, axes = plt.subplots(1, len( scene.sources ), figsize=(15,6),dpi=120)
for i, ax in enumerate(axes):
    if is_extended[i]==True:#i!=indtransient:
        y = np.array(scene.sources[i].morphology)  # should work if it's already 2D or 3D
    else:
        y = scene.sources[i].morphology()
    ax.imshow(y, cmap = cmap,interpolation=int_method)#,vmin = np.max([np.min(np.log(y)[np.log(y)>-15]),-15]))
    ax.set_title(f"source {i}", fontsize = 18)
    ax.invert_yaxis()
# plt.savefig(str(img_index)+'_source_models.png')
plt.show()
plt.clf()

#print(obs_hsc.data.shape,obs_hst.data.shape)
#print(frameall.channels,channels_zeroed)
#print(obs_hsc.render(scene()).shape,obs_hst.render(scene()).shape)
scarlet2.plot.scene(
    scene,
    observation=observations_sc2[48],
    norm=None,
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

# plt.savefig(str(img_index)+'_scene_prefit_roman.png')
plt.show()

# Plot the model, for each epoch
scarlet2.plot.scene(
    scene,
    observation=observations_sc2[3],
    norm=None,
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
# plt.savefig(str(img_index)+'_scene_prefit_rubin.png')
plt.show()


pos_step = 1e-4                                                  #change if SN positions look bad
morph_step = lambda p: scarlet2.relative_step(p, factor=1e-4)     #change factor if host/lens looks bad
SED_step = lambda p: scarlet2.relative_step(p, factor=1e-4)       #change factor if SN flux looks bad

# parameters = scene.make_parameters()   #old syntax

for obs in observations_sc2:
    # print(obs.renderer)
    with scarlet2.Parameters(obs):
        scarlet2.Parameter(obs.renderer[1].shift, name="shift", stepsize=1e-4 * u.arcsec)    



with Parameters(scene):   #as parameters:  --> old syntax
    for source_indx in range(len(scene.sources)):
    
        print(source_indx)
        
        scarlet2.Parameter(scene.sources[source_indx].spectrum.data, 
                                         name=f"spectrum.{source_indx}", 
                                         constraint=constraints.positive, stepsize=SED_step
                                        )   #are constrains supposed to be included? no in scarlet2 docs
    
        if is_extended[source_indx]:
            #Static host galaxy parameters; or lens
            scarlet2.Parameter(scene.sources[source_indx].morphology, #morphology.data or morphology.coeffs
                                             name=f"morph.{source_indx}", stepsize=morph_step
                                            )#, prior=prior)
            
        else:
            #Transient point source parameters
            scarlet2.Parameter(scene.sources[source_indx].center, 
                                             name=f"center.{source_indx}", 
                                             constraint=constraints.positive, stepsize=pos_step
                                            )
             


# Fit the scene
stepnum = 7000
# scene_ = scene.fit(observations_sc2, parameters, max_iter=stepnum, e_rel=1e-4, progress_bar=True)  #old synatx
scene_= scarlet2.fit(scene, observations_sc2, max_iter=stepnum, e_rel=1e-7, progress_bar=True)


# Plot the model, for each epoch
scarlet2.plot.scene(
    scene_,
    observation=observations_sc2[48],
    norm=None,
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
plt.show()


# Plot the model, for each epoch
scarlet2.plot.scene(
    scene_,
    observation=observations_sc2[38],
    norm=None,
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
plt.show()


# fig, axes = plt.subplots(1, len( scene.sources ), figsize=(15,6),dpi=120)
# for i, ax in enumerate(axes):
#     if True:#i!=indtransient:
#         y = scene_.sources[i].morphology#[0]
#     else:
#         y = scene_.sources[i].morphology.data
#     ax.imshow(y, cmap = cmap,interpolation=int_method)#,vmin = np.max([np.min(np.log(y)[np.log(y)>-15]),-15]))
#     ax.set_title(f"source {i}", fontsize = 18)
#     ax.invert_yaxis()
# plt.show()
# plt.clf()



cmap = cmr.lilac
int_method='none'
fig, axes = plt.subplots(1, len( scene_[0].sources ), figsize=(15,6),dpi=120)
for i, ax in enumerate(axes):
    if is_extended[i]==True:#i!=indtransient:
        y = np.array(scene_[0].sources[i].morphology)  # should work if it's already 2D or 3D
    else:
        y = scene_[0].sources[i].morphology()
    ax.imshow(y, cmap = cmap,interpolation=int_method)#,vmin = np.max([np.min(np.log(y)[np.log(y)>-15]),-15]))
    ax.set_title(f"source {i}", fontsize = 18)
    ax.invert_yaxis()
# plt.savefig(str(img_index)+'_source_models_postfit.png')
plt.show()
plt.clf()
    