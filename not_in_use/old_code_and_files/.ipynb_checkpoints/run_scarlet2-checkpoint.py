import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
matplotlib.rc('image', cmap='gray', interpolation='none', origin='lower')

import jax.numpy as jnp
from jax import random, jit
import jax


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

# ============================================================================
# 1. General File
# ============================================================================
img_index = 1952
t_start = 56170   #start epoch I think? #mjdstart in rubinromanTD notebook
# times = np.arange(56150,56350,2)
t_off   = 56150
t_on    = 56200

times = [56150, 56200]
epochs = [str(i) for i in range(len(times))] # basically just list of index keys correspond to each time?

path = '/scratch/network/vk9342/lenstronomy-tutorials/Notebooks/JP_spring_2025_modified_code/outputs/SNsims_TD_noise/'
data_path  = path + 'img_' + str(img_index) + '/'
psf_path   = '/scratch/network/vk9342/lenstronomy-tutorials/Notebooks/JP_spring_2025_modified_code/'

bands     = ["g","r","i"]
BANDS = ['G', 'R', 'I']
bandall = ['g','r','i','G','R','I']

channels = []
channels_on = []
observations = []

n_lsst  = len(bands) * len(times)   # e.g. 3 bands × (len(times) you picked)
n_roman  = len(BANDS) * 2   # 3 bands × 2 epochs = 6

# ============================================================================
# 2. PSF File paths for low Res Rubin
# ============================================================================
psf_lsst_data  = fits.open(psf_path+'psf_LSST.fits')[0].data.astype('float32')     
Np1, Np2        = psf_lsst_data.shape
psf_lsst_      = jnp.asarray([psf_lsst_data])  #jnp.asarray([psf_lsst_data,psf_lsst_data,psf_lsst_data])
psf_lsst       = scarlet2.ArrayPSF(psf_lsst_)

# ============================================================================
# 3. Load Observations for each Rubin epoch
# ============================================================================
for band in bands:
    for ind, epoch in enumerate(times):
        channel_sc2 = (band, str(ind)) 
        # save epoch ind corresponding to Roman images to append to channels later 
        if epoch == t_on:
            ind_t_on = str(ind)
        elif epoch == t_off:
            ind_t_off = str(ind)
        
        # get channels for only 'on' times
        if epoch >= t_start and epoch<t_start+400:  #Why 400 specifically?
            channels_on.append(channel_sc2)
        
        #image files
        data_file_name = 'image_LSST_'+band+'_'+str(img_index)+'_newSN_'+str(epoch)+'.fits' #is this pre psf data?
        img = data_path+ data_file_name

        obs_hdu = fits.open(img)
        data_lsst = obs_hdu[0].data
        N1, N2 = data_lsst.shape
        data_lsst = data_lsst.reshape(1, N1, N2) #reshape because?
        obs_hduw = fits.open(os.path.join(data_path, 'image_LSST_'+band+'_wcs_'+str(img_index)+'_newSN_'+str(epoch)+'.fits')) # is this post psf data?
        wcs_lsst = WCS(obs_hduw[0].header)
        data_lsst = jnp.array(np.asfarray(data_lsst).byteswap().newbyteorder(), jnp.float32) 

        #What should my weights be???????
        obs_lsst = scarlet2.Observation(data_lsst,
                                      wcs=wcs_lsst,
                                      psf=psf_lsst,
                                      channels=[channel_sc2],
                                      weights=None)
        observations.append(obs_lsst)

        channels.append(channel_sc2)

# ============================================================================
# 4. PSF file paths for high Res Roman
# ============================================================================
psf_roman = fits.open(psf_path + "psf_Roman.fits")[0].data.astype('float32')  
psf_roman = jnp.array(psf_roman, jnp.float32)

psf_roman_single = jnp.asarray([psf_roman]) 
psf_roman_single  = scarlet2.ArrayPSF(psf_roman_single)

psf_roman_ = jnp.asarray([psf_roman])  #jnp.asarray([psf_roman,psf_roman,psf_roman])
# Why line below for roman psf, but not rubin?
psf_roman_ = (psf_roman_-np.min(psf_roman_))/(np.max(psf_roman_)-np.min(psf_roman_)) 
psf_roman = scarlet2.ArrayPSF(psf_roman_)


# ============================================================================
# 5. Load high-res off/on images for Roman
# ============================================================================

t_on    = 56200
t_off   = 56150

#Read in a single epoch of high resolution imaging WITHOUT the SN
for band, BAND in zip(bands, BANDS):
    # Load the HST image data
    channel_sc2 = (BAND, ind_t_off)
    
    roman_hdu = fits.open(os.path.join(data_path, "image_Roman_"+band+"_"+str(img_index)+"_newSN_"+str(t_off)+".fits"))
    data_roman = roman_hdu[0].data
    N1, N2 = data_roman.shape
    data_roman = data_roman.reshape(1, N1, N2)
    roman_hdu = fits.open(os.path.join(data_path, "image_Roman_"+band+"_wcs_"+str(img_index)+"_newSN_"+str(t_off)+".fits"))
    wcs_roman = WCS(roman_hdu[0].header)
    
    
    data_roman_offSN = jnp.array(np.asfarray(data_roman).byteswap().newbyteorder(), jnp.float32)
    # Scale the HST data
    n,n1, n2 = jnp.shape(data_roman)
    #data_hst *= data_hsc.max() / data_hst.max()
    N,N1, N2 = data_roman.shape

    # define two observation packages and match to frame
    obs_roman_offSN = scarlet2.Observation(data_roman,
                                   wcs=wcs_roman,
                                   psf=psf_roman,
                                   channels=[channel_sc2],
                                   weights=None) #should weights be none?    
    
    channels.append(channel_sc2)
    observations.append(obs_roman_offSN)



#Read in a single epoch of high resolution imaging WITH the SN
for band, BAND in zip(bands, BANDS):

    channel_sc2 = (BAND, ind_t_on)
    
    roman_hdu = fits.open(os.path.join(data_path, "image_Roman_"+band+"_"+str(img_index)+"_newSN_"+str(t_on)+".fits"))
    data_roman = roman_hdu[0].data
    N1, N2 = data_roman.shape
    data_roman = data_roman.reshape(1, N1, N2)
    roman_hdu = fits.open(os.path.join(data_path, "image_Roman_"+band+"_wcs_"+str(img_index)+"_newSN_"+str(t_off)+".fits"))  
    wcs_roman = WCS(roman_hdu[0].header)
    print(wcs_roman)
    data_roman = jnp.array(np.asfarray(data_roman).byteswap().newbyteorder(), jnp.float32)
    # Load the HST PSF data
    
    
    if float(epoch)>t_start and float(epoch)<t_start+400:
        channels_on.append(channel_sc2)

    # define two observation packages and match to frame
    obs_roman_postSN = scarlet2.Observation(data_roman,
                                   wcs=wcs_roman,
                                   psf=psf_roman,
                                   channels=[channel_sc2],
                                   weights=None) #should weights be none? 
    
    channels.append(channel_sc2)
    observations.append(obs_roman_postSN)

#How many observations did we end up reading in? 
print(len(observations))
# ============================================================================
# 6. Make Frame and match observations
# ============================================================================
model_frame_psf = scarlet2.GaussianPSF(0.7)
model_frame = scarlet2.Frame.from_observations(
    observations=observations,
    coverage="union",  # or "intersection"
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
        img = np.asfarray(observations[ind].data)
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
lvl=1
wave=2
SNR=2.0

lsst_obs = observations[:-n_roman]
roman_obs = observations[-n_roman:]

observations_sc2=[]
normsingle=[]

# ============================================================================   
# DETECT SOURCES BASED ONLY ON ROMAN IMGS
catalog_single_roman, bgsingle_roman, detectsingle_roman, globalback_roman = makeCatalog(roman_obs, lvl, wave)
pixel = np.stack((catalog_single_roman['y'], catalog_single_roman['x']), axis=1)
ra_dec = pixel
print(ra_dec.shape)
# ============================================================================

#################################################################################
### MAKE observations_sc2 ###########################
bgarr_roman     = np.asfarray(bgsingle_roman)
bgarrall_roman  = np.hstack((bgarr_roman.flatten(),np.asfarray(bgsingle_roman).flatten()))
bgarr_roman     = np.asfarray(bgsingle_roman)
print(bgarrall_roman.shape)

catalog_single_lsst, bgsingle_lsst, detectsingle_lsst, globalback_lsst = makeCatalog(lsst_obs, lvl, wave)
bgarr_lsst    = np.asfarray(bgsingle_lsst)
bgarrall_lsst = np.hstack((bgarr_lsst.flatten(),np.asfarray(bgsingle_lsst).flatten()))
bgarr_lsst    = np.asfarray(bgsingle_lsst)
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
    weights2=np.ones(obs2.data.shape) / (10*bg**2)#[:, None, None]  #??in RubinRoman notebook insteads of bg it uses bgarr_all, is this a typo?
    data = obs2.data-back   #Subtract background flux

    print(channels[ind])
    
    obs_sc2 = scarlet2.Observation(jnp.asarray(data), jnp.asarray(weights2), psf=psf,channels=[channels[ind]],wcs=w)
    observations_sc2.append(obs_sc2)
    
    #Store norm based on observation data
    normsingle.append(
        AsinhPercentileNorm(
            jnp.asarray(np.asarray((obs2.data[:,10:-10,10:-10]),dtype=np.float32)),
            percentiles=[0.001, 50, 80])
    )


# # ============================================================================   
# # obssinglearr=np.asarray(observations) --> not used anywhere?
# bgarr_roman     = np.asfarray(bgsingle_roman)
# bgarrall_roman  = np.hstack((bgarr_roman.flatten(),np.asfarray(bgsingle_roman).flatten()))
# bgarr_roman     = np.asfarray(bgsingle_roman)
# print(bgarrall_roman.shape)

# normsingle=[]

# for ind,(obs2,bg,back) in enumerate(zip(roman_obs,bgarr_roman,globalback_roman)):
#     w = obs2.frame.wcs
#     weights2=np.ones(obs2.data.shape) / (10*bgarr_roman**2)[:, None, None]
#     #Subtract background flux
#     data = obs2.data-back
    
#     obs_sc2 = scarlet2.Observation(jnp.asarray(data), jnp.asarray(weights2), psf=psf_roman,channels=[channels[ind]],wcs=w)
#     observations_sc2.append(obs_sc2)
    
#     #Store norm based on observation data
#     normsingle.append(
#         AsinhPercentileNorm(
#             jnp.asarray(np.asarray((obs2.data[:,10:-10,10:-10]),dtype=np.float32)),
#             percentiles=[0.001, 50, 80])
#     )
#     ## the 50 above is random, it gave an error asking for a third number



model_frame = scarlet2.Frame.from_observations(
    observations=observations_sc2,
    coverage="union",  # or "intersection"
)
for obs in observations_sc2:
    obs.match(model_frame)

# ============================================================================
# 9. Get and visualize sources in image
# ============================================================================
world_coords = []
for y, x in ra_dec:
    sky = wcs_roman.pixel_to_world(x, y)   # note: pixel_to_world(x, y)
    world_coords.append(SkyCoord(sky.ra, sky.dec))

roman_pixel_coords = [wcs_roman.world_to_pixel(sky) for sky in world_coords]
lsst_pixel_coords = [wcs_lsst.world_to_pixel(sky) for sky in world_coords]

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
    observation=observations[0],
    norm=None,
    channel_map=None,
    show_psf=False,
    add_labels=False,
)

# # grab the current axes
ax = plt.gca()
# ax.imshow(obs2.data[4], origin='lower')

for x0, y0 in lsst_pixel_coords:
    x0, y0 = float(x0), float(y0) 
    ax.scatter(
        x0, y0,
        s=80,
        edgecolors='red',
        facecolors='none',
        linewidths=2
    )
plt.show()


print(channels)
print(epochs)

# ============================================================================
# 10. Source initialization
# ============================================================================
lsst_obs_sc2 = observations_sc2[:-n_roman]
roman_obs_sc2 = observations_sc2[-n_roman:]

band_selector  = lambda ch: ch[0]
epoch_selector = lambda ch: ch[1]

## Ra and dec centers from SEP catalog
sky_coords = world_coords

# ============================================================================
# 10.1 Classify if source is extended/point source based on PSF
# ============================================================================


##  Measure each detection's “size” via Gaussian moments
## Determine whether > or < than PSF
## if > then PSF then categorize as extended source
## if < than PSF then categorize as point source
## IS THE BELOW CORRECT ??????????    
sizes      = []
is_extended = []
psf_sigma = model_frame.psf.morphology.size   # sigma of GaussianPSF in pixels
for coords in sky_coords:
    try:
        with Scene(model_frame):  # should I only do roman here??
            coords_list = [coords] * n_roman
            #?? is coords supposed to be a list of touples of length len (obs) because I get an error if I just put one touple??
            roman_spectrum, roman_morphology = init.from_gaussian_moments(roman_obs_sc2, coords_list) 
        
        morph = np.array(roman_morphology.data)
        M00 = morph.sum()                            # zeroth moment
        ys, xs = np.indices(morph.shape)             # pixel grid
        xc = (xs * morph).sum() / M00                  # 1st moment → centroid x
        yc = (ys * morph).sum() / M00                  # 1st moment → centroid y
        Mxx = ((xs - xc)**2 * morph).sum() / M00     # 2nd central moment
        Myy = ((ys - yc)**2 * morph).sum() / M00     # 2nd central moment
        sigma = np.sqrt(0.5 * (Mxx + Myy))         # mean 1σ size
        sizes.append(sigma)
        ## extended if > PSF size by 20%
        is_extended.append(sigma > psf_sigma * 1.2)
    except ValueError:
        ## Gaussian fit failed: treat as point
        sizes.append(None)
        is_extended.append(False)
        print('Gaussian fit failed')



# ============================================================================
# 10.2 Get coords of source in center of image (assume it is the lens)
# ============================================================================

#### Check if in center of image:
roman_obs_sc2_on = roman_obs_sc2[-1]
_, H, W   = roman_obs_sc2_on.data.shape               # data.shape = (C, H, W)
center_pix = np.array([W//2, H//2], dtype=int)        # [x_center, y_center]
tol_pix    = 3                                        # how close (in px) counts as “at centre”
is_center = []
for i, roman_pix_coords in enumerate(roman_pixel_coords):
    is_center.append(np.all(np.abs(roman_pix_coords - center_pix) <= tol_pix))
    print(f"Source {i} at pixel {roman_pix_coords}, centre is {center_pix}, is_center={is_center[i]}")
print('is_center: ', is_center)

x,y = roman_pixel_coords[0]
print(x, y)
roman_coords = (float(x), float(y))
print(roman_coords)

# ============================================================================
# 10.3 Source initialization (point vs extended + Gaussian‐moment sizes)
# ============================================================================
## Build the Scarlet Scene with one Source per detection

with Scene(model_frame) as scene:
    for i, coords  in enumerate(sky_coords):
        
        ##############################################################
        ## EXTENDED SOURCE (e.g. host galaxy) 
        if is_extended[i]:
            print(f"Source {i}: EXTENDED (σ ≃ {sizes[i]:.2f} px)")
            try: # Should I only try with Roman?
                # coords_list = [roman_pixel_coords[i]] * n_roman
                spectrum_roman, morphology_roman = init.from_gaussian_moments(
                    roman_obs_sc2, 
                    coords
                    #box sizes = ?? do we want or nah
                )
                # coords_list = [lsst_pixel_coords[i]] * n_lsst
                spectrum_lsst, morphology_lsst = init.from_gaussian_moments(
                    lsst_obs_sc2, 
                    coords
                    #box sizes = ?? do we want or nah
                )
                mmorphology = orphology_roman
            ## Use compact_morphology if gaussian moments fails
            except IndexError:
                morphology = scarlet2.init.compact_morphology(
                    # min = ?, do we want??
                    # max = ?, do we want??
                    ## api has min and max, but example from guide does not include
                )
            
                spectrum_lsst = init.pixel_spectrum(lsst_obs_sc2, coords)
                spectrum_roman = init.pixel_spectrum(roman_obs_sc2, coords)
 
            ## initialize with  Starlet Array Morphology
            morphology = scarlet2.StarletMorphology.from_image(morphology)
            # Do I need some kind of normalization or smth for morph or can I just leave it at this?

            ## join lsst and roman fluxes
            fluxes = np.hstack((spectrum_lsst,spectrum_roman))
            fluxes[fluxes<1e-6]=1e-1
            
            ## Fit StaticArraySpectrum to source
            static_spectrum = scarlet2.StaticArraySpectrum(
                    fluxes,
                    bands=bandall,   #[g, r, i, G, R, I]
                    band_selector=band_selector
                )  
            scarlet2.Source(coords, static_spectrum, morphology)

        ###########################################################################
        # if not classified as extended, but it's at the centre --> force to be lens
        
        elif is_center[i]:
            print(f"Source {i}: CENTER → forcing as LENS")
            is_center[i] = True  # change value to extended
            try: # Should I only try with Roman?
                spectrum_roman, morphology_roman = init.from_gaussian_moments(
                    roman_obs_sc2, 
                    coords
                    #box sizes = ?? do we want or nah
                )
                
                spectrum_lsst, morphology_lsst = init.from_gaussian_moments(
                    lsst_obs_sc2, 
                    coords
                    #box sizes = ?? do we want or nah
                )
                morphology = morphology_roman
                
            ## Use compact_morphology if gaussian moments fails
            except IndexError:
                morphology = scarlet2.init.compact_morphology(
                    # min = ?, do we want??
                    # max = ?, do we want??
                    ## api has min and max, but example from guide does not include
                )
            
                spectrum_lsst = init.pixel_spectrum(lsst_obs_sc2, coords)
                spectrum_roman = init.pixel_spectrum(roman_obs_sc2, coords)
 
            ## initialize with  Starlet Array Morphology
            morphology = scarlet2.StarletMorphology.from_image(morphology)
            # Do I need some kind of normalization or smth for morph or can I just leave it at this?

            ## join lsst and roman fluxes
            fluxes = np.hstack((spectrum_lsst,spectrum_roman))
            fluxes[fluxes<1e-6]=1e-1
            
            ## Fit StaticArraySpectrum to source
            static_spectrum = scarlet2.StaticArraySpectrum(
                    fluxes,
                    bands=bandall,   #[g, r, i, G, R, I]
                    band_selector=band_selector
                )  
            scarlet2.Source(coords, static_spectrum, morphology)

        
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
            flux_var[flux_var<1e-6]=1e-6  #from RubinRoman_TD
            flux_var[np.isnan(flux_var)]=1e-6 #from RubinRoman_TD
    
            # flux = jnp.asarray(flux)

            # Build mask of al the “off‐SN” channels
            # Set flux for off channels to 1e-20 (almost 0, avoid division w 0 later on)
            # off_mask = [i for i,ch in enumerate(channels) if ch not in channels_on]
            flux = jnp.asarray(flux_var)
            # off_idx = jnp.array(off_mask, dtype=jnp.int32)
            # flux = flux.at[off_idx].set(1e-20)

            transient_spectrum = scarlet2.TransientArraySpectrum(
                    flux,
                    epochs=epochs, # all epochs or channels_zeroed?
                    epoch_selector=epoch_selector
                )
            scarlet2.PointSource(coords, transient_spectrum)
            

# By now `scene.sources` contains your point/transient sources and extended source(s).
print(scene)

print(spectrum_lsst)
print(spectrum_roman)
print("→ channels in observations_sc2:")
for obs in lsst_obs_sc2:
   print(obs.frame.channels)
print(flux)


print("flux.shape =", flux.shape)   # should print (2, 6)
print("spectrum.data.shape =", transient_spectrum.data.shape)  # should be (2,6)
print("frame channels:", model_frame.channels)
for ch in model_frame.channels:
    print(ch, "→ band_selector:", band_selector(ch),
                 " epoch_selector:", epoch_selector(ch))


import cmasher as cmr
cmap = cmr.lilac
int_method='none'
fig, axes = plt.subplots(1, len( scene.sources ), figsize=(15,6),dpi=120)
for i, ax in enumerate(axes):
    if True:#i!=indtransient:
        y = scene.sources[i].morphology()#[0]
    else:
        y = scene.sources[i].morphology.data
    ax.imshow(y, cmap = cmap,interpolation=int_method)#,vmin = np.max([np.min(np.log(y)[np.log(y)>-15]),-15]))
    ax.set_title(f"source {i}", fontsize = 18)
    ax.invert_yaxis()
plt.show()
plt.clf()


#print(obs_hsc.data.shape,obs_hst.data.shape)
#print(frameall.channels,channels_zeroed)
#print(obs_hsc.render(scene()).shape,obs_hst.render(scene()).shape)
scarlet2.plot.scene(
    scene,
    observation=observations_sc2[0],
    norm=None,#normsingle[0],
    channel_map=None,
    show_model=False,
    show_observed=True,
    show_rendered=True,
    show_residual=True,
    add_labels=True,
    add_boxes=True
)
plt.show()


pos_step = 1e-2
morph_step = lambda p: scarlet2.relative_step(p, factor=1e-3)
SED_step = lambda p: scarlet2.relative_step(p, factor=5e-2)  #factor=1e-3 or 1e-5

parameters = scene.make_parameters()

for source_indx in range(len(scene.sources)):
    
    parameters += scarlet2.Parameter(scene.sources[source_indx].spectrum.data, 
                                     name=f"spectrum.{source_indx}", 
                                     constraint=constraints.positive, stepsize=SED_step
                                    )   #are constrains supposed to be included? no in scarlet2 docs

    if is_extended[source_indx]:
        #Static host galaxy parameters; or lens
        parameters += scarlet2.Parameter(scene.sources[source_indx].morphology.data, #morphology.data or morphology.coeffs
                                         name=f"morph.{source_indx}", stepsize=morph_step
                                        )#, prior=prior)
        
    else:
        #Transient point source parameters
        parameters += scarlet2.Parameter(scene.sources[source_indx].center, 
                                         name=f"center.{source_indx}", 
                                         constraint=constraints.positive, stepsize=pos_step
                                        )
         
    

# Fit the scene
stepnum = 100
scene_ = scene.fit(observations_sc2, parameters, max_iter=stepnum, e_rel=1e-4, progress_bar=True)

print("----------------- {}".format(channels))
for k, src in enumerate(scene_.sources):
    print("Source {}, Fluxes: {}".format(k, scarlet2.measure.flux(src)))


