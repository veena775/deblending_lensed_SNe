#!/usr/bin/env python
# coding: utf-8

# # Simulating different telescopes
# This notebooks provides examples in how to use the lenstronomy.SimulationAPI modules in simulating (realistic) mock lenses taylored to a specific observation and instrument and makes a montage of different telescope settings currently available.
# 
# The module enables to use the astronomical magnitude conventions and can translate those into the lenstronomy core module configurations.

# In[1]:


import copy
import h5py
import os
import numpy as np
import scipy
import sncosmo
from PIL import Image
import galsim
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt

from astropy.table import Table
# make sure lenstronomy is installed, otherwise install the latest pip version
try:
    import lenstronomy
except:
    get_ipython().system('pip install lenstronomy')
from astropy.io import fits
from astropy import wcs
import galsim

# lenstronomy module import
from lenstronomy.Util import image_util, data_util, util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Plots.plot_util import coordinate_arrows, scale_bar
from lenstronomy.SimulationAPI.point_source_variability import PointSourceVariability

from astropy.table import Table
import sncosmo

def simulate_rgb_LS4(size, kwargs_LS4, time, source_x=0,source_y=0,returnsqrt = False):
   
    kwargs_g_band = kwargs_LS4[0]
    kwargs_r_band = kwargs_LS4[1]
    kwargs_i_band = kwargs_LS4[2]
 
    #print(kwargs_b_band) #'pixel_scale' : kwargs_b_band['pixel_scale']
    # set number of pixels from pixel scale
    pixel_scale = kwargs_g_band['pixel_scale']
    numpix = int(round(size / pixel_scale))
    #kwargs_model['pixel_scale'] = kwargs_b_band['pixel_scale']
    sim_i = SimAPI(numpix=numpix, kwargs_single_band=kwargs_i_band, kwargs_model=kwargs_model_time_var)
    sim_g = SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=kwargs_model_time_var)
    sim_r = SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model_time_var)

    # return the ImSim instance. With this class instance, you can compute all the
    # modelling accessible of the core modules. See class documentation and other notebooks.
    imSim_i = sim_i.image_model_class(kwargs_numerics)
    imSim_g = sim_g.image_model_class(kwargs_numerics)
    imSim_r = sim_r.image_model_class(kwargs_numerics)


    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g, kwargs_ps_g = sim_g.magnitude2amplitude(kwargs_lens_light_mag_g, kwargs_source_mag_g, kwargs_ps_mag_g)
    kwargs_lens_light_r, kwargs_source_r, kwargs_ps_r = sim_g.magnitude2amplitude(kwargs_lens_light_mag_r, kwargs_source_mag_r, kwargs_ps_mag_r)
    kwargs_lens_light_i, kwargs_source_i, kwargs_ps_i = sim_r.magnitude2amplitude(kwargs_lens_light_mag_i, kwargs_source_mag_i, kwargs_ps_mag_i)

    ps_var_g = PointSourceVariability(source_x, source_y, var_func_g, numpix, kwargs_g_band, kwargs_model_time_var, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag_g, kwargs_lens_light_mag_g, kwargs_ps_mag=kwargs_ps_mag_g)

    ps_var_r = PointSourceVariability(source_x, source_y, var_func_r, numpix, kwargs_r_band, kwargs_model_time_var, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag_r, kwargs_lens_light_mag_r, kwargs_ps_mag=kwargs_ps_mag_r)

    ps_var_i = PointSourceVariability(source_x, source_y, var_func_i, numpix, kwargs_i_band, kwargs_model_time_var, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag_i, kwargs_lens_light_mag_i, kwargs_ps_mag=kwargs_ps_mag_i)
    delays = ps_var_g.delays 
    
    image_g = ps_var_g.image_time(time=time)
    image_r = ps_var_r.image_time(time=time)
    image_i = ps_var_i.image_time(time=time)

    # add noise
    image_i += sim_i.noise_for_model(model=image_i)
    image_g += sim_g.noise_for_model(model=image_g)
    image_r += sim_r.noise_for_model(model=image_r)

    # and plot it

    img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
    #scale_max=10000
    def _scale_max(image): 
        flat=image.flatten()
        flat.sort()
        scale_max = flat[int(len(flat)*0.95)]
        return scale_max
    if returnsqrt==True:
        img[:,:,0] = plot_util.sqrt(image_g, scale_min=0, scale_max=_scale_max(image_g))
        img[:,:,1] = plot_util.sqrt(image_r, scale_min=0, scale_max=_scale_max(image_r))
        img[:,:,2] = plot_util.sqrt(image_i, scale_min=0, scale_max=_scale_max(image_i))
    else:
        img[:,:,0] = image_i
        img[:,:,1] = image_g
        img[:,:,2] = image_r
        
    data_class = sim_i.data_class
    #print(sim_r.psf_class.fwhm)
    psf_i = sim_i.psf_class.kernel_point_source
    psf_g = sim_g.psf_class.kernel_point_source
    psf_r = sim_r.psf_class.kernel_point_source
    psf = np.stack((psf_i,psf_g,psf_r))
    return img, data_class, psf
    

def simulate_rgb(ConfigList, size, kwargs_numerics, time, source_x=0,source_y=0,returnsqrt = False):
    band_g, band_r, band_i = ConfigList
    kwargs_i_band = band_i.kwargs_single_band()
    kwargs_g_band = band_g.kwargs_single_band()
    kwargs_r_band = band_r.kwargs_single_band()
    #print(kwargs_b_band) #'pixel_scale' : kwargs_b_band['pixel_scale']
    # set number of pixels from pixel scale
    pixel_scale = kwargs_g_band['pixel_scale']
    numpix = int(round(size / pixel_scale))
    #kwargs_model['pixel_scale'] = kwargs_b_band['pixel_scale']
    sim_i = SimAPI(numpix=numpix, kwargs_single_band=kwargs_i_band, kwargs_model=kwargs_model_time_var)
    sim_g = SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=kwargs_model_time_var)
    sim_r = SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model_time_var)

    # return the ImSim instance. With this class instance, you can compute all the
    # modelling accessible of the core modules. See class documentation and other notebooks.
    imSim_i = sim_i.image_model_class(kwargs_numerics)
    imSim_g = sim_g.image_model_class(kwargs_numerics)
    imSim_r = sim_r.image_model_class(kwargs_numerics)


    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g, kwargs_ps_g = sim_g.magnitude2amplitude(kwargs_lens_light_mag_g, kwargs_source_mag_g, kwargs_ps_mag_g)
    kwargs_lens_light_r, kwargs_source_r, kwargs_ps_r = sim_g.magnitude2amplitude(kwargs_lens_light_mag_r, kwargs_source_mag_r, kwargs_ps_mag_r)
    kwargs_lens_light_i, kwargs_source_i, kwargs_ps_i = sim_r.magnitude2amplitude(kwargs_lens_light_mag_i, kwargs_source_mag_i, kwargs_ps_mag_i)

    ps_var_g = PointSourceVariability(source_x, source_y, var_func_g, numpix, kwargs_g_band, kwargs_model_time_var, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag_g, kwargs_lens_light_mag_g, kwargs_ps_mag=kwargs_ps_mag_g)

    ps_var_r = PointSourceVariability(source_x, source_y, var_func_r, numpix, kwargs_r_band, kwargs_model_time_var, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag_r, kwargs_lens_light_mag_r, kwargs_ps_mag=kwargs_ps_mag_r)

    ps_var_i = PointSourceVariability(source_x, source_y, var_func_i, numpix, kwargs_i_band, kwargs_model_time_var, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag_i, kwargs_lens_light_mag_i, kwargs_ps_mag=kwargs_ps_mag_i)
    
    delays = ps_var_g.delays
    print('DELAYS',delays,np.max(delays))

    image_g = ps_var_g.image_time(time=time)#imSim_b.image(kwargs_lens, kwargs_source_g, kwargs_lens_light_g, kwargs_ps_g)
    image_r = ps_var_r.image_time(time=time)#imSim_g.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r, kwargs_ps_r)
    image_i = ps_var_i.image_time(time=time)#imSim_r.image(kwargs_lens, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)

    # add noise
    image_i += sim_i.noise_for_model(model=image_i)
    image_g += sim_g.noise_for_model(model=image_g)
    image_r += sim_r.noise_for_model(model=image_r)

    # and plot it

    img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
    #scale_max=10000
    def _scale_max(image): 
        flat=image.flatten()
        flat.sort()
        scale_max = flat[int(len(flat)*0.95)]
        return scale_max
    if returnsqrt==True:
        img[:,:,0] = plot_util.sqrt(image_g, scale_min=0, scale_max=_scale_max(image_g))
        img[:,:,1] = plot_util.sqrt(image_r, scale_min=0, scale_max=_scale_max(image_r))
        img[:,:,2] = plot_util.sqrt(image_i, scale_min=0, scale_max=_scale_max(image_i))
    else:
        img[:,:,0] = image_i
        img[:,:,1] = image_g
        img[:,:,2] = image_r
        
    data_class = sim_i.data_class
    #print(sim_r.psf_class.fwhm)
    psf_i = sim_i.psf_class.kernel_point_source
    psf_g = sim_g.psf_class.kernel_point_source
    psf_r = sim_r.psf_class.kernel_point_source
    psf = np.stack((psf_i,psf_g,psf_r))
    return img, data_class, psf
    


nbands = 3
nepochs = 15
bands = ['lsstg', 'lsstr', 'lssti']

times = np.arange(56150,56350,2)

# ## Define camera and observations
# As an example, we define the camera and observational settings of a LSST-like observation. We define one camera setting and three different observations corresponding to g,r,i imaging.
# 
# For the complete list of possible settings, we refer to the SimulationAPI.observation_api classes. There are pre-configured settings which approximately mimic observations from current and future instruments. Be careful using those and check whether they are sufficiently accurate for your specific science case!

# In[3]:


# Instrument setting from pre-defined configurations

from lenstronomy.SimulationAPI.ObservationConfig.DES import DES
#from lenstronomy.SimulationAPI.ObservationConfig.LSSTsingleobs import LSST
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST

from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman

DES_g = DES(band='g', psf_type='GAUSSIAN', coadd_years=3)
DES_r = DES(band='r', psf_type='GAUSSIAN', coadd_years=3)
DES_i = DES(band='i', psf_type='GAUSSIAN', coadd_years=3)
des = [DES_g, DES_r, DES_i]

LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
LSST_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
LSST_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)
lsst = [LSST_g, LSST_r, LSST_i]

Roman_g = Roman(band='F062', psf_type='PIXEL', survey_mode='wide_area')
Roman_r = Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')
Roman_i = Roman(band='F184', psf_type='PIXEL', survey_mode='wide_area')#'single_exposure')#'wide_area')
roman = [Roman_g, Roman_r, Roman_i]
LS4_g_band_obs = {'exposure_time': 15.,  # exposure time per image (in seconds)
                   'sky_brightness': 22.26,  # sky brightness (in magnitude per square arcseconds)
                   'magnitude_zero_point': 28.30,  # magnitude in which 1 count per second per arcsecond square is registered (in ADU's)
                   'num_exposures': 30,  # number of exposures that are combined
                   'seeing': 1.5,  # full width at half maximum of the PSF (if not specific psf_model is specified)
                   'psf_type': 'GAUSSIAN',  # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
                   'kernel_point_source': None  # 2d numpy array, model of PSF centered with odd number of pixels per axis (optional when psf_type='PIXEL' is chosen)
                  }

LS4_r_band_obs = {'exposure_time': 15.,
                   'sky_brightness': 21.2,
                   'magnitude_zero_point': 28.13,
                   'num_exposures': 30,
                   'seeing': 1.5,
                   'psf_type': 'GAUSSIAN'}

LS4_i_band_obs = {'exposure_time': 15.,
                   'sky_brightness': 20.48,
                   'magnitude_zero_point': 27.79,
                   'num_exposures': 30,
                   'seeing': 1.5,
                   'psf_type': 'GAUSSIAN'}

LS4_camera = {'read_noise': 10,  # std of noise generated by read-out (in units of electrons)
               'pixel_scale': 1.0,  # scale (in arcseonds) of pixels
               'ccd_gain': 4.5  # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              }
# lenstronomy provides these setting to be imported with the SimulationAPI.ObservationConfig routines.

kwargs_g_band_LS4 = util.merge_dicts(LS4_camera, LS4_g_band_obs)
kwargs_r_band_LS4 = util.merge_dicts(LS4_camera, LS4_r_band_obs)
kwargs_i_band_LS4 = util.merge_dicts(LS4_camera, LS4_i_band_obs)
kwargs_LS4 = [kwargs_g_band_LS4,kwargs_r_band_LS4,kwargs_i_band_LS4]

#Where to save plots and fits file
plotdir = '/scratch/gpfs/cw1074/SNsimplots_TD_noise'
storedir = '/scratch/gpfs/cw1074/SNsims_TD_noise'
# import NGC1300 jpg image and decompose it
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
from lenstronomy.Cosmo.micro_lensing import einstein_radius
# find path to data
path = os.getcwd()
dirpath, _ = os.path.split(path)
module_path, _ = os.path.split(dirpath)

def var_func(time, band): #band either desg, desi, desr
    obs = Table({'time': [time],
             'band': [band],          #filters we are observing in
             'gain': [1.],              #dw
             'skynoise': [0],           #depends on exposure time, but will be addig noise later in lenstronomy, could set to 0
             'zp': [30.],               # zero point, what corresponds to 0 flux in this image - pixel values to magnitudes
             'zpsys':['ab']})           # ab magnitudes, units of zero point

    model = sncosmo.Model(source='salt2')
    params = {'z': 0.01, 't0': 56200.0, 'x0':1.e-5, 'x1': 0.1, 'c': -0.1}

    lcs = sncosmo.realize_lcs(obs, model, [params])
    flux = np.array(lcs[0]['flux'])
    # print(lcs[0]['time'])
    # print(flux)
    zero_points = np.array(obs['zp'])
    magnitude = zero_points - 2.5 * np.log10(flux)
    print(time,magnitude)
    return magnitude
var_func_g = lambda t: var_func(t, 'lsstg')
var_func_r = lambda t: var_func(t, 'lsstr')
var_func_i = lambda t: var_func(t, 'lssti')
# read data, this only works if you execute this notebook within the environment of the github repo!
# ## Define model settings
# 
# The model settings are handled by the SimulationAPI.model_api ModelAPI class. 
# The role is to return instances of the lenstronomy LightModel, LensModel, PointSource modules according to the options chosen by the user. Currently, all other model choices are equivalent to the ones provided by LightModel, LensModel, PointSource.
# The current options of the class instance only describe a subset of possibilities and we refer to the specific class instances for details about all the possibilities.
# 
# For this example, we chose a single lens plane and a single source plane, elliptical Sersic profile for the deflector, the interpolated Galaxy as the source and an additional lensed point source.

# In[5]:


# ## Brightness definitions in magnitude space
# One core feature is the support of light profile amplitudes in astronomical magnitude space (at least for few selected well defined brightness profiles).
# 
# We first define all parameters in magnitude space and then use the SimAPI routine to translate the arguments into lenstronomy conventions used by the ImSim module. The second model of each light component we defined as 'INTERPOL', which sets an interpolation grid given an image. This can be used to past real galaxies as lenses or sources into lenstronomy.

# In[69]:


f = h5py.File('lsst-altsched-1a-lowz.h5','r')
simitems = np.asarray(f['system']['block0_items'][()],dtype=str)
simdataall = f['system']['block0_values'][()]
print(simdataall[0],simitems)
rtest_source = [0.944,0.766, 0.678, 0.967, 0.717]
rtest_lens = [0.410, 0.396, 0.344, 0.680, 0.056]

#This is where I match to that table I provided by redshift. You'll want to replace this with a search for the time delay parameter and match by that!
indcatalog =[]
for i in range(5):
    indD = np.argwhere((np.abs(rtest_lens[i]-simdataall[:,7])<0.005)&(np.abs(rtest_source[i]-simdataall[:,6])<0.005))
    if i==0:
        indcatalog=indD[:,0]
    else:
        indcatalog = np.hstack((indcatalog,indD[:,0]))


for ind in indcatalog:
    simdata=simdataall[ind]
    z_lens = simdata[simitems == "zl"][0]
    z_source = simdata[simitems == "zs"][0]
    print('z les',z_lens,'z source',z_source)
    kwargs_model_time_var = {'lens_model_list': ['SIE', 'SHEAR'],  # list of lens models to be used
                'lens_light_model_list': ['SERSIC_ELLIPSE'],  # list of unlensed light models to be used
                'source_light_model_list': ['SERSIC_ELLIPSE'],#['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'point_source_model_list': ['SOURCE_POSITION'],  # list of point source models to be used
                'z_lens': z_lens, 
                         'z_source': z_source
    } 
    Dl = cosmo.angular_diameter_distance(simdata[simitems == "zl"][0]).value
    Ds = cosmo.angular_diameter_distance(simdata[simitems == "zs"][0]).value
    Dls = cosmo.angular_diameter_distance_z1z2(simdata[simitems == "zl"][0],simdata[simitems == "zs"][0]).value
 
    theta_E_sim = simdata[simitems == "theta_e"][0]
    lens_reff = simdata[simitems == "lensgal_reff"][0]
    lens_n = simdata[simitems == "lensgal_n"][0]
    lens_theta = simdata[simitems == "lensgal_theta"][0]
    lens_ellip = simdata[simitems == "lensgal_ellip"][0]
    s = galsim.Shear(e=lens_ellip, beta=lens_theta*galsim.degrees)
    lens_e1 = s.e1
    lens_e2 = s.e2
   
    lens_gamma = simdata[simitems == "gamma"][0]
    gamma_theta = simdata[simitems == "theta_gamma"][0]
    lens_g1 = np.sin(gamma_theta*np.pi/180)*lens_gamma
    lens_g2 = np.cos(gamma_theta*np.pi/180)*lens_gamma
    lens_x = simdata[simitems == "lensgal_x"][0]
    lens_y = simdata[simitems == "lensgal_y"][0]

    host_reff = simdata[simitems == "host_reff"][0]
    host_n = simdata[simitems == "host_n"][0]
    host_theta = simdata[simitems == "host_theta"][0]
    host_ellip = simdata[simitems == "host_ellip"][0]
    try:
        sh = galsim.Shear(e=host_ellip, beta=host_theta*galsim.degrees)
    except:
        continue
    host_e1 = sh.e1
    host_e2 = sh.e2
    host_x = simdata[simitems == "host_x"][0]
    host_y = simdata[simitems == "host_y"][0]
    sn_x = simdata[simitems == "snx"][0]
    sn_y = simdata[simitems == "sny"][0]

    #My failed attempts to extract a meaningful magnitude from whatever units Danny was using
    '''
    lens_amp = 22.5-2.5*np.log10(simdata[simitems == "lensgal_amplitude"][0])
    mu = -2.5*np.log(simdata[simitems == "lensgal_amplitude"][0])
    mag = mu - np.log10(2*np.pi*host_reff*60*60/180*np.pi*1e3)-36.57#mu - 5*np.log10(lens_reff/(Dl/60/60))-2.5*np.log10(2*np.pi)+10*np.log10(1+simdata[simitems == "zl"][0])
    print('MAG',mag,host_reff)
    print('VMAG',-2.5*np.log10(simdata[simitems == "lensgal_amplitude"][0])-48.6)
    host_amp = 22.5-2.5*np.log10(simdata[simitems == "host_amplitude"][0])
    print(str(simitems),'lens_amp',lens_amp,'host_amp',host_amp,simdata[simitems == "lensgal_amplitude"][0])
    '''
    
    magdiff = 2.5*(np.log10(simdata[simitems == "lensgal_amplitude"][0])-np.log10(simdata[simitems == "host_amplitude"][0]))
   
    magdiffps = 2.5*(np.log10(simdata[simitems == "lensgal_amplitude"][0])-np.log10(simdata[simitems == "transient_amplitude"][0]))
  
    sig = simdata[simitems == "sigma"][0]
    theta_E = 4*np.pi*(sig/3e5)**2*Dls/Ds*180/np.pi*60*60
    M0=1.9*(sig/200)**5.1*1e8
    Mstar= 10**((np.log10(M0)-7.45)/1.05)*1e11
 
    radius = einstein_radius(Mstar, Dl*1e6, Ds*1e6)

    kwargs_lens = [
        {'theta_E': theta_E, 'e1': lens_e1, 'e2': lens_e2, 'center_x': lens_x, 'center_y': lens_y},  # SIE model
        {'gamma1': lens_g1, 'gamma2': lens_g2, 'ra_0': 0, 'dec_0': 0}  # SHEAR model
    ]
    # lens light
    kwargs_lens_light_mag_g = [{'magnitude': 23, 'R_sersic': lens_reff, 'n_sersic': lens_n, 'e1': lens_e1, 'e2': lens_e2, 'center_x': lens_x, 'center_y': lens_y}]
    # source light
    kwargs_source_mag_g = [{'magnitude': 23, 'R_sersic': host_reff, 'n_sersic': host_n, 'e1': host_e1, 'e2': host_e2, 'center_x': host_x, 'center_y': host_y}]

    # point source
    kwargs_ps_mag_g = [{'magnitude': 23, 'ra_source': sn_x, 'dec_source': sn_y}]

   
    # and now we define the colors of the other two bands

    # r-band
    g_r_source = 1  # color mag_g - mag_r for source
    g_r_lens = -1  # color mag_g - mag_r for lens light
    g_r_ps = 0
    kwargs_lens_light_mag_r = copy.deepcopy(kwargs_lens_light_mag_g)
    kwargs_lens_light_mag_r[0]['magnitude'] -= g_r_lens

    kwargs_source_mag_r = copy.deepcopy(kwargs_source_mag_g)
    kwargs_source_mag_r[0]['magnitude'] -= g_r_source

    kwargs_ps_mag_r = copy.deepcopy(kwargs_ps_mag_g)
    kwargs_ps_mag_r[0]['magnitude'] -= g_r_ps


    # i-band
    g_i_source = 2
    g_i_lens = -2
    g_i_ps = 0
    kwargs_lens_light_mag_i = copy.deepcopy(kwargs_lens_light_mag_g)
    kwargs_lens_light_mag_i[0]['magnitude'] -= g_i_lens

    kwargs_source_mag_i = copy.deepcopy(kwargs_source_mag_g)
    kwargs_source_mag_i[0]['magnitude'] -= g_i_source

    kwargs_ps_mag_i = copy.deepcopy(kwargs_ps_mag_g)
    kwargs_ps_mag_i[0]['magnitude'] -= g_i_ps



    # here we define the numerical options used in the ImSim module. 
    # Have a look at the ImageNumerics class for detailed descriptions.
    # If not further specified, the default settings are used.
    kwargs_numerics = {'point_source_supersampling_factor': 1}

    # Hack hard coded solution to get out PSF on line 52 of ~/.conda/envs/scarlet/lib/python3.10/site-packages/lenstronomy/Data/psf.py 
    #to get out PSF image!
    size = 6. # width of the image in units of arc seconds

    #img_des, coords_des, sim_b = simulate_rgb(des, size=size, kwargs_numerics=kwargs_numerics)
    
    for indt,time in enumerate(times):
        img_lsst, coords_lss, psf_lsst = simulate_rgb(lsst, size=size, kwargs_numerics=kwargs_numerics,time=time,source_x=sn_x,source_y=sn_y,returnsqrt=False)
        
        img_roman, coords_roman, psf_roman = simulate_rgb(roman, size=size, kwargs_numerics=kwargs_numerics,time=time, source_x=sn_x,source_y=sn_y,returnsqrt=False)
        #print(psf_lsst,psf_roman)
        img_LS4, coords_LS4, psf_LS4 = simulate_rgb_LS4(size=size, kwargs_LS4=kwargs_LS4,time=time, source_x=sn_x,source_y=sn_y,returnsqrt=False) 
        if indt==0:
            vminlsst=np.min(img_lsst)
            vmaxlsst= 1.5*np.max(img_lsst)
            vminLS4=np.min(img_LS4)
            vmaxLS4= 1.5*np.max(img_LS4)
            vminroman=np.min(img_roman)
            vmaxroman= 1.5*np.max(img_roman)
        print(vmaxlsst)


        f, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[1]
        ax.imshow(img_lsst/vmaxlsst,aspect='equal', origin='lower',extent=[0, size, 0, size])
        ax.set_title('LSST')
        scale_bar(ax, d=size, dist=1., text='1"', color='w', font_size=15, flipped=False)

        ax = axes[2]
        ax.imshow(img_roman/vmaxroman, aspect='equal', origin='lower',extent=[0, size, 0, size])
        ax.set_title('Roman')
        scale_bar(ax, d=size, dist=1., text='1"', color='w', font_size=15, flipped=False)
        
        ax = axes[0]
        ax.imshow(img_LS4/vmaxLS4, aspect='equal', origin='lower',extent=[0, size, 0, size])
        ax.set_title('LS4')
        scale_bar(ax, d=size, dist=1., text='1"', color='w', font_size=15, flipped=False)
        if not os.path.exists(plotdir+'/img_'+str(ind)):
            os.mkdir(plotdir+'/img_'+str(ind))
        plt.savefig(plotdir+'/img_'+str(ind)+'/'+str(time)+'.png')
        print('SAVE',plotdir+'/img_'+str(ind)+'/'+str(time)+'.png')
        plt.clf()
       
        img_lsst, coords_lss, psf_lsst = simulate_rgb(lsst, size=size, kwargs_numerics=kwargs_numerics,time=time)
        img_roman, coords_roman, psf_roman = simulate_rgb(roman, size=size, kwargs_numerics=kwargs_numerics,time=time)
     
        pad_LS4 = int((psf_LS4.shape[1]-img_LS4.shape[1])/2)
        pad_lsst = int((psf_lsst.shape[1]-img_lsst.shape[1])/2)
        pad_roman = int((psf_roman.shape[1]-img_roman.shape[1])/2)
        
        image = galsim.ImageF(img_lsst[0])
        affine_wcs = galsim.PixelScale(0.2).affine().withOrigin(image.center)
        ra = 0.0
        dec = 0.0
        wcs_LSST = galsim.TanWCS(affine_wcs,world_origin=galsim.CelestialCoord(ra*galsim.degrees,dec*galsim.degrees))
      
        psf = galsim.Image(psf_lsst[0,pad_lsst:-pad_lsst,pad_lsst:-pad_lsst])
        from scipy.signal import resample as interp
        psf_lsst_data = psf_lsst[0,pad_lsst:-pad_lsst,pad_lsst:-pad_lsst]
        psf_LS4_resample_shape = int(psf_lsst.shape[0]/5)
        psf_LS4_data = interp(psf_lsst,psf_LS4_resample_shape)[0]#,psf_LS4_resample_shape)) 
       
        
        psf.wcs = wcs_LSST
        #psf.write(storedir+'/psf_LSST'+str(ind)+'_newSN.fits')

        image = galsim.ImageF(img_roman[0])
        affine_wcs2 = galsim.PixelScale(0.11).affine().withOrigin(image.center)
        ra = 0.0
        dec = 0.0
        wcs_Roman = galsim.TanWCS(affine_wcs2,world_origin=galsim.CelestialCoord(ra*galsim.degrees,dec*galsim.degrees))

        psf = galsim.Image(psf_roman[0,pad_roman:-pad_roman,pad_roman:-pad_roman])
        image_epsf = galsim.ImageF(psf_roman[0,pad_roman:-pad_roman,pad_roman:-pad_roman].shape[0],psf_roman[0,pad_roman:-pad_roman,pad_roman:-pad_roman].shape[1])
        psf.wcs = wcs_Roman
        #psf.write(storedir+'/'+str(ind)+'psf_Roman_'+str(ind)+'_newSN.fits')
        image = galsim.ImageF(img_LS4[0])
        affine_wcs2 = galsim.PixelScale(1.0).affine().withOrigin(image.center)
        ra = 0.0
        dec = 0.0
        wcs_LS4 = galsim.TanWCS(affine_wcs2,world_origin=galsim.CelestialCoord(ra*galsim.degrees,dec*galsim.degrees))

        psf = galsim.Image(psf_LS4[0,pad_LS4:-pad_LS4,pad_LS4:-pad_LS4])
        image_epsf = galsim.ImageF(psf_LS4[0,pad_LS4:-pad_LS4,pad_LS4:-pad_LS4].shape[0],psf_LS4[0,pad_LS4:-pad_LS4,pad_LS4:-pad_LS4].shape[1])
        psf.wcs = wcs_LS4
        psf.write('/home/cw1074/ZTF/lensing/psf_LS4.fits')

        # In[84]:
        img = galsim.Image(img_LS4[:,:,0])
        img.wcs = wcs_LS4
        img.write(storedir+'/image_LS4_g_'+str(ind)+'_newSN_'+str(time)+'fits')
        x = galsim.Kolmogorov(fwhm=0.7)
        final = galsim.Convolve([x])
        final.drawImage(image=img, wcs=wcs_LS4)
        img.write(storedir+'/image_LS4_g_wcs_'+str(ind)+'_newSN.fits')
        img = galsim.Image(img_LS4[:,:,1])
        img.wcs = wcs_LS4
        img.write(storedir+'/image_LS4_r_'+str(ind)+'_newSN.fits')
        x = galsim.Kolmogorov(fwhm=0.7)
        final = galsim.Convolve([x])
        final.drawImage(image=img, wcs=wcs_LS4)
        img.write(storedir+'/image_LS4_r_wcs_'+str(ind)+'_newSN.fits')
        img = galsim.Image(img_LS4[:,:,2])
        img.wcs = wcs_LS4
        img.write(storedir+'/image_LS4_i_'+str(ind)+'_newSN.fits')
        final = galsim.Convolve([x])
        final.drawImage(image=img, wcs=wcs_LS4)
        img.write(storedir+'/image_LS4_i_wcs_'+str(ind)+'_newSN.fits')


        img = galsim.Image(img_lsst[:,:,0])
        img.wcs = wcs_LSST
        img.write(storedir+'/image_LSST_g_'+str(ind)+'_newSN'+str(time)+'.fits')
        x = galsim.Kolmogorov(fwhm=0.11)
        final = galsim.Convolve([x])
        final.drawImage(image=img, wcs=wcs_LSST)
        img.write(storedir+'/image_LSST_g_wcs_'+str(ind)+'_newSN'+str(time)+'.fits')
        img = galsim.Image(img_lsst[:,:,1])
        img.wcs = wcs_LSST
        img.write(storedir+'/image_LSST_r_'+str(ind)+'_newSN'+str(time)+'.fits')
        x = galsim.Kolmogorov(fwhm=0.11)
        final = galsim.Convolve([x])
        final.drawImage(image=img, wcs=wcs_LSST)
        img.write(storedir+'/image_LSST_r_wcs_'+str(ind)+'_newSN'+str(time)+'.fits')
        img = galsim.Image(img_lsst[:,:,2])
        img.wcs = wcs_LSST
        img.write(storedir+'/image_LSST_i_'+str(ind)+'_newSN'+str(time)+'.fits')
        final = galsim.Convolve([x])
        final.drawImage(image=img, wcs=wcs_LSST)
        img.write(storedir+'/image_LSST_i_wcs_'+str(ind)+'_newSN'+str(time)+'.fits')

        
        img.wcs = wcs_Roman
        img = galsim.Image(img_roman[:,:,0])
        x = galsim.Kolmogorov(fwhm=0.11)
        final = galsim.Convolve([x])
        img.wcs = wcs_Roman
        img.write(storedir+'/image_Roman_g_'+str(ind)+'_newSN'+str(time)+'.fits')
        final.drawImage(image=img, wcs=wcs_Roman)
        img.write(storedir+'/image_Roman_g_wcs_'+str(ind)+'_newSN'+str(time)+'.fits')
        img = galsim.Image(img_roman[:,:,1])
        x = galsim.Kolmogorov(fwhm=0.11)
        final = galsim.Convolve([x])
        img.wcs = wcs_Roman
        img.write(storedir+'/image_Roman_r_'+str(ind)+'_newSN'+str(time)+'.fits')
        final.drawImage(image=img, wcs=wcs_Roman)
        img.write(storedir+'/image_Roman_r_wcs_'+str(ind)+'_newSN'+str(time)+'.fits')
        img = galsim.Image(img_roman[:,:,2])
        x = galsim.Kolmogorov(fwhm=0.11)
        final = galsim.Convolve([x])
        img.wcs = wcs_Roman
        img.write(storedir+'/image_Roman_i_'+str(ind)+'_newSN'+str(time)+'.fits')
        final.drawImage(image=img, wcs=wcs_Roman)
        img.write(storedir+'/image_Roman_i_wcs_'+str(ind)+'_newSN'+str(time)+'.fits')



