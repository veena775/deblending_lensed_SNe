import numpy as np
import h5py
from astropy.cosmology import WMAP9 as cosmo
import galsim
import matplotlib.pyplot as plt
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
# # make sure lenstronomy is installed, otherwise install the latest pip version
# try:
#     import lenstronomy
# except:
#     get_ipython().system('pip install lenstronomy')
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
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
from lenstronomy.Cosmo.micro_lensing import einstein_radius


# we define a time variable function in magnitude space
def var_func(time, band): #band either desg, desi, desr
    obs = Table({'time': [time],
             'band': [band],            #filters we are observing in 
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
    return magnitude*0.5

var_func_g = lambda t: var_func(t, 'desg')
var_func_r = lambda t: var_func(t, 'desr')
var_func_i = lambda t: var_func(t, 'desi')



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
    # print('DELAYS',delays,np.max(delays))
    max_delays = np.max(delays)
    n_delays = len(delays)

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
    # #print(sim_r.psf_class.fwhm)
    # psf_i = sim_i.psf_class.kernel_point_source
    # psf_g = sim_g.psf_class.kernel_point_source
    # psf_r = sim_r.psf_class.kernel_point_source
    # psf = np.stack((psf_i,psf_g,psf_r))
    return img, data_class, max_delays







f = h5py.File('lsst-altsched-1a-lowz.h5','r')
simitems = np.asarray(f['system']['block0_items'][()],dtype=str)
simdataall = f['system']['block0_values'][()]
# print(simdataall[0],simitems)
rtest_source = [0.944,0.766, 0.678, 0.967, 0.717]
rtest_lens = [0.410, 0.396, 0.344, 0.680, 0.056]
rtest_delays = [11.36, 9.07, 20.28, 1.82, 3.06]

#This is where I match to that table I provided by redshift. You'll want to replace this with a search for the time delay parameter and match by that!
indcatalog =[]
delay_catalog = []
for i in range(len(rtest_source)):
    indD = np.argwhere((np.abs(rtest_lens[i]-simdataall[:,7])<0.005)&(np.abs(rtest_source[i]-simdataall[:,6])<0.005))
    if i==0:
        indcatalog=indD[:,0]
        
    else:
        indcatalog = np.hstack((indcatalog,indD[:,0]))
    delay_catalog.extend([rtest_delays[i]] * len(indD))


for i, ind in enumerate(indcatalog):
    simdata=simdataall[ind]
    z_lens = simdata[simitems == "zl"][0]
    z_source = simdata[simitems == "zs"][0]
    # print('z les',z_lens,'z source',z_source)
    
    target_delay = delay_catalog[i]
    # print('Target max time delay: ', target_delay)
    
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
    kwargs_ps_mag_g = [{'magnitude': 23, 'ra_source': sn_x, 'dec_source': sn_y}]    # and now we define the colors of the other two bands

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


    for indt,time in enumerate(times):
        img_lsst, coords_lss, max_delay = simulate_rgb(lsst, size=size, kwargs_numerics=kwargs_numerics,time=time,source_x=sn_x,source_y=sn_y,returnsqrt=False)

        if (np.abs(max_delay - target_delay) < 1):
            print('Index: ', ind)
            print('Target delay: ', target_delay)
            print('Max delay: ', max_delay)
            print('z lens',z_lens,'z source',z_source)
            print(time)




