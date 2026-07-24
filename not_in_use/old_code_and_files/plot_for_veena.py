import numpy as np
import matplotlib.pyplot as plt
import sncosmo

from gaussn import gausSN, kernels, meanfuncs, lensingmodels, utils

def plot_fitted_object(data, factor, results, kernel, meanfunc, lensingmodel, fix_kernel_params=False, fix_mean_params=False, fix_lensing_params=False,
                       predict_times = np.linspace(-20, 60, 50), N_iter=10, n_images=None,
                       color_dict_data = {'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'tab:orange', 'unresolved': 'k'},
                       color_dict_fit = {'image_1': 'tab:blue', 'image_2': 'palevioletred', 'image_3': 'tab:green', 'image_4': 'darkorange', 'unresolved': 'dimgray'},
                       marker_dict={'image_1': 's', 'image_2': 'D', 'image_3': '>', 'image_4': '<', 'unresolved': 'o'}, title=''):
    """
    Plots the fitted glSN with uncertainties, assuming the GausSN2 model.

    Args:
        data (Table): Data containing flux measurements.
        results (obj): Results of the fitting.
        kernel (obj): Kernel object.
        meanfunc (obj): Mean function object.
        lensingmodel (obj): Lensing model object.
        fix_kernel_params (bool): If True, fixes kernel parameters.
        fix_mean_params (bool): If True, fixes mean function parameters.
        fix_lensing_params (bool): If True, fixes lensing model parameters.
        color_dict_data (dict): Dictionary mapping image number (i.e. 'image_1') to colors for plotting data.
        color_dict_fit (dict): Dictionary mapping image number (i.e. 'image_1') to colors for plotting fitted light curves.
        title (str): Title of the plot.

    Returns:
        tuple: The figure and axis of the plot.
    """

    # Array specifying the order of bands
    unresolved_bands = utils.ordered[np.isin(utils.ordered, np.unique(data[data['image'] == 'unresolved']['band']))]
    resolved_bands = utils.ordered[np.isin(utils.ordered, np.unique(data[data['image'] != 'unresolved']['band']))]
    image_num_labels = np.unique(data[data['image'] != 'unresolved']['image'])
    if n_images is None:
        n_images = len(image_num_labels)

    # Create subplots based on the number of unique bands
    fig, ax = plt.subplots(2*len(np.unique(data['band'])), n_images, figsize=(8, 1.5 * ( 2 * (len(unresolved_bands) + len(resolved_bands)) ) ),
                           sharex=True, sharey='row', gridspec_kw={'height_ratios': np.tile([3, 2], len(np.unique(data['band'])))})

    # Plot flux measurements for each band and image
    for b, pb_id in enumerate(list(unresolved_bands)):
        band = data[data['band'] == pb_id]

        try:
            color_dict_data_temp = color_dict_data[pb_id]
        except:
            color_dict_data_temp = color_dict_data

        try:
            marker_dict_temp = marker_dict[pb_id]
        except:
            marker_dict_temp = marker_dict

        try:
            color = color_dict_data_temp['unresolved']
        except:
            color = color_dict_data_temp

        try:
            marker = marker_dict_temp['unresolved']
        except:
            marker = marker_dict_temp

        image_label = 'Unresolved'
        ax[b*2,0].errorbar(band['time'], band['mag'], yerr=band['magerr'], ls='None', marker=marker,
                           color=color, label=image_label, zorder=1)
        ax[b*2,0].invert_yaxis()
        ax[(b*2)+1,0].invert_yaxis()

    for b, pb_id in enumerate(list(resolved_bands)):
        band = data[data['band'] == pb_id]

        try:
            color_dict_data_temp = color_dict_data[pb_id]
        except:
            color_dict_data_temp = color_dict_data

        try:
            marker_dict_temp = marker_dict[pb_id]
        except:
            marker_dict_temp = marker_dict

        for m,im_id in enumerate(np.unique(data[data['image'] != 'unresolved']['image'])):
            image = band[band['image'] == im_id]

            try:
                color = color_dict_data_temp[im_id]
            except:
                color = color_dict_data_temp

            try:
                marker = marker_dict_temp[im_id]
            except:
                marker = marker_dict_temp

            image_label = 'Image '+im_id[-1]
            ax[(b+len(unresolved_bands))*2,m].errorbar(image['time'], image['mag'], yerr=image['magerr'], ls='None', marker=marker,
                                                       color=color, label=image_label, zorder=1)
        
        ax[(b+len(unresolved_bands))*2,0].set_ylabel(pb_id, fontsize=16)
        ax[((b+len(unresolved_bands))*2)+1,0].set_ylabel('$\\beta(t)$', fontsize=16)

        ax[(b+len(unresolved_bands))*2,0].invert_yaxis()
        ax[((b+len(unresolved_bands))*2)+1,0].invert_yaxis()

    # Get equal-weighted samples from the results
    samples = results.samples_equal()

    # Iterate over random samples from the posterior
    for iter in np.random.choice(len(samples), N_iter):
        sample = samples[iter]

        # Reset parameters based on whether they are fixed
        if not fix_lensing_params and not fix_mean_params and not fix_kernel_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            meanfunc_params = [sample[i+len(kernel.params)] for i in range(len(meanfunc.params))]
            lensing_params = [sample[i+len(kernel.params)+len(meanfunc.params)] for i in range(len(lensingmodel.params))]
            kernel._reset(kernel_params)
            #meanfunc._reset(meanfunc_params)
            lensingmodel._reset(lensing_params)
        elif not fix_mean_params and not fix_kernel_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            meanfunc_params = [sample[i+len(kernel.params)] for i in range(len(meanfunc.params))]
            kernel._reset(kernel_params)
            #meanfunc._reset(meanfunc_params)
        elif not fix_mean_params and not fix_lensing_params:
            meanfunc_params = [sample[i] for i in range(len(meanfunc.params))]
            lensing_params = [sample[i+len(meanfunc.params)] for i in range(len(lensingmodel.params))]
            #meanfunc._reset(meanfunc_params)
            lensingmodel._reset(lensing_params)
        elif not fix_kernel_params and not fix_lensing_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            lensing_params = [sample[i+len(kernel.params)] for i in range(len(lensingmodel.params))]
            kernel._reset(kernel_params)
            lensingmodel._reset(lensing_params)
        elif not fix_kernel_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            kernel._reset(kernel_params)
        elif not fix_mean_params:
            meanfunc_params = [sample[i] for i in range(len(meanfunc.params))]
            #meanfunc._reset(meanfunc_params)
        elif not fix_lensing_params:
            lensing_params = [sample[i] for i in range(len(lensingmodel.params))]
            lensingmodel._reset(lensing_params)
        
        # Create GP object
        gp = gausSN.GP(kernel, meanfunc, lensingmodel)

        for b, pb_id in enumerate(unresolved_bands):
            band = data[data['band'] == pb_id]
            eff_wave = [sncosmo.get_bandpass(b).wave_eff for b in band['band']]

            if len(band) < 1:
                continue

            try:
                color_dict_fit_temp = color_dict_fit[pb_id]
            except:
                color_dict_fit_temp = color_dict_fit

            repeated_times = np.tile(band['time'], n_images)
            repeated_deltas = np.repeat(gp.lensingmodel.deltas, len(band))
            repeated_betas = np.repeat(gp.lensingmodel.betas, len(band))
            for m in range(n_images):
                if m == 0:
                    T = np.diag(repeated_betas[m * len(band) : (m+1) * len(band)])
                else:
                    T = np.hstack([T, np.diag(repeated_betas[m * len(band) : (m+1) * len(band)])])

            x = np.subtract(repeated_times, repeated_deltas)
            template = gp.meanfunc.mean(x, bands=np.repeat(band['band'], n_images), images=np.repeat(image_num_labels, len(band)),
                                        zp=np.repeat(band['zp'], n_images), zpsys=np.repeat(band['zpsys'], n_images), params=meanfunc_params)
            y = band['flux'] / np.matmul(T, template)
            yerr = band['fluxerr'] / np.matmul(T, template)

            repeated_predict_times = np.tile(predict_times, n_images)
            repeated_predict_deltas = np.repeat(gp.lensingmodel.deltas, len(predict_times))
            repeated_predict_betas = np.repeat(gp.lensingmodel.betas, len(predict_times))
            for m in range(n_images):
                if m == 0:
                    predict_T = np.diag(repeated_predict_betas[m * len(predict_times) : (m+1) * len(predict_times)])
                else:
                    predict_T = np.hstack([predict_T, np.diag(repeated_predict_betas[m * len(predict_times) : (m+1) * len(predict_times)])])

            predict_x = np.subtract(repeated_predict_times, repeated_predict_deltas)
            template_predict = np.matmul(predict_T, gp.meanfunc.mean(predict_x, bands=np.repeat(pb_id, len(predict_x)),
                                                                     images=np.repeat(image_num_labels, len(predict_times)),
                                                                     zp=np.repeat(band['zp'][0], len(predict_x)),
                                                                     zpsys=np.repeat(band['zpsys'][0], len(predict_x)), params=meanfunc_params))

            predict_wave = np.repeat(sncosmo.get_bandpass(pb_id).wave_eff, len(predict_times))

            mu_U = np.repeat(1., len(predict_times))
            mu_V = np.repeat(1., len(band))

            cov_UU = gp.kernel.covariance(np.vstack([predict_times, predict_wave]), params=kernel_params)
            cov_UV = gp.kernel.covariance(np.vstack([predict_times, predict_wave]), x_prime=np.vstack([band['time'], eff_wave]), params=kernel_params)
            cov_VV = gp.kernel.covariance(np.vstack([band['time'], eff_wave]), params=kernel_params) + np.diagflat(yerr**2)

            exp = mu_U + (cov_UV @ np.linalg.solve(cov_VV, y-mu_V))
            cov = cov_UU - (cov_UV @ np.linalg.solve(cov_VV, np.transpose(cov_UV)))

            for i in range(1):
                beta_realization = np.random.multivariate_normal(mean=exp, cov=cov, size=1)
                    
                try:
                    color = color_dict_fit_temp['unresolved']
                except:
                    color = color_dict_fit_temp
                
                fluxes = beta_realization[0]*factor*template_predict
                mags = band['zp'][0] - 2.5 * np.log10( fluxes )
                ax[b*2,0].plot(predict_times, mags, color=color, alpha=0.2, zorder=2)
                ax[(b*2)+1,0].plot(predict_times, -2.5*np.log10(beta_realization[0]), color=color, alpha=0.2)


        for m, im_id in enumerate(image_num_labels):
            image = data[data['image'] == im_id]
            ax[0,m].set_title(f'Image {m+1}', fontsize=16)
            ax[-1,m].set_xlabel('Time [days]', fontsize=16)

            if len(image) < 1:
                continue

            x = np.subtract(image['time'], gp.lensingmodel.deltas[m])
            mask = np.logical_and(x > gp.meanfunc.model.mintime(), x < gp.meanfunc.model.maxtime())
            y = np.array([sncosmo.get_bandpass(b).wave_eff for b in image['band']])
            template = gp.meanfunc.mean(x[mask], bands=image['band'][mask], images=image['image'][mask], zp=image['zp'][mask], zpsys=image['zpsys'][mask], params=meanfunc_params)
            z = image['flux'][mask] / np.multiply(gp.lensingmodel.betas[m], template)
            zerr = image['fluxerr'][mask] / np.multiply(gp.lensingmodel.betas[m], template)

            for b, pb_id in enumerate(resolved_bands):
                band = image[image['band'] == pb_id]

                if len(band) < 1:
                    continue

                predict_x = np.subtract(predict_times, gp.lensingmodel.deltas[m])
                predict_y = np.repeat(sncosmo.get_bandpass(pb_id).wave_eff, len(predict_x))

                template_predict = np.multiply(gp.lensingmodel.betas[m], gp.meanfunc.mean(predict_x, bands=np.repeat(pb_id, len(predict_x)),
                                                                                          images=np.repeat(im_id, len(predict_x)),
                                                                                          zp=np.repeat(band['zp'][0], len(predict_x)),
                                                                                          zpsys=np.repeat(band['zpsys'][0], len(predict_x)),
                                                                                          params=meanfunc_params))

                mu_U = np.repeat(1., len(predict_x))
                mu_V = np.repeat(1., len(image))

                cov_UU = gp.kernel.covariance(np.vstack([predict_x, predict_y]), params=kernel_params)
                cov_UV = gp.kernel.covariance(np.vstack([predict_x, predict_y]), x_prime=np.vstack([x[mask], y[mask]]), params=kernel_params)
                cov_VV = gp.kernel.covariance(np.vstack([x[mask], y[mask]]), params=kernel_params) + np.diagflat(zerr**2)

                exp = mu_U + (cov_UV @ np.linalg.solve(cov_VV, z-mu_V))
                cov = cov_UU - (cov_UV @ np.linalg.solve(cov_VV, np.transpose(cov_UV)))

                for i in range(1):
                    beta_realization = np.random.multivariate_normal(mean=exp, cov=cov, size=1)
                    
                    try:
                        color = color_dict_fit[im_id][pb_id]
                    except:
                        try:
                            color = color_dict_fit[pb_id]
                        except:
                            try:
                                color = color_dict_fit[im_id]
                            except:
                                color = color_dict_fit

                    fluxes = beta_realization[0]*factor*template_predict
                    mags = band['zp'][0]-2.5*np.log10(fluxes)
                    ax[(b+len(unresolved_bands))*2,m].plot(predict_times, mags, color=color, alpha=0.2, zorder=2)
                    ax[((b+len(unresolved_bands))*2)+1,m].plot(predict_times, -2.5*np.log10(beta_realization[0]), color=color, alpha=0.2)

    # Set title, ylabel for the flux and adjust subplot spacing
    fig.supylabel('Flux', fontsize=20)
    fig.suptitle(title, fontsize=24)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax