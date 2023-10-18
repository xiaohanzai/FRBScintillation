import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from WaterfallLoader import WaterfallLoader, data_reduction_pipeline_chime
from SpectraCalculator import SpectraCalculator
from SpectraFitter import SpectraFitter
from ACFCalculator import ACFCalculator
from ACFFitter import ACFFitter, f_lorenzian

def log_rebin(xs, ys, edge0, n_linear):
    '''
    Bin the data linearly and logarithmically.
    n_linear is the number of linear bins: [0, egde0], [edge0, 2*edge0], [2*edge0, 3*edge0], ...
    The rest of the bins are logarithmic with a factor of (1 + 1/n_linear).
    '''
    # sort the data
    inds = np.argsort(xs)
    xs = xs[inds]
    ys = ys[inds]

    bin_r = edge0
    ind_l = 0
    ind_r = None
    xs_new = []
    ys_new = []
    while bin_r < xs[-1]:
        ind_r = np.searchsorted(xs, bin_r, side='right')
        xs_new.append(np.mean(xs[ind_l:ind_r]))
        ys_new.append(np.mean(ys[ind_l:ind_r]))
        ind_l = ind_r
        if len(xs_new) < n_linear:
            bin_r = edge0 * (len(xs_new) + 1)
        else:
            bin_r = (1. + 1. / n_linear) * bin_r
    return np.array(xs_new), np.array(ys_new)

class FitScintPipeline():
    def __init__(self, data_reduction_pipeline):
        self.wfloader = WaterfallLoader(data_reduction_pipeline)

    def load_ww(self, fname, DM, **kwargs):
        ww = self.wfloader.load_data(fname, DM, **kwargs)
        return ww

    def calc_spec(self, ww, freqs=None, n_noise=3,
                list_time_slcs=[np.s_[:], np.s_[::2], np.s_[1::2]],
                plot=True, save_path='', save_plot=False, return_all=False,
                calc_deripple_arr=False, fftsize=32, downfreq=2, **kwargs):
        ax1 = ax2 = None
        if plot:
            fig, axes = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3)
            ax1 = axes[0]
            ax2 = axes[1]
        # prepare to calculate spectra
        spec_calculator = SpectraCalculator(ww, freqs=freqs)
        # calculate on range
        spec_calculator.calc_on_range(plot=plot, ax=ax2, n_noise=n_noise)
        if np.isnan(spec_calculator.l_on):
            return None, None, None
        # plot waterfall
        if plot:
            spec_calculator.plot_waterfall(ax=ax1)
            if save_plot:
                fig.savefig(save_path + '/matched_filtering.png')
            plt.show()
        # calculate new deripple array if required
        deripple_arr = None
        if calc_deripple_arr:
            deripple_arr = spec_calculator.calc_deripple_arr(fftsize=fftsize, downfreq=downfreq)
        # calculate on-pulse and off-pulse spectra
        spec_on_, freqs_new = spec_calculator.calc_spec_on(list_time_slcs=list_time_slcs, deripple_arr=deripple_arr,
                                                        fftsize=fftsize, downfreq=downfreq)
        spec_offs_, freqs_new = spec_calculator.calc_spec_off(list_time_slcs=list_time_slcs, deripple_arr=deripple_arr,
                                                        fftsize=fftsize, downfreq=downfreq)
        if return_all:
            return spec_calculator, spec_on_, spec_offs_, freqs_new
        return spec_on_, spec_offs_, freqs_new

    def fit_spec(self, spec_on_, spec_offs_, freqs, num_splines=50,
                plot=True, save_path='', save_plot=False, return_all=False, **kwargs):
        # prepare to fit smooth spectra
        spec_fitter = SpectraFitter(spec_on_, spec_offs_, freqs, clean_rfi=True)
        # fit smooth
        spec_smooth_on_, spec_smooth_offs_ = spec_fitter.fit_smooth(num_splines=num_splines, **kwargs)
        # diagnostic plots
        if plot:
            fig, axes = plt.subplots(figsize=(15,5), nrows=1, ncols=3)
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3)
            ind0 = 0
            inds_off = np.random.choice(spec_offs_.shape[1], 3, replace=False)
            # plot spec_on
            axes[0].plot(freqs, spec_on_[ind0], label='on-pulse')
            # plot smooth spectra
            axes[1].plot(freqs, spec_on_[ind0], label='fitted smooth')
            for i in range(spec_smooth_on_.shape[0]):
                linestyle = '-'
                if i != ind0:
                    linestyle = '--'
                axes[1].plot(freqs, spec_smooth_on_[i], linestyle=linestyle, label='time slice %d' % i)
            axes[1].text(axes[1].get_xlim()[1], axes[1].get_ylim()[1], 'on-pulse', ha='right', va='top')
            axes[1].legend(loc=2)
            # plot spec_offs
            for ind in inds_off:
                axes[0].plot(freqs, spec_offs_[ind0, ind], alpha=0.3, label='off-pulse')
            #     axes[2].plot(freqs, spec_offs_[ind0, ind], alpha=0.3)
            #     axes[2].plot(freqs, spec_smooth_offs_[ind0, ind])
            # axes[2].text(axes[2].get_xlim()[1], axes[2].get_ylim()[1], 'off-pulse', ha='right', va='top')
            # plot dspec_on and dspec_off
            mean_spec_off = np.nanmean(spec_offs_[ind0], axis=0)
            spec_smooth_off = np.nanmean(spec_smooth_offs_[ind0], axis=0)
            spec_smooth = spec_smooth_on_[ind0] - spec_smooth_off
            dspec_on = (spec_on_[ind0] - mean_spec_off) / spec_smooth - 1
            axes[2].plot(freqs, dspec_on, label='on-pulse')
            for ind in inds_off:
                dspec_off = (spec_offs_[ind0, ind] - mean_spec_off) / spec_smooth
                axes[2].plot(freqs, dspec_off, alpha=0.2, label='off-pulse')
            axes[2].set_ylabel('spec / spec_smooth - 1')
            axes[2].set_ylim(-8, 8)
            # labels and legend
            for ax in axes:
                ax.set_xlabel('freq (MHz)')
            axes[0].set_ylabel('spectra')
            axes[0].legend()
            # save plot
            if save_plot:
                fig.savefig(save_path + '/fit_spec.png')

        if return_all:
            return spec_fitter, spec_smooth_on_, spec_smooth_offs_
        return spec_smooth_on_, spec_smooth_offs_

    def calc_acf(self, time_slc_i, time_slc_j, freq_min, freq_max,
                dspec_on_=None, dspec_offs_=None, freqs=None, acf_calculator=None, return_all=False):
        if acf_calculator is None:
            acf_calculator = ACFCalculator(dspec_on_, dspec_offs_, freqs)
        acf_on, acf_offs, nus = acf_calculator.calc_acf_band(time_slc_i, time_slc_j, freq_min, freq_max)
        if return_all:
            return acf_calculator, acf_on, acf_offs, nus
        return acf_on, acf_offs, nus

    def fit_acf(self, acf_on, acf_offs, nus, ebar_scale_fac=1., exclude_zero=False):
        acf_fitter = ACFFitter(acf_on, acf_offs, nus, ebar_scale_fac=ebar_scale_fac)
        m, m_err, nu_dc, nu_dc_err = acf_fitter.fit_acf(exclude_zero=exclude_zero)
        return m, m_err, nu_dc, nu_dc_err

    def plot_acf(self, nus, acf_on, m, nu_dc, y_offset=0., label=None, acf_offs=None, axes=None):
        if axes is None:
            fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3)
        if acf_offs is not None:
            # randomly choose 3 off-pulse ACFs to plot
            inds_off = np.random.choice(acf_offs.shape[0], 3, replace=False)
            cs = ['b', 'g', 'm']
        # log binning
        xs, ys = log_rebin(nus, acf_on, 0.02, 1)
        # plot
        for ax in axes:
            ax.plot(nus, 0*nus+y_offset, 'k--')
            # plot on-pulse ACF and fitted result
            l = ax.plot(nus, acf_on+y_offset, alpha=0.5)[0]
            c = l.get_color()
            ax.plot(xs, ys+y_offset, 'o-', color=c, label=label)
            ax.plot(nus, f_lorenzian(nus, m, nu_dc)+y_offset, 'r')
            # plot off-pulse ACFs
            if acf_offs is not None:
                for c, ind in zip(cs, inds_off):
                    ax.plot(nus, acf_offs[ind]+y_offset, c, alpha=0.3)
            ax.set_xlabel(r'$\Delta\nu$ (MHz)')
            ax.set_ylabel('ACF')
            if np.isnan(acf_on[0]):
                ax.set_ylim(y_offset-0.02, y_offset+1)
            else:
                ax.set_ylim(y_offset-0.02, y_offset+acf_on[0]+0.02)
        axes[0].set_xlim(1e-6, 2)
        axes[1].set_xscale('log')
        axes[1].set_xlim(None, 2)
        axes[0].legend(loc=1)
        return axes

    def calc_and_fit_acf(self, dspec_on_, dspec_offs_, freqs, time_slc_i, time_slc_j, ebar_scale_fac=1.,
                freq_bins=[[400,500],[500,600]], plot=True, save_path='', save_plot=False,
                exclude_zero=False, **kwargs):
        # prepare to calculate ACFs
        acf_calculator = ACFCalculator(dspec_on_, dspec_offs_, freqs)
        # and plot results
        if plot:
            fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3)
            ylim = [-0.5, len(freq_bins)+0.5]
        # for each frequency bin
        N = len(freq_bins)
        ms, m_errs, nu_dcs, nu_dc_errs = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        acf_on_ = [None]*N
        acf_offs_ = [None]*N
        for i in range(N):
            freq_bin = freq_bins[i]
            acf_on, acf_offs, nus = self.calc_acf(time_slc_i, time_slc_j, freq_bin[0], freq_bin[1], acf_calculator=acf_calculator)
            # fit ACF
            ms[i], m_errs[i], nu_dcs[i], nu_dc_errs[i] = self.fit_acf(acf_on, acf_offs, nus, ebar_scale_fac=ebar_scale_fac,
                                                                    exclude_zero=exclude_zero)
            # plot
            if plot:
                self.plot_acf(nus, acf_on, ms[i], nu_dcs[i], y_offset=i*1., label=f'{freq_bin[0]}-{freq_bin[1]} MHz',
                            acf_offs=acf_offs, axes=axes)
                # if i == 0:
                #     ylim[0] = axes[0].get_ylim()[0]
                # if i == N-1:
                #     ylim[1] = axes[0].get_ylim()[1]
            # for dumping
            ii = nus < 10 # MHz
            nus = nus[ii]
            acf_on_[i] = acf_on[ii]
            acf_offs_[i] = acf_offs[:,ii]
        # change xlim ylim
        if plot:
            for ax in axes:
                ax.set_ylim(ylim)
        # save plot
        if plot and save_plot:
            fig.savefig(save_path + '/fit_acf.png')

        return nus, acf_on_, acf_offs_, ms, m_errs, nu_dcs, nu_dc_errs

    def run(self, fname, DM, out_path, freqs_orig=None, list_time_slcs=[np.s_[:], np.s_[::2], np.s_[1::2]],
            time_slc_i=1, time_slc_j=2, freq_bins=np.stack((np.arange(400,790,50),np.arange(450,810,50))).T, dump=True,
            save_plot=True, **kwargs):
        # create dir to save all diagnostic plots and data
        save_path = out_path + '/' + fname[fname.rindex('/'):-3]
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # load waterfall data
        ww = self.load_ww(fname, DM, save_path=save_path, save_plot=save_plot, **kwargs)
        if ww is None:
            return

        # calculate spectra
        spec_on_, spec_offs_, freqs = self.calc_spec(ww, freqs=freqs_orig, list_time_slcs=list_time_slcs,
                                                    plot=True, save_path=save_path, save_plot=save_plot, **kwargs)
        if spec_on_ is None:
            return

        # fit smooth spectra
        spec_smooth_on_, spec_smooth_offs_ = self.fit_spec(spec_on_, spec_offs_, freqs,
                                                    plot=True, save_path=save_path, save_plot=save_plot, **kwargs)

        # calculate and fit ACFs
        ebar_scale_fac = 1.#np.nanmean(spec_smooth_on_[0])/np.nanmean(spec_smooth_offs_[0]) # TODO: errorbar calculation?
        mean_spec_off_ = np.nanmean(spec_offs_, axis=1)
        spec_smooth_off_ = np.nanmean(spec_smooth_offs_, axis=1)
        spec_smooth_ = spec_smooth_on_ - spec_smooth_off_
        dspec_on_ = (spec_on_-mean_spec_off_)/spec_smooth_ - 1
        dspec_offs_ = (spec_offs_-mean_spec_off_[:,np.newaxis,:])/spec_smooth_[:,np.newaxis,:]
        nus, acf_on_, acf_offs_, ms, m_errs, nu_dcs, nu_dc_errs = self.calc_and_fit_acf(
            dspec_on_, dspec_offs_, freqs, time_slc_i, time_slc_j,
            ebar_scale_fac=ebar_scale_fac, freq_bins=freq_bins,
            plot=True, save_path=save_path, save_plot=save_plot, **kwargs)
        nus, acf_on0_, acf_offs0_, ms0, m_errs0, nu_dcs0, nu_dc_errs0 = self.calc_and_fit_acf(
            dspec_on_, dspec_offs_, freqs, 0, 0,
            ebar_scale_fac=ebar_scale_fac, freq_bins=freq_bins,
            plot=True, save_path=save_path, save_plot=False, exclude_zero=False, **kwargs)
        print('mod indices from cross corr vs autocorr:', ms, ms0)

        # save results
        if dump:
            rst_specs = {
                'spec_on_': spec_on_,
                'spec_offs_': spec_offs_,
                'freqs': freqs,
            }
            with open(save_path + '/specs.pkl', 'wb') as f:
                pickle.dump(rst_specs, f)
            rst_acfs = {
                'freq_bins': freq_bins,
                'nus': nus,
                'acf_on_': acf_on_,
                'acf_offs_': acf_offs_,
                'ms': ms,
                'm_errs': m_errs,
                'nu_dcs': nu_dcs,
                'nu_dc_errs': nu_dc_errs,
                'acf_on0_': acf_on0_,
                'acf_offs0_': acf_offs0_,
                'ms0': ms0,
                'm_errs0': m_errs0,
                'nu_dcs0': nu_dcs0,
                'nu_dc_errs0': nu_dc_errs0,
            }
            with open(save_path + '/acfs.pkl', 'wb') as f:
                pickle.dump(rst_acfs, f)

def main():
    path = '/arc/projects/chime_frb/baseband_catalog/beamformed_files/'
    fnames = glob.glob(path + 'singlebeam*.h5')

    out_path = '/arc/projects/chime_frb/xiaohan/scintillation'
    fit_scint_pipeline = FitScintPipeline(data_reduction_pipeline_chime)
    for fname in fnames:
        fit_scint_pipeline.run(fname, out_path)

if __name__ == '__main__':
    main()
