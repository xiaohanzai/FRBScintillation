import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import os
import glob
from WaterfallLoader import WaterfallLoader, data_reduction_pipeline_chime
from SpectraCalculator import SpectraCalculator
from SpectraFitter import SpectraFitter
from ACFCalculator import ACFCalculator, calc_dspec
from ACFFitter import ACFFitter
from plot_functions import *
try:
    from fitburst_functions import get_fb_model, create_full_fb_filter
except:
    print('Warning: not importing fitburst')

class FitScintPipeline():
    def __init__(self, data_reduction_pipeline):
        self.wfloader = WaterfallLoader(data_reduction_pipeline)

    def load_ww(self, fname, DM, **kwargs):
        ww, offpulse_range = self.wfloader.load_data(fname, DM, **kwargs)
        return ww, offpulse_range

    def load_fitburst_model(self, ww=None, fname='', ds_factor=16, **kwargs):
        if fname == '':
            return None
        model, _ = get_fb_model(fname, ds_factor=ds_factor)
        model_full = create_full_fb_filter(ww, model, ds_factor=ds_factor)
        return model_full

    def calc_spec(self, ww, offpulse_range, freqs=None, spec_calculator=None,
                list_time_slcs=[np.s_[:], np.s_[::2], np.s_[1::2]],
                plot=True, save_path='', save_plot=False, return_all=False,
                calc_deripple_arr=False, fftsize=32, downfreq=2, do_upchannel=True,
                interactive=False, fitburst_model=None, **kwargs):
        if spec_calculator is None:
            # prepare to calculate spectra
            spec_calculator = SpectraCalculator(ww, offpulse_range, freqs=freqs, fitburst_model=fitburst_model)
            # calculate on-range
            if fitburst_model is None:
                spec_calculator.calc_on_range(interactive=interactive)
            # plot waterfall and on-range
            if plot:
                fig, axes = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
                fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3)
                plot_on_range(spec_calculator.power, spec_calculator.filter, spec_calculator.on_range, axes[1])
                plot_waterfall(spec_calculator.power, spec_calculator.on_range, ax=axes[0])
                if save_plot:
                    fig.savefig(save_path + '/matched_filtering.png')
                plt.show()
        # make sure fftsize is not too large
        if do_upchannel:
            while fftsize > spec_calculator.on_range[1] - spec_calculator.on_range[0] and fftsize > 0:
                fftsize = fftsize // 2
            if fftsize < 32:
                print('warning: pulse width too small; trying fftsize=32')
                fftsize = 32
            else:
                print(f'using actual fftsize {fftsize}')
        # calculate new deripple array if required
        deripple_arr = None
        if calc_deripple_arr:
            deripple_arr = spec_calculator.calc_deripple_arr(fftsize=fftsize, downfreq=downfreq)
        # calculate on-pulse and off-pulse spectra
        spec_on_, freqs_new = spec_calculator.calc_spec_on(list_time_slcs=list_time_slcs, deripple_arr=deripple_arr,
                                                        fftsize=fftsize, downfreq=downfreq, **kwargs)
        spec_offs_, freqs_new = spec_calculator.calc_spec_off(list_time_slcs=list_time_slcs, deripple_arr=deripple_arr,
                                                        fftsize=fftsize, downfreq=downfreq, **kwargs)
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
            fig = plot_specs(spec_on_, spec_offs_, spec_smooth_on_, spec_smooth_offs_, freqs)
            # save plot
            if save_plot:
                fig.savefig(save_path + '/fit_spec_upchan{}.png'.format(spec_smooth_on_.shape[-1]//1024))
            plt.show()

        if return_all:
            return spec_fitter, spec_smooth_on_, spec_smooth_offs_
        return spec_smooth_on_, spec_smooth_offs_

    def calc_acf(self, time_slc_i, time_slc_j, freq_min, freq_max,
                dspec_on_=None, dspec_offs_=None, freqs=None, acf_calculator=None, return_all=False):
        if acf_calculator is None:
            acf_calculator = ACFCalculator(dspec_on_, dspec_offs_, freqs)
        acf_on, acf_offs, nus, flag = acf_calculator.calc_acf_band(time_slc_i, time_slc_j, freq_min, freq_max)
        if return_all:
            return acf_calculator, acf_on, acf_offs, nus, flag
        return acf_on, acf_offs, nus, flag

    def fit_acf(self, acf_on, acf_offs, nus, ebar_scale_fac=1., exclude_zero=False):
        acf_fitter = ACFFitter(acf_on, acf_offs, nus, ebar_scale_fac=ebar_scale_fac)
        m, m_err, nu_dc, nu_dc_err = acf_fitter.fit_acf(exclude_zero=exclude_zero)
        return m, m_err, nu_dc, nu_dc_err

    def calc_and_fit_acf(self, dspec_on_, dspec_offs_, freqs, time_slc_i, time_slc_j, ebar_scale_fac=1.,
                freq_bins=[[400,500],[500,600]], plot=True, save_path='', save_plot=False,
                exclude_zero=False, **kwargs):
        # prepare to calculate ACFs
        acf_calculator = ACFCalculator(dspec_on_, dspec_offs_, freqs)
        # for each frequency bin
        N = len(freq_bins)
        ms, m_errs, nu_dcs, nu_dc_errs = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        flags = np.zeros(N, dtype=bool)
        acf_on_ = [None]*N
        acf_offs_ = [None]*N
        for i in range(N):
            freq_bin = freq_bins[i]
            acf_on, acf_offs, nus, flags[i] = self.calc_acf(time_slc_i, time_slc_j, freq_bin[0], freq_bin[1], acf_calculator=acf_calculator)
            # fit ACF
            ms[i], m_errs[i], nu_dcs[i], nu_dc_errs[i] = self.fit_acf(acf_on, acf_offs, nus, ebar_scale_fac=ebar_scale_fac,
                                                                    exclude_zero=exclude_zero)
            # for dumping
            ii = nus < 10 # MHz
            nus = nus[ii]
            acf_on_[i] = acf_on[ii]
            acf_offs_[i] = acf_offs[:,ii]

        if plot:
            fig = plot_acfs_two_subplots(nus, acf_on_, freq_bins, acf_offs_=acf_offs_, ms=ms, nu_dcs=nu_dcs, flags=flags)
            # save plot
            if save_plot:
                fig.savefig(save_path + '/fit_acf_upchan{}.png'.format(dspec_on_.shape[-1]//1024))
            plt.show()

        return nus, acf_on_, acf_offs_, ms, m_errs, nu_dcs, nu_dc_errs, flags

    def run(self, frb_id, DM, out_path, freqs_orig=None, list_time_slcs=[np.s_[:], np.s_[::2], np.s_[1::2]],
            time_slc_i=1, time_slc_j=2, freq_bins=np.stack((np.arange(400,790,50),np.arange(450,810,50))).T, dump=True,
            save_plot=True, fitburst_fname='', fb_ds_factor=16, read_spec_calcultor_from_disk=False, **kwargs):
        # get bbdata file name
        fname = get_bbdata_fname(frb_id)
        if fname == '':
            print('check frb id again:', frb_id)
            return

        # create dir to save all diagnostic plots and data
        save_path = out_path + '/' + frb_id
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not read_spec_calcultor_from_disk:
            # load waterfall data
            ww, offpulse_range = self.load_ww(fname, DM, save_path=save_path, save_plot=save_plot, **kwargs)
            if ww is None:
                return

            # load fitburst model if use it
            fitburst_model = None
            if fitburst_fname != '':
                fitburst_model = self.load_fitburst_model(ww=ww, fname=fitburst_fname, ds_factor=fb_ds_factor, **kwargs)

            # calculate spectra
            spec_calcultor, spec_on_, spec_offs_, freqs = self.calc_spec(ww, offpulse_range, freqs=freqs_orig, list_time_slcs=list_time_slcs,
                plot=True, save_path=save_path, save_plot=save_plot, fitburst_model=fitburst_model, return_all=True, **kwargs)
            if spec_on_ is None:
                return
            # save spec_calcultor
            with open(save_path + '/spec_calcultor.pkl', 'wb') as f:
                pickle.dump(spec_calcultor, f)
        else:
            print('reading previously calculated spec_calcultor from disk')
            # load spec_calcultor
            with open(save_path + '/spec_calcultor.pkl', 'rb') as f:
                spec_calcultor = pickle.load(f)
            # load fitburst model if use it
            fitburst_model = None
            if fitburst_fname != '':
                fitburst_model = self.load_fitburst_model(ww=spec_calcultor.ww, fname=fitburst_fname, ds_factor=fb_ds_factor, **kwargs)
            spec_on_, spec_offs_, freqs = self.calc_spec(None, None, spec_calculator=spec_calcultor, freqs=freqs_orig, list_time_slcs=list_time_slcs,
                plot=True, save_path=save_path, save_plot=save_plot, fitburst_model=fitburst_model, **kwargs)

        # fit smooth spectra
        spec_smooth_on_, spec_smooth_offs_ = self.fit_spec(spec_on_, spec_offs_, freqs,
            plot=True, save_path=save_path, save_plot=save_plot, **kwargs)

        # calculate and fit ACFs
        ebar_scale_fac = 1.#np.nanmean(spec_smooth_on_[0])/np.nanmean(spec_smooth_offs_[0]) # TODO: errorbar calculation?
        dspec_on_, dspec_offs_ = calc_dspec(spec_on_, spec_offs_, spec_smooth_on_, spec_smooth_offs_)
        nus, acf_on_, acf_offs_, ms, m_errs, nu_dcs, nu_dc_errs, flags = self.calc_and_fit_acf(
            dspec_on_, dspec_offs_, freqs, time_slc_i, time_slc_j,
            ebar_scale_fac=ebar_scale_fac, freq_bins=freq_bins,
            plot=True, save_path=save_path, save_plot=save_plot, **kwargs)
        nus, acf_on0_, acf_offs0_, ms0, m_errs0, nu_dcs0, nu_dc_errs0, flags0 = self.calc_and_fit_acf(
            dspec_on_, dspec_offs_, freqs, 0, 0,
            ebar_scale_fac=ebar_scale_fac, freq_bins=freq_bins,
            plot=True, save_path=save_path, save_plot=False, exclude_zero=False, **kwargs)
        # print('mod indices from cross corr vs autocorr:', ms, ms0)
        print('mod indices and error-bars:')
        for m, m_err in zip(ms, m_errs):
            print(f'{m:.2f} +/- {m_err:.2f}')

        # save results
        if dump:
            rst_specs = {
                'spec_on_': spec_on_,
                'spec_offs_': spec_offs_,
                'freqs': freqs,
            }
            with open(save_path + '/specs_upchan{}.pkl'.format(spec_on_.shape[-1]//1024), 'wb') as f:
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
                'flags': flags,
                'acf_on0_': acf_on0_,
                'acf_offs0_': acf_offs0_,
                'ms0': ms0,
                'm_errs0': m_errs0,
                'nu_dcs0': nu_dcs0,
                'nu_dc_errs0': nu_dc_errs0,
                'flags0': flags0,
            }
            with open(save_path + '/acfs_upchan{}.pkl'.format(spec_on_.shape[-1]//1024), 'wb') as f:
                pickle.dump(rst_acfs, f)

def get_bbdata_fname(frb_id, path='/arc/projects/chime_frb/baseband_catalog/beamformed_files/'):
    fnames = glob.glob(path + '*.h5')
    for fname in fnames:
        if frb_id in fname:
            return fname
    return ''

def get_fitburst_fname(frb_id, fitburst_path='/arc/projects/chime_frb/Basecat_morph/fitburst_run_ketan/'):
    fnames = glob.glob(fitburst_path + f'/{frb_id}/results_fitburst*json')
    if len(fnames) == 0:
        print(f'no fitburst results found for {frb_id}')
        return ''
    if len(fnames) == 1:
        fname = fnames[0]
    else:
        fname = ''
        for i in range(len(fnames)):
            fnames[i] = fnames[i][fnames[i].index('results'):]
            if fnames[i] in [f'results_fitburst_scat_{frb_id}.json', f'results_fitburst_{frb_id}.json']:
                fname = fnames[i]
                print('by default using', fname)
                break
        if fname == '':
            fname = input(f'choose from these files to use for the DM {fnames}:')
        fname = fitburst_path + f'/{frb_id}/' + fname
    return fname

def get_fitburst_dm(frb_id, fitburst_path='/arc/projects/chime_frb/Basecat_morph/fitburst_run_ketan/'):
    fname = get_fitburst_fname(frb_id, fitburst_path=fitburst_path)
    if fname == '':
        return -1
    data = json.load(open(fname, 'r'))
    return data['initial_dm'] + data['model_parameters']['dm'][0]

def main():
    out_path = '/arc/projects/chime_frb/xiaohan/scintillation'
    fit_scint_pipeline = FitScintPipeline(data_reduction_pipeline_chime)

    frb_id = '24365582'
    fname = get_bbdata_fname(frb_id)
    if fname == '':
        print('no bbdata file name found; check frb ID')
        return
    fit_scint_pipeline.run(fname, DM=get_fitburst_dm(frb_id), out_path=out_path, interactive=False)

if __name__ == '__main__':
    main()
