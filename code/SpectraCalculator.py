import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq
from baseband_analysis.core.sampling import _upchannel as upchannel 
from baseband_analysis.core.signal import get_main_peak_lim

f_spec_I = lambda ww: np.sum(np.abs(ww)**2, axis=(1,2)) # Stokes I from baseband data
f_power = lambda ww: np.nansum(np.abs(ww)**2, axis=1) # total intensity from baseband data

def calc_spec(ww, time_slc, f_spec=f_spec_I):
    '''
    Calculate spectrum by summing over time of the baseband data.
    The summation is done over the time bins specified by the slice time_slc, e.g. np.s_[::2], np.s_[:].
    By default, calculate Stokes I (sum over both polarization and time; not nansum).
    By adjusting f_spec, can also do other Stokes parameters.
    '''
    return f_spec(ww[:,:,time_slc])

def deripple(spectra):
    fix = np.array([0.67461854, 0.72345924, 0.8339084 , 0.96487784, 1.0780604 ,
       1.1574216 , 1.2020986 , 1.2204636 , 1.2236487 , 1.2168874 ,
       1.1857561 , 1.1230892 , 1.0274547 , 0.9002419 , 0.7799086 ,
       0.68808776])
    big_fix = np.tile(fix, 1024)
    spectra = np.multiply(spectra, 1/big_fix)
    return spectra

class SpectraCalculator():
    def __init__(self, ww, n_freq=1024, freq_min=400., freq_max=800., f_power=f_power):
        self.ww = ww.copy()

        self.n_freq = n_freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freqs = np.linspace(self.freq_max, self.freq_min, self.n_freq, endpoint=False)

        self.power = f_power(self.ww) # total intensity

        # a rough estimate of the on-pulse region; need to call calc_on_range() to get a better estimate
        self.on_range = get_main_peak_lim(self.power, floor_level=0.1, diagnostic_plots=False, normalize_profile=True)
        # length of the on-range
        self.l_on = self.on_range[1] - self.on_range[0]
        # the full time range of the FRB data
        self.time_range = (0, self.ww.shape[-1])

    def plot_waterfall(self, ax=None):
        vmin, vmax = np.nanpercentile(self.power[~np.isnan(self.power)], [5, 95])
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.imshow(self.power, aspect='auto', origin='lower', extent=[0, self.ww.shape[-1], self.freq_max, self.freq_min],
                   vmin=vmin, vmax=vmax)
        ax.vlines(self.on_range, self.freq_min, self.freq_max, colors='r', linestyles='dashed')
        ax.set_xlim([self.on_range[0]-1000, self.on_range[1]+1000])
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Frequency (MHz)')
        ax.invert_yaxis()
        return ax

    def calc_on_range(self, ds_factor=16, floor_level=0.1, plot=False, ax=None):
        '''
        Calculate the on-pulse region by obtaining the matched filter.
        '''
        flux_filt, noise = get_smooth_matched_filter(self.power, ds_factor=ds_factor, floor_level=floor_level)
        flux_filt, noise = flux_filt[0], noise[0]
        ind = np.argmax(flux_filt)
        if flux_filt[ind] < 3*noise:
            print('Warning: no on-pulse region found!')
            self.on_range = (np.nan, np.nan)
            self.l_on = np.nan
            return None, None, None
        ind_l = np.argmin(np.abs(flux_filt[:ind] - 3*noise))
        ind_r = np.argmin(np.abs(flux_filt[ind:] - 3*noise))+ind
        self.on_range = (ind_l, ind_r)
        self.l_on = self.on_range[1] - self.on_range[0]
        fig = None
        if plot:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            ax.plot(np.nanmean(self.power, axis=0) - np.nanmean(self.power[:,:ind_l]))
            ax.plot(flux_filt)
            ax.hlines(3*noise, 0, len(flux_filt), colors='k', linestyles='dashed')
            ax.vlines([ind_l, ind_r], 0, np.max(flux_filt), colors='r', linestyles='dotted')
            ax.set_xlim([ind_l-100, ind_r+100])
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Flux')
        return flux_filt, noise, ax

    def calc_spec_off(self, do_upchannel=True, f_spec=f_spec_I, list_time_slcs=[np.s_[:]]):
        '''
        Randomly sample N=50 noise spectrums from the off-pulse region.
        Upchannelizes and deripples noise spectrums from waterfall.
        '''
        N = 50 # number of noise spectrums to average
        # randomly select N noise ranges; xs indicates the left starting point
        xs = np.zeros(N, dtype=int)
        # left of on_range
        n = int(N*(self.on_range[0] - self.time_range[0] - self.l_on)/(self.time_range[1] - self.on_range[1] + self.on_range[0] - self.time_range[0] - self.l_on*2))
        xs[:n] = np.random.choice(np.arange(self.time_range[0], self.on_range[0] - self.l_on), n)
        # right of on_range
        xs[n:] = np.random.choice(np.arange(self.on_range[1], self.time_range[1] - self.l_on), N-n)
        # print(xs[:n], xs[n:])

        freqs = self.freqs
        f_deripple = lambda x: x
        if do_upchannel:
            f_deripple = deripple

        spec_offs_ = [[None] * N for _ in list_time_slcs] # for each of the N off-pulse regions, get the spectra according to the time slices

        for j, x in enumerate(xs):
            ww = self.ww[:,:,x:x+self.l_on].copy()
            if do_upchannel:
                ww, freqs = upchannel(self.ww[:,:,x:x+self.l_on], np.arange(self.n_freq))[:2] # ww[freq, pol, time]; return [pol, time//32, freq*16]
                ww = np.swapaxes(ww, 1, 2)
                ww = np.swapaxes(ww, 0, 1) # back to [freq, pol, time]
            for i, time_slc in enumerate(list_time_slcs):
                spec = calc_spec(ww, time_slc, f_spec=f_spec)
                spec_offs_[i][j] = f_deripple(spec)
        return np.asarray(spec_offs_), freqs

    def calc_spec_on(self, do_upchannel=True, f_spec=f_spec_I, list_time_slcs=[np.s_[:]]):
        '''
        Upchannelizes and deripples on-pulse spectrum from waterfall.
        '''
        ww = self.ww[:,:,self.on_range[0]:self.on_range[1]].copy()

        freqs = self.freqs
        f_deripple = lambda x: x
        if do_upchannel:
            f_deripple = deripple
            ww, freqs = upchannel(ww, np.arange(self.n_freq))[:2] # ww[freq, pol, time]; return [pol, time//32, freq*16]
            ww = np.swapaxes(ww, 1, 2)
            ww = np.swapaxes(ww, 0, 1) # back to [freq, pol, time]

        # for each time slice, calculate a spectrum and return a list of spectra
        spec_on_ = [None]*len(list_time_slcs)
        for i, time_slc in enumerate(list_time_slcs):
            spec = calc_spec(ww, time_slc, f_spec=f_spec)
            spec_on_[i] = f_deripple(spec)
        return np.asarray(spec_on_), freqs

def get_smooth_matched_filter(
    bb_d,
    ds_factor=40,
    floor_level=0.1,
    smooth_factor=None,
):
    """
    Get a gaussian smoothed matched filter from a waterfall array.
    Parameters
    ----------
    bb_d : np.array(freq,pol,frames) or np.array(freq,frames)
        Waterfall array.
    ds_factor : integer (optional)
        Downsampling factor (number of frames), automatically selected if not input
    floor_level : float (optional)
        Level from the floor from which the width of a pulse is determined.
        e.g. 0.5 is FWHM
    smoothing_factor : None or float (optional)
       manually apply a smoothing factor for the gaussian filter
    Returns
    -------
    flux_filt : np.array(freq,frames)
        Filter profile from the power of the input waterfall array.
    """
    if smooth_factor is None:
        smooth_factor = ds_factor

    # downsample
    power = bb_d*1.
    if len(power.shape) == 3:
        power = np.nansum(np.abs(power) ** 2, axis=1)
    power_ds = power[:, : power.shape[1] // ds_factor * ds_factor]
    power_ds = np.nanmean(
        power_ds.reshape(
            [power_ds.shape[0], power_ds.shape[1] // ds_factor, ds_factor]
        ),
        axis=-1,
    )

    ts = np.nanmean(power_ds, axis=0)
    lim = (
        np.array(get_main_peak_lim(ts, floor_level=floor_level, normalize_profile=True))
        * ds_factor
    )

    if len(bb_d.shape) == 3:
        flux = np.nanmean(np.abs(bb_d) ** 2, axis=0)
    else:
        flux = np.nanmean(bb_d, axis=0)[np.newaxis,:]
    noise_temp = np.nanmedian(flux, axis=-1)
    flux_c = flux.copy() - noise_temp[..., None]
    noise = np.nanstd(np.concatenate((flux_c[:,:lim[0]], flux_c[:,lim[1]:]), axis=1), axis=1)
    flux_c[flux_c < 0] = 0
    flux_c[:, 0 : lim[0]] = 0
    flux_c[:, lim[1] :] = 0

    # convolve
    k = rfftfreq(flux_c.shape[-1])
    flux_filt = irfft(
        rfft(flux_c, axis=-1) * np.exp(-np.pi**2 * k**2 * smooth_factor**2),
        axis=-1,
    )
    flux_filt[..., 0 : lim[0]] = 0
    flux_filt[..., lim[1] :] = 0

    return flux_filt, noise
