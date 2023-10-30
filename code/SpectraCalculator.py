import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq
from baseband_analysis.core.sampling import _upchannel as upchannel 
from baseband_analysis.core.sampling import scrunch
from baseband_analysis.core.signal import get_main_peak_lim
from chime_frb_constants import FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ
from data_reduction_functions import get_weights_

f_spec_I = lambda ww: np.sum(np.abs(ww)**2, axis=(1,2)) # Stokes I from baseband data
f_power = lambda ww: np.sum(np.abs(ww)**2, axis=1) # total intensity from baseband data

def calc_spec(ww, time_slc, f_spec=f_spec_I):
    '''
    Calculate spectrum by summing over time of the baseband data.
    The summation is done over the time bins specified by the slice time_slc, e.g. np.s_[::2], np.s_[:].
    By default, calculate Stokes I (sum over both polarization and time; not nansum).
    By adjusting f_spec, can also do other Stokes parameters.
    '''
    return f_spec(ww[:,:,time_slc])

def calc_deripple_arr(ww, fftsize=32, downfreq=2):
    '''
    Calculate the deripple array of upchannelization for an off-pulse waterfall array.
    '''
    # upchannelize
    ww_up = upchannel(ww, np.arange(ww.shape[0]), fftsize=fftsize, downfreq=downfreq)[0]
    ww_up = np.swapaxes(ww_up, 1, 2)
    ww_up = np.swapaxes(ww_up, 0, 1)

    # calculate deripple function
    flux = (np.abs(ww_up)**2).sum(axis=(1,2))
    flux[flux == 0] = np.nan
    offset, weight = get_weights_(flux)
    # show diagnostic plot
    plt.plot(flux)
    plt.axhline(y=offset, color='k', linestyle='dashed')
    plt.ylim(offset-2/weight, offset+2/weight)
    plt.show()

    flux1 = flux.reshape(ww.shape[0],-1)
    flux1[np.abs(flux1 - offset) > 1/weight] = np.nan
    flux1 = np.nanmean(flux1, axis=0)
    # show diagnostic plot
    plt.plot(flux1/offset)
    plt.show()

    return flux1/offset

def deripple(spectra, deripple_arr=None):
    if deripple_arr is None:
        deripple_arr = np.array([0.85342883, 0.8473684 , 0.8372402 , 0.96410074, 1.07566724,
       1.15265638, 1.18938046, 1.19670342, 1.19823932, 1.19589376,
       1.17811194, 1.11955408, 1.02364112, 0.90226406, 0.79020459,
       0.8486396 ])
    big_fix = np.tile(deripple_arr, 1024)
    spectra = np.multiply(spectra, 1/big_fix)
    return spectra

class SpectraCalculator():
    def __init__(self, ww, offpulse_range, freqs=None, f_power=f_power):
        self.ww = ww.copy()
        self.offpulse_range = offpulse_range # the off-pulse range used for noise estimation etc.

        if freqs is None:
            self.freqs = np.linspace(FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ, ww.shape[0])
        else:
            self.freqs = freqs.copy()
        self.n_freq = len(self.freqs)
        self.freq_min = self.freqs[-1]
        self.freq_max = self.freqs[0]

        self.power = f_power(self.ww) # total intensity

        # a rough estimate of the on-pulse region; need to call calc_on_range() to get a better estimate
        self.on_range = get_main_peak_lim(self.power, floor_level=0, diagnostic_plots=False, normalize_profile=True)
        self.on_ranges = [self.on_range] # allow for multiple components
        self.l_on = self.on_range[1] - self.on_range[0]
        # height of the on-range regions; need it to define filters
        self.h_ons = [1.]
        # construct a boxcar filter for later use
        self.filter = construct_filter_of_boxcars(self.on_ranges, self.h_ons)

        # the full time range of the FRB data
        self.time_range = (0, self.ww.shape[-1])

    def calc_on_range(self, ds_factor=16, interactive=False, **kwargs):
        '''
        Calculate the on-pulse region.
        '''
        noise = np.nanmean(self.power[:,self.offpulse_range[0]:self.offpulse_range[1]])
        flux_filt = np.nanmean(scrunch(self.power, tscrunch=ds_factor, fscrunch=1), axis=0) - noise
        # TODO: I thought the lines below would help with automatic burst finidng but no, unfortunately.
        # maybe there's still improvement?  but I'd just do the burst finding interactively for now.
        # if np.max(flux_filt) > 2*noise:
        #     tmp = flux_filt[1:]*flux_filt[:-1]
        #     ind_l, ind_r = np.where(tmp < 0)[0][0, -1]
        #     self.on_range = (ind_l * ds_factor + ds_factor / 2, ind_r * ds_factor + ds_factor / 2)
        #     self.l_on = self.on_range[1] - self.on_range[0]

        ## use convolution with the boxcar kernel to determine on-pulse range
        ts = np.nanmean(self.power, axis=0) - noise
        if not interactive:
            ind_l = max(self.on_range[0] - 2*self.l_on, 0)
            ind_r = min(self.on_range[1] + 2*self.l_on, len(ts))
            peak, width, _ = find_burst(ts[ind_l:ind_r], max_width=(ind_r-ind_l))
            ind_l = peak - width//2 + ind_l
            ind_r = ind_l + width + 1
            self.on_range = [ind_l, ind_r]
            self.on_ranges = [self.on_range]
            self.h_ons = [1.]
        else:
            _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
            for ax in axes:
                ax.plot(ts)
                ax.plot(np.arange(len(flux_filt)) * ds_factor + ds_factor/2, flux_filt)
            # zoom in on the second subplot
            axes[1].set_xlim(self.on_range[0]-1000, self.on_range[1]+1000)
            plt.show()
            i = 0
            while True:
                answer = input('Please define a region (beginbin,endbin) to find a pulse : ')
                answer = answer.split(',')
                answer[0] = int(answer[0])
                answer[1] = min(int(answer[1]), len(ts))
                peak, width, snr = find_burst(ts[answer[0]:answer[1]], max_width=answer[1]-answer[0])
                ind_l = peak - width//2 + answer[0]
                ind_r = peak + width//2 + 1 + answer[0]
                if i < len(self.on_ranges):
                    self.on_ranges[i] = (ind_l, ind_r)
                    self.h_ons[i] = snr # can try different values
                else:
                    self.on_ranges.append((ind_l, ind_r))
                    self.h_ons.append(snr)
                i += 1
                answer = input('Do you want to work on another component? (y/n): ')
                if answer == 'n':
                    break
            # get rid of previous runs if there are any
            self.on_ranges = self.on_ranges[:i]
            self.h_ons = self.h_ons[:i]
            # entire on-range
            self.on_range = find_bounds_on_range(self.on_ranges)

        self.l_on = self.on_range[1] - self.on_range[0]

        # construct filter of boxcars
        self.filter = construct_filter_of_boxcars(self.on_ranges, self.h_ons)

    def calc_deripple_arr(self, fftsize=32, downfreq=2, interactive=True):
        if interactive:
            d = np.sum(np.abs(self.ww)**2, axis=1)
            vmin, vmax = np.nanpercentile(d[~np.isnan(d)], [5, 95])
            plt.imshow(d, vmin=vmin, vmax=vmax, aspect='auto')
            plt.show()
            answer = input('Please define the bin range to use for the off burst statistics (beginbin,endbin): ')
            answer = answer.split(',')
            ww_off = self.ww[:,:,int(answer[0]):int(answer[1])]
        else:
            ww_off = np.concatenate((self.ww[:,:,0:self.on_range[0]], self.ww[:,:,self.on_range[1]:]), axis=2)

        return calc_deripple_arr(ww_off, fftsize=fftsize, downfreq=downfreq)

    def calc_spec_off(self, do_upchannel=True, fftsize=32, downfreq=2, deripple_arr=None, f_spec=f_spec_I, list_time_slcs=[np.s_[:]],
                    separate_components=False, **kwargs):
        '''
        Randomly sample N=50 noise spectrums from the off-pulse region.
        Upchannelizes and deripples noise spectrums from waterfall.
        '''
        N = 50 # number of noise spectrums to average
        # randomly select N noise ranges; xs indicates the left starting point
        xs = np.zeros(N, dtype=int)
        # the entire on-range
        
        # left of on_range
        n = int(N*(self.on_range[0] - self.time_range[0] - self.l_on)/(self.time_range[1] - self.time_range[0] - self.on_range[1] + self.on_range[0] - 2*self.l_on))
        n = min(n, N)
        print(n, 'noise ranges left of on_range')
        xs[:n] = np.random.choice(np.arange(self.time_range[0], self.on_range[0] - self.l_on), n)
        # right of on_range
        xs[n:] = np.random.choice(np.arange(self.on_range[1], self.time_range[1] - self.l_on), N-n)
        # print(xs[:n], xs[n:])

        freqs = self.freqs
        f_deripple = lambda x: x
        if do_upchannel:
            f_deripple = lambda x: deripple(x, deripple_arr=deripple_arr)

        spec_offs_ = [[None] * N for _ in list_time_slcs] # for each of the N off-pulse regions, get the spectra according to the time slices

        for j, x in enumerate(xs):
            ww = self.ww[:,:,x:x+self.l_on] * self.filter
            if do_upchannel:
                ww, freqs = upchannel(ww, np.arange(self.n_freq), fftsize=fftsize, downfreq=downfreq)[:2] # ww[freq, pol, time]; return [pol, time//32, freq*16]
                ww = np.swapaxes(ww, 1, 2)
                ww = np.swapaxes(ww, 0, 1) # back to [freq, pol, time]
            for i, time_slc in enumerate(list_time_slcs):
                if time_slc == 'first half':
                    time_slc = np.s_[:self.l_on//2]
                elif time_slc == 'second half':
                    time_slc = np.s_[self.l_on//2:]
                spec = calc_spec(ww, time_slc, f_spec=f_spec)
                spec_offs_[i][j] = f_deripple(spec)
        return np.asarray(spec_offs_), freqs

    def calc_spec_on(self, do_upchannel=True, fftsize=32, downfreq=2, deripple_arr=None, f_spec=f_spec_I, list_time_slcs=[np.s_[:]],
                    separate_components=False, **kwargs):
        '''
        Upchannelizes and deripples on-pulse spectrum from waterfall.
        '''
        ww = self.ww[:,:,self.on_range[0]:self.on_range[1]] * self.filter

        freqs = self.freqs
        f_deripple = lambda x: x
        if do_upchannel:
            f_deripple = lambda x: deripple(x, deripple_arr=deripple_arr)
            ww, freqs = upchannel(ww, np.arange(self.n_freq), fftsize=fftsize, downfreq=downfreq)[:2] # ww[freq, pol, time]; return [pol, time//32, freq*16]
            ww = np.swapaxes(ww, 1, 2)
            ww = np.swapaxes(ww, 0, 1) # back to [freq, pol, time]

        # for each time slice, calculate a spectrum and return a list of spectra
        spec_on_ = [None]*len(list_time_slcs)
        for i, time_slc in enumerate(list_time_slcs):
            if time_slc == 'first half':
                time_slc = np.s_[:self.l_on//2]
            elif time_slc == 'second half':
                time_slc = np.s_[self.l_on//2:]
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

# codes from Ziggy
def boxcar_kernel(width):
    """Returns the boxcar kernel of given width normalized by sqrt(width) for S/N reasons.
    Parameters
    ----------
    width : int
        Width of the boxcar.
    Returns
    -------
    boxcar : array_like
        Boxcar of width `width` normalized by sqrt(width).
    """
    width = int(round(width, 0))
    return np.ones(width) / width**0.5

def find_burst(ts, min_width=1, max_width=128):
    """Find burst peak and width using boxcar convolution.
    Parameters
    ----------
    ts : array_like
        Time-series.
    min_width : int, optional
        Minimum width to search from, in number of time samples.
        1 by default.
    max_width : int, optional
        Maximum width to search up to, in number of time samples.
        128 by default.
    Returns
    -------
    peak : int
        Index of the peak of the burst in the time-series.
    width : int
        Width of the burst in number of samples.
    snr : float
        S/N of the burst.

    """
    min_width = int(min_width)
    max_width = int(max_width)

    # do not search widths bigger than timeseries
    widths = list(range(min_width, min(max_width + 1, len(ts)-2)))

    # envelope finding
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)

    for i in range(len(widths)):
        convolved = scipy.signal.convolve(ts, boxcar_kernel(widths[i]), mode="same")
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]

    best_idx = np.nanargmax(snrs)

    return peaks[best_idx], widths[best_idx], snrs[best_idx]

def find_bounds_on_range(on_ranges):
    '''
    Find the left and right most boundaries of on_ranges.
    '''
    ind_l = on_ranges[0][0]
    ind_r = on_ranges[0][1]
    for on_range in on_ranges[1:]:
        ind_l = min(on_range[0], ind_l)
        ind_r = max(on_range[1], ind_r)
    return ind_l, ind_r

def construct_filter_of_boxcars(on_ranges, h_ons):
    '''
    Construct a filter containing a bunch on boxcars with boundaries defined by on_ranges and heights given by h_ons.
    '''
    ind_l, ind_r = find_bounds_on_range(on_ranges)
    filter = np.zeros(ind_r - ind_l)
    for i, on_range in enumerate(on_ranges):
        filter[on_range[0]-ind_l:on_range[1]-ind_l] = h_ons[i]
    return filter
