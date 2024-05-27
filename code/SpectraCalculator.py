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
        fac = spectra.size // 1024
        if fac == 16:
            deripple_arr = np.array([
                0.867378  , 0.8610733 , 0.8365388 , 0.9638175 , 1.0759603 ,
                1.1530735 , 1.1878963 , 1.192497  , 1.195381  , 1.1915883 ,
                1.1769028 , 1.12039   , 1.0246618 , 0.9017509 , 0.79447484,
                0.8591826 ])
        elif fac == 32:
            deripple_arr = np.array([
                0.79593295, 0.803383  , 0.77029073, 0.7640511 , 0.81419826,
                0.8802964 , 0.9473549 , 1.0098867 , 1.0668812 , 1.1142088 ,
                1.1518388 , 1.176814  , 1.1937075 , 1.2041535 , 1.2087117 ,
                1.2093117 , 1.2113247 , 1.2106435 , 1.2073773 , 1.1992799 ,
                1.1878287 , 1.16624   , 1.1351968 , 1.092322  , 1.0408001 ,
                0.9801698 , 0.9152618 , 0.84713817, 0.78618824, 0.7552816 ,
                0.80179554, 0.7943418 ])
        elif fac == 64:
            deripple_arr = np.array([
                0.7452757 , 0.7290304 , 0.7387419 , 0.73082316, 0.7249346 ,
                0.73331326, 0.74989367, 0.77435184, 0.803317  , 0.83638924,
                0.8703127 , 0.90431976, 0.93790126, 0.97189003, 1.0023589 ,
                1.0331117 , 1.0619051 , 1.0872736 , 1.1126125 , 1.1309304 ,
                1.1503338 , 1.1625925 , 1.1749586 , 1.1867456 , 1.1937394 ,
                1.2002723 , 1.2050626 , 1.2085154 , 1.2092745 , 1.2126521 ,
                1.2126592 , 1.2128538 , 1.2122942 , 1.2157859 , 1.2133232 ,
                1.210204  , 1.2094959 , 1.2072551 , 1.2024918 , 1.197286  ,
                1.1920321 , 1.1825746 , 1.1705716 , 1.1573578 , 1.1420345 ,
                1.1216283 , 1.0995693 , 1.0777116 , 1.0490614 , 1.020217  ,
                0.9887928 , 0.95666605, 0.92113614, 0.8875626 , 0.85356015,
                0.81947607, 0.78786784, 0.7624131 , 0.74097383, 0.72755355,
                0.72567457, 0.72970754, 0.7380053 , 0.7397578 ])
        elif fac == 128:
            deripple_arr = np.array([
                0.6628779 , 0.6630977 , 0.66557133, 0.66622305, 0.6686244 ,
                0.67579204, 0.68200815, 0.68665195, 0.6942135 , 0.70430005,
                0.71289825, 0.72447205, 0.7365986 , 0.7519126 , 0.76464605,
                0.7805499 , 0.7969706 , 0.811682  , 0.82944804, 0.84496534,
                0.8658404 , 0.8813347 , 0.8991711 , 0.9173826 , 0.93241227,
                0.95205086, 0.9663987 , 0.9825004 , 0.99812835, 1.0141453 ,
                1.0291388 , 1.0427108 , 1.0568755 , 1.0714134 , 1.0832793 ,
                1.0979198 , 1.1087985 , 1.1197774 , 1.1295495 , 1.1379462 ,
                1.1494311 , 1.1561632 , 1.1632179 , 1.1690667 , 1.1722028 ,
                1.1809683 , 1.1866528 , 1.1914349 , 1.1951855 , 1.1993835 ,
                1.2015686 , 1.2079792 , 1.2090026 , 1.2111064 , 1.210189  ,
                1.2110314 , 1.211578  , 1.2154396 , 1.2190078 , 1.2178575 ,
                1.2173592 , 1.2192713 , 1.2177615 , 1.2197309 , 1.2135569 ,
                1.2215624 , 1.2212036 , 1.217231  , 1.2193189 , 1.2186979 ,
                1.216008  , 1.214735  , 1.2169948 , 1.2159383 , 1.2131416 ,
                1.2112048 , 1.2050207 , 1.2023501 , 1.1991618 , 1.1991509 ,
                1.1916056 , 1.189193  , 1.1835227 , 1.1809971 , 1.1745164 ,
                1.1683918 , 1.1581085 , 1.153397  , 1.1438788 , 1.1346438 ,
                1.1246089 , 1.112944  , 1.1024925 , 1.092185  , 1.0787216 ,
                1.0663044 , 1.0532495 , 1.0398548 , 1.0221529 , 1.0077356 ,
                0.99007064, 0.97449076, 0.9591087 , 0.94435704, 0.9239462 ,
                0.9084702 , 0.8925746 , 0.87392527, 0.8575313 , 0.83870625,
                0.8205463 , 0.8042214 , 0.7885378 , 0.7736779 , 0.7593024 ,
                0.7461531 , 0.7314771 , 0.7202165 , 0.70845836, 0.6990256 ,
                0.6913884 , 0.68568003, 0.67803675, 0.6725124 , 0.6681136 ,
                0.6651788 , 0.6638106 , 0.6632537 ])

    big_fix = np.tile(deripple_arr, 1024)
    spectra = np.multiply(spectra, 1/big_fix)
    return spectra

class SpectraCalculator():
    def __init__(self, ww, offpulse_range, freqs=None, f_power=f_power, fitburst_model=None):
        self.ww = ww.copy()
        self.offpulse_range = offpulse_range # the off-pulse range used for noise estimation etc.
        self.fitburst_model = fitburst_model / fitburst_model.max() if fitburst_model is not None else None

        if freqs is None:
            self.freqs = np.linspace(FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ, ww.shape[0])
        else:
            self.freqs = freqs.copy()
        self.n_freq = len(self.freqs)
        self.freq_min = self.freqs[-1]
        self.freq_max = self.freqs[0]

        self.power = f_power(self.ww) # total intensity

        # a rough estimate of the on-pulse region; need to call calc_on_range() to get a better estimate
        if self.fitburst_model is None:
            self.on_range = get_main_peak_lim(self.power, floor_level=0, diagnostic_plots=False, normalize_profile=True)
            self.on_ranges = [self.on_range] # allow for multiple components
            self.l_on = self.on_range[1] - self.on_range[0]
            # height of the on-range regions; need it to define filters
            self.h_ons = [1.]
            # construct a boxcar filter for later use
            self.filter = construct_filter_of_boxcars(self.on_ranges, self.h_ons)[np.newaxis,np.newaxis,:]
        else:
            self.calc_on_range()

        # the full time range of the FRB data
        self.time_range = (0, self.ww.shape[-1])

    def calc_on_range(self, ds_factor=16, interactive=False, **kwargs):
        '''
        Calculate the on-pulse region.
        '''
        # it's a completely different process if we have a fitburst model
        if self.fitburst_model is not None:
            flux = np.nanmean(self.power*self.fitburst_model, axis=0)
            flux /= np.nanmax(flux)
            tmp = np.where(flux > 1e-4)[0]
            ind_l = tmp[0]
            ind_r = tmp[-1]
            self.on_range = (ind_l, ind_r)
            self.on_ranges = [self.on_range]
            self.l_on = self.on_range[1] - self.on_range[0]
            self.h_ons = [1.]
            self.filter = self.fitburst_model[:,np.newaxis,self.on_range[0]:self.on_range[1]]
            return

        noise = np.nanmean(self.power[:,self.offpulse_range[0]:self.offpulse_range[1]])
        # TODO: I thought the lines below would help with automatic burst finidng but no, unfortunately.
        # maybe there's still improvement?  but I'd just do the burst finding interactively for now.
        # flux_filt = np.nanmean(scrunch(self.power, tscrunch=ds_factor, fscrunch=1), axis=0) - noise
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
            flux_filt = np.nanmean(scrunch(self.power, tscrunch=ds_factor, fscrunch=1), axis=0) - noise
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
        self.filter = construct_filter_of_boxcars(self.on_ranges, self.h_ons)[np.newaxis,np.newaxis,:]

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
                    **kwargs):
        '''
        Randomly sample N=50 noise spectrums from the off-pulse region.
        Upchannelizes and deripples noise spectrums from waterfall.
        '''
        N = 50 # number of noise spectrums to average
        # randomly select N noise ranges; xs indicates the left starting point
        xs = np.zeros(N, dtype=int)
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
                    time_slc = np.s_[:ww.shape[-1]//2]
                elif time_slc == 'second half':
                    time_slc = np.s_[ww.shape[-1]//2:]
                spec = calc_spec(ww, time_slc, f_spec=f_spec)
                spec_offs_[i][j] = f_deripple(spec)
        return np.asarray(spec_offs_), freqs

    def calc_spec_on(self, do_upchannel=True, fftsize=32, downfreq=2, deripple_arr=None, f_spec=f_spec_I, list_time_slcs=[np.s_[:]],
                    **kwargs):
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
                time_slc = np.s_[:ww.shape[-1]//2]
            elif time_slc == 'second half':
                time_slc = np.s_[ww.shape[-1]//2:]
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
