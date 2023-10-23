from baseband_analysis.core.signal import get_main_peak_lim, frequency_average
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.sampling import scrunch, clip
from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp, delay_across_the_band
import numpy as np
import matplotlib.pyplot as plt

# this function is adapted from baseband_analysis.core.signal
def get_weights_(power_org):
    """
    Get inverse variance weights for each freq channel from noise estimate.

    Parameters
    ----------
    power_org: ndarray
        The off-pulse voltages.

    Returns
    -------
    offset: ndarray
        Means of power.
    weight: ndarray
        Inverse stddev of noise.
    """
    # if len(power_org.shape) == 1:
    #     new_shape = 1
    #     spectrum_lim = False
    # elif len(power_org.shape) == 2:
    #     new_shape = f_id.size
    # elif len(power_org.shape) == 3:
    #     new_shape = [f_id.size, power_org.shape[1]]

    # Initial rescaling
    power = power_org.copy()
    power[power == 0] = np.nan
    mean = np.nanmedian(power, axis=-1)
    power -= mean[..., np.newaxis]
    std = np.nanstd(power, axis=-1)
    power /= std[..., np.newaxis]

    # First refinement
    idx_floor = np.isnan(frequency_average(power))
    floor = power_org.copy()
    floor[..., idx_floor] = np.nan
    mean = np.nanmean(floor, axis=-1)
    floor -= mean[..., np.newaxis]
    floor[np.isinf(floor)] = np.nan
    std = np.nanstd(floor, axis=-1)

    power = power_org.copy()
    power -= mean[..., np.newaxis]
    power /= std[..., np.newaxis]

    # Second refinement
    idx_floor = np.isnan(frequency_average(power))
    floor = power_org.copy()
    floor[..., idx_floor] = np.nan
    mean = np.nanmean(floor, axis=-1)
    floor -= mean[..., np.newaxis]
    floor[np.isinf(floor)] = np.nan
    std = np.nanstd(floor, axis=-1)

    power = power_org.copy()
    power -= mean[..., np.newaxis]
    power /= std[..., np.newaxis]

    # # Fit for signal with limited bandwidth
    # if spectrum_lim:
    #     f_lim = get_spectrum_lim(
    #         f_id, power, diagnostic_plots=diagnostic_plots, min_snr=spectrum_thresh
    #     )
    #     if f_lim[1] - f_lim[0] < 20:
    #         # Ignore if less than 20 channels are left
    #         log.warning("Less than 20 channels left after spectrum fit.")
    #     else:
    #         f_mask = np.nan + np.zeros(new_shape, dtype=float)
    #         f_mask[f_lim[0] : f_lim[1]] = 0
    #         mean += f_mask
    #         std += f_mask

    offset = mean
    weight = np.array(1.0 / std)
    weight[~np.isfinite(weight)] = 0
    return offset, weight

def normalize_data(data, data_offpulse):
    '''
    data should be the dedispersed voltage data, shape (freq, pol, time).
    data_offpulse is the pre-selcted off-pulse range used to calculate the mean and std along the time axis.
    '''
    offsets, weights = get_weights_(data_offpulse)
    return (data - offsets[..., np.newaxis]) * weights[..., np.newaxis]

def clip_data(bbdata, valid_span_power_bins, DM):
    # bottom corners taken from get_snr
    valid_span_unix_bottom = bbdata["time0"]["ctime"][-1] + 2.56e-6 * np.array(valid_span_power_bins)

    # top corners taken from data shape
    dm_delay_sec = delay_across_the_band(
        DM=DM,
        freq_high=bbdata.index_map["freq"]["centre"][0],
        freq_low=bbdata.index_map["freq"]["centre"][-1],
    )

    valid_span_unix_top = (
        bbdata["time0"]["ctime"][0]
        + dm_delay_sec
        + np.array([0, bbdata.ntime]) * 2.56e-6
    )

    valid_times = [
        max(valid_span_unix_bottom[0], valid_span_unix_top[0]),
        min(valid_span_unix_bottom[1], valid_span_unix_top[1]),
    ]

    # translate times to TOA and duration; run the clipper
    toa_400 = np.mean(
        valid_times
    )  # calculate the center of the data within the valid region
    duration = valid_times[1] - valid_times[0]  # calculate duration in seconds

    data_clipped = clip(
        bbdata,
        toa_400=toa_400,
        duration=duration,
        ref_freq=bbdata.index_map["freq"]["centre"][-1],
        dm=DM,
        inplace=False,
        pad=True,
    )
    return data_clipped

def dedisperse_and_normalize_bbdata(bbdata, DM, downsample_factor=32, interactive=False,
                                    plot=True, save_path='', save_plot=False, **kwargs):
    """
    given a bbdata, dedisperse and normalize it
    output the data (RFI zapped, missing channels filled, derippled, coherent dedispersed),
      the frequencies and frequency channel IDs
    """
    # freq_id, freq, power, offset, weight, valid_channels_out, time_range_out, DM_out, downsampling_factor = get_snr(...)
    try:
        output = get_snr(bbdata, DM=DM, DM_range=None, diagnostic_plots=True, return_full=True,
                            downsample=downsample_factor, spectrum_lim=False)
        valid_channels = output[5]
        # TODO: check if sometimes the dumped bbdata seems problematic
        # valid_span_power_bins = output[6]
        d = np.abs(bbdata['tiedbeam_baseband'][-1,0,:])
        ind = -1
        for i in range(2, len(d)):
            if np.all(np.isnan(d[-i:])):
                ind = len(d) - i
            else:
                break
        valid_span_power_bins = [0, min(ind, output[6][1])]
        if DM is None: # in case input DM is None
            DM = output[7]
    except:
        print('get_snr failed')
        return

    # # clip
    # try:
    #     bbdata = clip_data(bbdata, valid_span_power_bins, DM)
    # except:
    #     print('clip_data failed')
    #     return

    # dedisperse 1
    data = coherent_dedisp(bbdata, DM, time_shift=False)

    # dedisperse 2
    data = incoherent_dedisp(bbdata, DM, matrix_in=data, fill_wfall=False)[0]
    data = data[:,:,valid_span_power_bins[0]:valid_span_power_bins[1]]

    # identify off burst region to use
    Iscr0 = scrunch(np.sum(np.abs(data)**2, axis=1), tscrunch=downsample_factor, fscrunch=1)
    nanind = np.where(np.isnan(Iscr0[-1]))[0]
    if len(nanind) == 0:
        nanind = Iscr0.shape[1]
    else:
        nanind = nanind[0]
    st_tbin, end_tbin = get_main_peak_lim(Iscr0[:,:nanind+500], diagnostic_plots=False, normalize_profile=True)
    if interactive:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        vmin, vmax = np.nanpercentile(Iscr0, [5, 95])
        plt.imshow(Iscr0, aspect='auto', vmin=vmin, vmax=vmax)
        plt.subplot(122)
        plt.plot(np.nanmean(Iscr0, axis=0))
        plt.show()
        answer = input(f'Please define the bin range to use for the off burst statistics (beginbin,endbin), reference {st_tbin}, {end_tbin}, {nanind}: ')
        answer = answer.split(',')
        offpulse_range = [int(answer[0]), int(answer[1])]
    else:
        offpulse_range = [0, st_tbin] # usually prefer this range
        if st_tbin < nanind - end_tbin: # but in case st_tbin is too close to 0
            offpulse_range = [end_tbin, nanind]

    # subtract mean and divide by std per channel
    data = normalize_data(data, data[:,:,offpulse_range[0]*downsample_factor:offpulse_range[1]*downsample_factor])

    # get rid of invalid channels as determined by get_snr
    data[~valid_channels] = 0

    # # figure out what range of data we want to keep
    # I = np.sum(np.abs(data)**2, axis=1)
    # Iscr = scrunch(I, tscrunch=downsample_factor, fscrunch=1)
    # st_tbin, end_tbin = get_main_peak_lim(Iscr[:,:nanind+500], diagnostic_plots=False, normalize_profile=True)
    # if interactive==True:
    #     plt.figure(figsize=(6,10))
    #     plt.subplot(211)
    #     plt.imshow(Iscr,aspect='auto')
    #     plt.subplot(212)
    #     plt.plot(np.nanmean(Iscr,axis=0))
    #     plt.show()
    #     answer=input(f'Please define the time bin range to keep (beginbin,endbin), reference {st_tbin}, {end_tbin}, {nanind}): ')
    #     answer = answer.split(',')
    # else:
    #     l = 50*(end_tbin - st_tbin)
    #     answer = [max(st_tbin - l, 0), min(end_tbin + l, nanind)]

    # fill missing channels with 0s
    data = fill_missing_chans(data,
        # data[:,:,int(answer[0])*downsample_factor:int(answer[1])*downsample_factor],
        bbdata.index_map["freq"]["id"], bbdata.index_map["freq"]["centre"])

    # TODO: I still think missing data should be nan's, not 0's
    data[data==0] = np.nan
    # but if there's nan's somewhere along the time axis, set to 0 (for now)
    for i in range(data.shape[0]):
        ii = np.isnan(data[i,0,:])
        n = ii.sum()
        if n != 0 and n != data.shape[-1]:
            print('NaNs found at freq', i, np.where(ii)[0][0], n)
            data[i,:,ii] = 0

    # show how we did
    if plot:
        fig = plt.figure(figsize=(12,12))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.95)
        # the original bbdata
        plt.subplot(221)
        Iscr = scrunch(np.sum(np.abs(bbdata['tiedbeam_baseband'])**2, axis=1), tscrunch=downsample_factor, fscrunch=1)
        vmin, vmax = np.nanpercentile(Iscr, [5, 95])
        plt.imshow(Iscr, vmin=vmin, vmax=vmax, aspect='auto')
        # the final results of our processing
        plt.subplot(222)
        Iscr = scrunch(np.sum(np.abs(data)**2, axis=1), tscrunch=downsample_factor, fscrunch=1)
        vmin, vmax = np.nanpercentile(Iscr, [5, 95])
        plt.imshow(Iscr, vmin=vmin, vmax=vmax, aspect='auto')
        # time series before normalizing; I just wanted to see what the noise is like
        plt.subplot(223)
        plt.plot(np.nanmean(Iscr0, axis=0))
        plt.axvline(x=offpulse_range[0], color='k', linestyle='--')
        plt.axvline(x=offpulse_range[1], color='k', linestyle='--')
        # time series of the final results
        plt.subplot(224)
        plt.plot(np.nanmean(Iscr, axis=0))
        plt.axvline(x=offpulse_range[0], color='k', linestyle='--')
        plt.axvline(x=offpulse_range[1], color='k', linestyle='--')
        if save_plot:
            plt.savefig(save_path + '/dedispersed_data.png')
        plt.show()

    offpulse_range[0] *= downsample_factor
    offpulse_range[1] *= downsample_factor
    return data, offpulse_range

def fill_missing_chans(data, freq_id, freqs):
    """
    data: shape [freq<1024,pol,time]
    freq_id, freqs: index and frequency of each channel
    """
    # fill missing channels with 0s
    new_data = np.zeros([1024, data.shape[1], data.shape[2]], dtype=data.dtype)
    new_data[freq_id,:,:] = data

    # # get the new frequency array
    # df = (freqs[1] - freqs[0]) / (freq_id[1] - freq_id[0])
    # f0 = freqs[0] - df * freq_id[0]
    # new_freqs = np.linspace(f0, f0+df*1023, 1024)
    return new_data#, new_freqs
