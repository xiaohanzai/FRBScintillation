import numpy as np
import matplotlib.pyplot as plt
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.dev.lensing import get_dedispersed_clipped_bbdata
from baseband_analysis.core.dedispersion import delay_across_the_band
# from baseband_analysis.analysis.snr import get_snr
# from baseband_analysis.core.sampling import fill_waterfall, clip
# from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power, get_weights, squarewave_pattern_index, get_onsource_beam
# from baseband_analysis.core.dedispersion import delay_across_the_band, coherent_dedisp
# from baseband_analysis.core.flagging import get_valid_time_range

class WaterfallLoader():
    '''
    Load in waterfall data given a data reduction pipeline, e.g. Zarif's get_dedispersed_clipped_bbdata.
    For CHIME, the returned data is dedispersed baseband voltages with shape (1024, 2, time).
    For Masui+15, the data stored on disk is IQUV so we need a format compatible with CHIME.
    '''
    def __init__(self, data_reduction_pipeline):
        '''
        Parameters
        ----------
        data_reduction_pipeline : function
            A function that takes in a filename and returns the waterfall data with shape (freq, 2, time).
        '''
        self.data_reduction_pipeline = data_reduction_pipeline

    def load_data(self, fname, **kwargs):
        return self.data_reduction_pipeline(fname, **kwargs)

def data_reduction_pipeline_chime(fname, floor_level=0.1, **kwargs):
    data = BBData.from_file(fname)
    ww = get_dedispersed_clipped_bbdata(data, downsampling_factor=1, diagnostic_plots=False, full_output=False,
                                        floor_level=floor_level, **kwargs) # shape (freq, pol, time)
    ww[ww == 0] = np.nan
    return ww

def data_reduction_pipeline_masui(fname):
    # fname is either filtered_short or filtered
    path = '/arc/home/xiaohanwu/scintillation/data/Masui15data/'
    postfix = ''
    if fname == 'filtered_short':
        postfix = '_short'

    # load frequency and time
    times = np.load(path + '/time%s.npy' % postfix)
    freqs = np.load(path + '/freq.npy').astype(float)

    # t0, DM, amp_800, alpha, width, scatter_800
    fit_pars = [54290.138578290294, 623.35540269174237, 0.0039736854175394781, 
            7.8287795616885543, 0.00071318994294502418, 0.00160942937546788]
    # # change t0 to frame number
    # fit_pars[0] = int((fit_pars[0] - times[0]) / (times[1] - times[0]))

    # load data
    data = np.load(path + fname + '.npy')
    data = incoherent_dedisp(data, fit_pars[1], freqs, times)
    return data[:,:1,:]

def incoherent_dedisp(data, DM, freqs, times):
    """
    Apply incoherent dedispersion.  Adapted from the original CHIME code to work with Masui+15 data.
    Parameters
    ----------
    data: array_like
        Data to dedisperse.  Shape (freq, 2 pol, time).
    DM: float
        DM to apply for dedispersion.
    freqs: array_like
        Frequencies.
    times: array_like
        The corresponding times of data.
    Returns
    -------
    ndarray of complex128:
        Waterfall data.
    """
    wfall = np.zeros_like(data)

    f_ref = freqs[0]
    dt = np.median(np.diff(times))
    for i in range(data.shape[0]):
        start_t = delay_across_the_band(DM, f_ref, freqs[i])
        bins_shift = np.round(start_t / dt).astype(int)
        wfall[i] = np.roll(data[i], bins_shift, axis=-1)

    return wfall

## These are functions from Ziggy; I don't need them right now
# def boxcar_kernel(width):
#     """Returns the boxcar kernel of given width normalized by sqrt(width) for S/N reasons.
#     Parameters
#     ----------
#     width : int
#         Width of the boxcar.
#     Returns
#     -------
#     boxcar : array_like
#         Boxcar of width `width` normalized by sqrt(width).
#     """
#     width = int(round(width, 0))
#     return np.ones(width, dtype="float32") / np.sqrt(width)

# def find_burst(ts, min_width=1, max_width=128):
#     """Find burst peak and width using boxcar convolution.
#     Parameters
#     ----------
#     ts : array_like
#         Time-series.
#     min_width : int, optional
#         Minimum width to search from, in number of time samples.
#         1 by default.
#     max_width : int, optional
#         Maximum width to search up to, in number of time samples.
#         128 by default.
#     Returns
#     -------
#     peak : int
#         Index of the peak of the burst in the time-series.
#     width : int
#         Width of the burst in number of samples.
#     snr : float
#         S/N of the burst.

#     """
#     min_width = int(min_width)
#     max_width = int(max_width)

#     # do not search widths bigger than timeseries
#     widths = list(range(min_width, min(max_width + 1, len(ts)-2)))

#     # envelope finding
#     snrs = np.empty_like(widths, dtype=float)
#     peaks = np.empty_like(widths, dtype=int)

#     for i in range(len(widths)):
#         convolved = scipy.signal.convolve(ts, boxcar_kernel(widths[i]),
#                                           mode="same")
#         peaks[i] = np.nanargmax(convolved)
#         snrs[i] = convolved[peaks[i]]

#     best_idx = np.nanargmax(snrs)

#     return peaks[best_idx], widths[best_idx], snrs[best_idx]

## There are small changes in the code below compared to the original version from Zarif
## I think it's basically replace 0 with nan
# def get_ds_power(wfall, ds_factor, time=False):
#     """Get downsampled power for a baseband array or timestream."""
#     if not time:
#         pow2 = np.nansum(np.abs(wfall) ** 2, axis=1)
#         pow2_ds = pow2[:, : pow2.shape[1] // ds_factor * ds_factor]
#         pow2_ds = np.nansum(
#             pow2_ds.reshape(
#                 [pow2_ds.shape[0], pow2_ds.shape[1] // ds_factor, ds_factor]
#             ),
#             axis=-1,
#         )

#     else:
#         pow2 = np.nansum(np.abs(wfall) ** 2, axis=0)
#         pow2_ds = pow2[: pow2.size // ds_factor * ds_factor]
#         pow2_ds = np.nansum(
#             pow2_ds.reshape([pow2_ds.size // ds_factor, ds_factor]), axis=-1
#         )

#     return pow2_ds

# def get_dedispersed_clipped_bbdata(
#     data,
#     DM=None,
#     DM_range=10,
#     DM_step=0.01,
#     downsampling_factor=None,
#     pulse_lim=None,
#     diagnostic_plots=True,
#     full_output=True,
#     floor_level=0.1,
#     subtract_dc=True,
#     valid_time_range=None,
#     dedisperse=None,
#     smoothing_factor=None,
#     seed=0,
# ):
#     """
#     Dedisperse and clip a bbdata array.

#     Data is clipped to the valid region, gain normalized, dc offset corrected,
#     and dedispersed at full 2.56 us resolution.

#     Parameters
#     ----------
#     data : BBData object or eventid (int)
#         For a BBData
#     DM : float (optional)
#         DM to dedisperse to
#     DM_range : float (optional)
#         Perform additional DM refinement of DM +/- DM_range
#     DM_step : float (optional)
#        step size for DM refinement
#     downsampling_factor : integer (optional)
#         Downsampling factor (number of frames), automatically selected if not input
#     pulse_lim : list (optional)
#         time range in frames to select pulse region
#     diagnostic_plots : bool or string (optional)
#         Generate diagnostic plots or saves plots to file
#     full_output: bool (optional)
#         Return full output (True) or only baseband array (False)
#     floor_level : float (optional)
#         Level from the floor from which the width of a pulse is determined.
#         e.g. 0.5 is FWHM
#     subtract_dc : bool (optional)
#         Subtract off the complex mean per freq
#     valid_time_range : None or list of two numbers (optional)
#         Manually provide the time range for a baseband array
#     dedisperse : bool (optional)
#         Force whether or not to dedispersed the array
#     smoothing_factor : None or float (optional)
#        manually apply a smoothing factor for the gaussian filter
#     seed : float (optional)
#         Random number generator seed. must be >= 0

#     Returns
#     -------
#     (full output)
#     data_clipped : BBData object
#         BBData object.
#     bb_d : np.array(freq,beam,frames)
#         Coherently Dedispersed waterfall
#     valid_channels : np.array(freq)
#         Valid freq. channels
#     downsampling_factor : int
#         Downsampling factor used
#     signal_temp : float
#         Signal flux estimate.
#     noise_temp : float
#         Noise flux estimate.
#     lim : list
#         Valid limits of data
#     (else)
#     bb_d : np.array(freq,beam,frames)
#         Coherently Dedispersed waterfall
#     """
#     diagnostic_bool = False
#     if diagnostic_plots is not False:
#         diagnostic_bool = True  # only for get_snr to plot but not save

#     # 0) mask invalid values near the edge of the dump

#     sawtooth_inds = squarewave_pattern_index(
#         data["tiedbeam_power"][:], data.index_map["freq"]["centre"]
#     )
#     new_mask = np.zeros(data["tiedbeam_power"][:].shape)
#     for fi in range(new_mask.shape[0]):
#         new_mask[fi, :, sawtooth_inds[fi] :] = True

#     data["tiedbeam_baseband"][:] = np.where(
#         new_mask, np.nan, data["tiedbeam_baseband"][:]
#     )

#     # 1) run get_snr to find what valid_span_power_bins and
#     #    downsampling_factor and DM_clip is.

#     # ----------------- NOTE ---------------------
#     # power_temp is downsampled i.e. IS NOT in units of frames
#     # valid_span_power_bins IS in units of frames and the valid range
#     # of the full baseband dump power_ds_factor IS in units of frames
#     # --------------------------------------------

#     powerfactor = data["tiedbeam_power"].attrs["time_downsample_factor"]

#     if type(downsampling_factor) is int:
#         downsampling_factor = max(
#             np.round(downsampling_factor / powerfactor).astype(int), 1
#         )
#         downsampling_factor = int(downsampling_factor)
#         # get_snr does not like numpy ints

#     (
#         _,
#         _,
#         _,# power_temp,
#         _,
#         _,
#         valid_channels,
#         valid_span_power_bins,
#         DM_clip,
#         power_ds_factor,
#     ) = get_snr(
#         data,
#         diagnostic_plots=diagnostic_bool,
#         downsample=downsampling_factor,
#         DM=DM,
#         DM_range=DM_range,
#         DM_step=DM_step,
#         spectrum_lim=False,
#         return_full=True,
#         fill_missing_time=False,
#     )
#     print(f'Refined DM : {DM_clip} for event {data.attrs["event_id"]}')

#     # start_power_bins_temp, end_power_bins_temp = get_main_peak_lim(
#     #     power_temp,
#     #     floor_level=floor_level,
#     #     diagnostic_plots=False,
#     #     normalize_profile=True,
#     # )
#     # calculate start and end here to make sure the clipper does not lose the pulse.

#     # 2) find valid time range and clip the BBData object,
#     #    by looking at the corners of the "trapezoid" and
#     #    cutting the biggest rectangle out of it possible.
#     # If the lensing pipeline can't find a burst this is probably why;
#     # the burst is too close to the edge.
#     # bottom corners taken from get_snr
#     valid_span_unix_bottom = data["time0"]["ctime"][-1] + 2.56e-6 * np.array(
#         valid_span_power_bins
#     )

#     # top corners taken from data shape
#     dm_delay_sec = delay_across_the_band(
#         DM=DM_clip,
#         freq_high=data.index_map["freq"]["centre"][0],
#         freq_low=data.index_map["freq"]["centre"][-1],
#     )

#     valid_span_unix_top = (
#         data["time0"]["ctime"][0] + dm_delay_sec + np.array([0, data.ntime]) * 2.56e-6
#     )

#     valid_times = [
#         max(valid_span_unix_bottom[0], valid_span_unix_top[0]),
#         min(valid_span_unix_bottom[1], valid_span_unix_top[1]),
#     ]

#     # translate times to TOA and duration; run the clipper
#     toa_400 = np.mean(
#         valid_times
#     )  # calculate the center of the data within the valid region
#     duration = valid_times[1] - valid_times[0]  # calculate duration in seconds

#     # baseband data got conjugated at some point,
#     # but need to account for both files that are and aren't conjugated
#     # if there's a better way to identify for old and new files, please update
#     if dedisperse is None:
#         if ("conjugate_beamform" not in data["tiedbeam_baseband"].attrs) or (
#             not data["tiedbeam_baseband"].attrs["conjugate_beamform"]
#         ):
#             dedisperse = False
#         else:
#             dedisperse = True

#     # 3) Create a copy of BBData containing the clipped, dedispersed baseband array
#     data_clipped = clip(
#         data,
#         toa_400=toa_400,
#         duration=duration,
#         ref_freq=data.index_map["freq"]["centre"][-1],
#         dm=DM_clip,
#         inplace=False,
#         pad=True,
#     )

#     if dedisperse:
#         bb_d = coherent_dedisp(data_clipped, DM=DM_clip, time_shift=False)
#     else:
#         bb_d = data_clipped["tiedbeam_baseband"][:]
#     data_clipped["tiedbeam_baseband"].attrs["DM"] = DM_clip

#     # 4) Fix gains and correct dc offset
#     bb_d = np.where(np.isnan(bb_d), 0, bb_d)
#     weights0, offsets0 = get_weights(
#         np.abs(bb_d[:, 0]) ** 2,
#         f_id=data_clipped.index_map["freq"]["id"],
#         spectrum_lim=False,
#     )
#     weights1, offsets1 = get_weights(
#         np.abs(bb_d[:, 1]) ** 2,
#         f_id=data_clipped.index_map["freq"]["id"],
#         spectrum_lim=False,
#     )
#     bb_d[:, 0] /= np.sqrt(weights0)[:, None]
#     bb_d[:, 1] /= np.sqrt(weights1)[:, None]
#     dc_offset = np.mean(bb_d, axis=-1)
#     bb_d -= dc_offset[..., None]
#     bb_d[~valid_channels] = 0

#     data_clipped["tiedbeam_baseband"][:] = bb_d

#     tiedbeam_baseband_to_power(
#         data_clipped,
#         time_downsample_factor=powerfactor,
#         dedisperse=False,
#         dm=DM_clip,
#         time_shift=False,
#     )
#     # post process the data after clipping

#     # 5) downsample the clipped power and get valid range
#     power_ds_factor = downsampling_factor * powerfactor
#     power = get_ds_power(bb_d, power_ds_factor)

#     time_range_power_bins = get_valid_time_range(power)
#     power = power[..., time_range_power_bins[0] : time_range_power_bins[1]]
#     time_range_power_bins = np.array(time_range_power_bins) * power_ds_factor

#     if pulse_lim is None:
#         # Calculate pulse limits and convert from power bins back into frame bins.
#         start_power_bins, end_power_bins = get_main_peak_lim(
#             power,
#             floor_level=floor_level,
#             diagnostic_plots=False,
#             normalize_profile=True,
#         )
#         pulse_lim = [
#             start_power_bins * power_ds_factor,
#             (end_power_bins + 1) * power_ds_factor,
#         ]

#     if type(pulse_lim) is list:
#         pulse_lim = np.array(pulse_lim)

#     # 6) Truncate to valid range, useful for selecting a small region of the dump
#     if valid_time_range is None:
#         valid_time_range = time_range_power_bins

#     if pulse_lim[0] < valid_time_range[0] or pulse_lim[1] > valid_time_range[1]:
#         print(
#             f"WARNING: pulse limit = {pulse_lim} is outside"
#             f" valid time range = {valid_time_range}"
#         )

#     if (valid_time_range[1] - valid_time_range[0]) % 2 != 0:
#         valid_time_range[1] -= 1

#     bb_d = bb_d[..., valid_time_range[0] : valid_time_range[-1]]

#     flux = np.nansum(np.abs(bb_d[valid_channels, ...]) ** 2, axis=0)
#     noise_temp = np.nanmedian(flux, axis=-1)
#     signal_temp = np.nanmean(flux[:, pulse_lim[0] : pulse_lim[1]], axis=-1)

#     print(f"output bbd.shape = {bb_d.shape}")

#     _, bb_dfilled = fill_waterfall(data_clipped, matrix_in=bb_d, write=False)
#     bb_dfilled[np.where(bb_dfilled == 0)] = np.nan

#     print(f"output downsampling_factor = {power_ds_factor}")

#     if full_output:
#         return (
#             data_clipped,
#             bb_dfilled,
#             valid_channels,
#             power_ds_factor,
#             signal_temp,
#             noise_temp,
#             pulse_lim,
#         )
#     else:
#         return bb_dfilled
