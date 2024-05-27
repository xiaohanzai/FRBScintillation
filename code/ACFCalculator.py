import numpy as np

def calc_dspec(spec_on_, spec_offs_, spec_smooth_on_, spec_smooth_offs_):
    mean_spec_off_ = np.nanmean(spec_offs_, axis=1)
    spec_smooth_off_ = np.nanmean(spec_smooth_offs_, axis=1)
    spec_smooth_ = spec_smooth_on_ - spec_smooth_off_
    dspec_on_ = (spec_on_-mean_spec_off_)/spec_smooth_ - 1
    dspec_offs_ = (spec_offs_-mean_spec_off_[:,np.newaxis,:])/spec_smooth_[:,np.newaxis,:]
    return dspec_on_, dspec_offs_

def f_corr(xs, ys=None, xweights=None, yweights=None):
    if ys is None:
        ys = xs
    if xweights is None:
        xweights = np.ones(xs.shape[-1])
    if yweights is None:
        yweights = np.ones(ys.shape[-1])

    if len(xs.shape) > 1:
        xweights = xweights[np.newaxis,:]
        yweights = yweights[np.newaxis,:]
    wxs = xs * xweights
    wys = ys * yweights
    if len(xs.shape) == 1:
        corr = np.array([np.nansum(wxs[i:] * wys[:len(xs)-i]) for i in range(len(xs))])
        norm = np.array([np.nansum(xweights[i:] * yweights[:len(xs)-i]) for i in range(len(xs))])
    else:
        corr = np.array([np.nansum(wxs[:,i:] * wys[:,:xs.shape[1]-i], axis=1) for i in range(xs.shape[1])]).T
        norm = np.array([np.nansum(xweights[:,i:] * yweights[:,:xs.shape[1]-i], axis=1) for i in range(xs.shape[1])]).T
    # TODO: not sure if this the best protection mechanism
    flag = False
    if norm.max() / xs.shape[-1] < 1e-4:
        flag = True
    return corr / norm, flag

class ACFCalculator():
    def __init__(self, dspec_on_, dspec_offs_, freqs):
        '''
        Input dspec = spec / spec_smooth - 1
        TODO: for now, input dspec_offs_ as spec_offs_ / spec_smooth_on_[:,np.newaxis,:]
        '''
        self.dspec_on_ = dspec_on_.copy()
        self.dspec_offs_ = dspec_offs_.copy()

        self.chan_weights = 1/np.nanvar(np.sum(self.dspec_offs_, axis=0), axis=0)
        self.chan_weights[np.isinf(self.chan_weights)] = np.nan

        self.freqs = freqs.copy()
        self.freq_min = freqs.min()
        self.freq_max = freqs.max()
        self.n_freq = len(freqs) # should equal dspec_on_.shape[-1]

    def calc_acf_band(self, time_slc_i, time_slc_j, freq_min=None, freq_max=None, i_start=None, i_end=None):
        '''
        Calculates autocorrelation function using channels i_start:i_end, or frequencies freq_min:freq_max.
        Must input one of these two.
        Cross-correlate time slices time_slc_i and time_slc_j to get the "autocorrelation".
        Parameters
        ----------
        time_slc_i : int
            The index of the first time slice to correlate.
        time_slc_j : int
            The index of the second time slice to correlate.
        freq_min : float
            The minimum frequency in MHz.
        freq_max : float
            The maximum frequency in MHz.
        i_start : int
            The index of the first channel to use.
        i_end : int
            The index of the last channel to use.
        Returns
        ----------
        acf_on : array_like
            The autocorrelation function of the on-pulse region.  Shape is (i_end-i_start,).
        acf_offs : array_like
            The autocorrelation functions of the N off-pulse region.  Not normalized.  Shape is (N, i_end-i_start).
        '''
        dfreq = (self.freq_max - self.freq_min)/self.n_freq
        # get indices of frequencies
        if i_start is None:
            inds = np.where((self.freqs >= freq_min) & (self.freqs <= freq_max))[0]
            i_start = inds[0]
            i_end = inds[-1] + 1

        ws = self.chan_weights[i_start:i_end]

        acf_on, flag = f_corr(self.dspec_on_[time_slc_i, i_start:i_end], self.dspec_on_[time_slc_j, i_start:i_end],
                    xweights=ws, yweights=ws)
        acf_offs, _ = f_corr(self.dspec_offs_[time_slc_i, :, i_start:i_end], self.dspec_offs_[time_slc_j, :, i_start:i_end],
                    xweights=ws, yweights=ws)
        nus = np.arange(len(acf_on)) * dfreq
        if flag:
            print('ACF calc warning: norm is small, freqs =', freq_min, '-' , freq_max)
        return acf_on, acf_offs, nus, flag
