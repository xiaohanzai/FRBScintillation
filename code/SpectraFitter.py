import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline
import scipy.optimize

def fit_spline(spec, num_splines=50, k=3):
    ws = np.isnan(spec)
    xs = np.arange(len(spec))[~ws]
    ts = xs[1:-1][::(len(xs)-2)//num_splines]
    ts = np.r_[(xs[0],)*(k+1),
            ts,
            (xs[-1],)*(k+1)]
    ys = spec[~ws]
    spline = make_lsq_spline(xs, ys, ts, k=k)
    spec_smooth = spec*0.
    spec_smooth[~ws] = spline(xs)
    return spec_smooth

def f_powerlaw_spec(freq, coeffs):
    '''
    Power-law spectrum as a function of log(freq):
    spec = exp( coeff[0] + coeff[1]*log(freq) + coeff[2]*log(freq)**2 + ... )
    '''
    l_freq_norm = np.log(freq / 750.)
    coeffs = coeffs.reshape(1,-1)
    exponent = np.sum(l_freq_norm.reshape(-1,1)**np.arange(0,coeffs.size,1) * coeffs, axis=1)
    return np.exp(exponent)

def fit_powerlaw(spec, freqs, mask_chans):
    spec = spec.copy()
    spec[mask_chans] = 0.
    weights = 1. - mask_chans
    residuals = lambda coeffs: (spec - f_powerlaw_spec(freqs, coeffs)) * weights

    coeff0 = np.log(np.mean(spec[~mask_chans & (freqs > 700.) & (freqs < 800.)]).clip(1e-6))
    coeffs, cov, info, msg, ierr = scipy.optimize.leastsq(
            residuals,
            [coeff0, -2, 0],
            epsfcn=0.001,
            full_output=True,
            xtol=0.0001
            )
    spec_smooth = f_powerlaw_spec(freqs, coeffs)
    return spec_smooth, coeffs

class SpectraFitter():
    def __init__(self, spec_on_, spec_offs_, freqs):
        '''
        The 0th dimension of spec_on_ and spec_offs_ corresponds to the number of time slices.
        The next dimension of spec_offs_ is the number of randomly sampled off-pulse regions.
        The last dimension of spec_on_ and spec_offs_ is the number of frequency channels.
        '''
        self.spec_on_ = spec_on_.copy() # shape (# of time slices, # of freq)
        self.spec_offs_ = spec_offs_.copy() # shape (# of time slices, N, # of freq)

        self.freqs = freqs.copy()

    def clean_rfi_3sigma(self):
        '''
        Cleans RFI by removing points that are 3 sigma away from the mean.
        '''
        spec_offs = self.spec_offs_.sum(axis=0) # full-length off-pulse regions
        med = np.nanmedian(spec_offs) # median flux, one number
        sigma = np.nanstd(spec_offs) # standard deviation, one number
        spec_off = np.nanmean(spec_offs, axis=0) # average off-pulse spectrum
        ii = np.abs(spec_off - med) > 3*sigma
        self.spec_on_[:,ii] = np.nan
        self.spec_offs_[:,:,ii] = np.nan

    def fit_smooth(self, use_powerlaw=False):
        '''
        Fit a smooth spectrum for spec_on and each of spec_offs.
        If use_powerlaw is True, then fit a power-law spectrum.  Otherwise fit spline.
        '''
        mask_chans = np.isnan(self.spec_on_[0]) | (self.spec_on_[0] == 0)

        spec_on_ = self.spec_on_*1.
        spec_offs_ = self.spec_offs_*1.
        def f_fit(spec, *args, **kwargs):
            spec_smooth = fit_spline(spec)
            return spec_smooth
        if use_powerlaw:
            tmp = np.nanmean(self.spec_offs_, axis=1)
            spec_on_ -= tmp
            spec_offs_ -= tmp[:, np.newaxis, :]
            def f_fit(spec, print_coeffs=False):
                spec_smooth, coeffs = fit_powerlaw(spec, self.freqs, mask_chans)
                if print_coeffs:
                    print(coeffs)
                return spec_smooth

        # fit smooth spectrum for each of the time-sliced on-pulse and off-pulse regions
        spec_smooth_on_ = spec_on_*0.
        spec_smooth_offs_ = spec_offs_*0.
        for i in range(spec_on_.shape[0]):
            spec_smooth_on_[i] = f_fit(spec_on_[i], print_coeffs=True)
            for j in range(spec_offs_.shape[1]):
                spec_smooth_offs_[i,j] = f_fit(spec_offs_[i,j])

        return spec_smooth_on_, spec_smooth_offs_
