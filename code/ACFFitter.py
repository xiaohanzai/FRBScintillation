import numpy as np
from scipy.optimize import curve_fit

f_cauchy = lambda dnu, m, nu_dc: m / (dnu**2/nu_dc**2 + 1.)

def hmhw_cauchy(nus, acf, ebars=None, exclude_zero=True):
    """
    Fits the half-maximum half-width of the autocorrelation function to a Cauchy distribution.
    """
    start = 0
    if exclude_zero:
        start = 1
    ii = ~np.isnan(acf[start:])
    sigma = None
    if ebars is not None:
        sigma = ebars[start:][ii]
    popt, pcov = curve_fit(f_cauchy, nus[start:][ii], acf[start:][ii], sigma=sigma, bounds=([0, 0], [np.max(acf[:1]), np.inf]), check_finite=False)
    return popt, pcov

class ACFFitter():
    def __init__(self, acf_on, acf_offs, nus, ebar_scale_fac=1.):
        self.acf_on = acf_on.copy()
        self.acf_offs = acf_offs.copy()

        self.ebar_scale_fac = ebar_scale_fac

        self.nus = nus

    def fit_nu_dc(self, calc_ebars=True, exclude_zero=True):
        offs_std = None
        if calc_ebars:
            offs_std = np.nanstd(self.acf_offs, axis=0)
        popt, pcov = hmhw_cauchy(self.nus, self.acf_on, ebars=offs_std*self.ebar_scale_fac, exclude_zero=exclude_zero)
        perr = np.sqrt(np.diag(pcov))
        m = popt[0]
        m_err = perr[0]
        nu_dc = popt[1]
        nu_dc_err = perr[1]
        return m, m_err, nu_dc, nu_dc_err