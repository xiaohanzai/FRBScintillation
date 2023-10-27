import matplotlib.pyplot as plt
import numpy as np
from baseband_analysis.core.sampling import scrunch
from ACFCalculator import calc_dspec
from ACFFitter import f_lorenzian

def plot_on_range(power, filter, on_range, ax=None):
    '''
    Calculate a time series from power, plot it, and plot the on-pulse range.
    The input filter is (the set of) boxcar filter(s) that define the on-pulse range.
    The code will also calculate a smoothed (downsampled) version of the time series for better visual inspection.
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ind_l, ind_r = on_range
    # calculate noise
    noise = np.nanmean(power[:,:ind_l])
    # the time series
    flux = np.nanmean(power, axis=0) - noise
    ax.plot(flux)
    # the smoothed time series
    ds_factor = 16
    flux_smoothed = np.nanmean(scrunch(power, tscrunch=ds_factor, fscrunch=1), axis=0) - noise
    ax.plot(np.arange(len(flux_smoothed)) * ds_factor + ds_factor/2, flux_smoothed)
    # plot the filter
    flux_filt = np.zeros(power.shape[-1])
    flux_filt[ind_l:ind_r] = filter
    flux_filt *= flux.max() / flux_filt.max()
    ax.plot(flux_filt)
    # ax.hlines(2*noise, 0, len(flux_filt), colors='k', linestyles='dashed')
    # plot the on-pulse range
    ax.vlines([ind_l, ind_r], 0, np.max(flux_filt), colors='r', linestyles='dotted')
    ax.set_xlim([ind_l-5000, ind_r+5000])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Flux')
    return ax

def plot_waterfall(power, on_range, ax=None, freq_max=800., freq_min=400.):
    '''
    Plot the waterfall and show the on-pulse range.
    The plot will be zoomed in on the on-pulse range.
    TODO: maybe make the vmin, vmax, and xlim adjustable.
    '''
    vmin, vmax = np.nanpercentile(power, [5, 95])
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(power, aspect='auto', origin='lower', extent=[0, power.shape[-1], freq_min, freq_max],
            vmin=vmin, vmax=vmax)
    ax.vlines(on_range, freq_min, freq_max, colors='r', linestyles='dashed')
    ax.set_xlim([on_range[0]-5000, on_range[1]+5000])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Frequency (MHz)')
    ax.invert_yaxis()
    return ax

def plot_specs(spec_on_, spec_offs_, spec_smooth_on_, spec_smooth_offs_, freqs):
    '''
    Plot the on and off-pulse spectra and the fitted smooth spectra.
    Also plot dspec_on and dspec_off in the 3rd subplot.
    '''
    fig, axes = plt.subplots(figsize=(15,5), nrows=1, ncols=3)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.98, wspace=0.3)
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
    dspec_on_, dspec_offs_ = calc_dspec(spec_on_, spec_offs_, spec_smooth_on_, spec_smooth_offs_)
    axes[2].plot(freqs, dspec_on_[ind0], label='on-pulse')
    for ind in inds_off:
        axes[2].plot(freqs, dspec_offs_[ind0][ind], alpha=0.2, label='off-pulse')
    axes[2].set_ylabel('spec / spec_smooth - 1')
    axes[2].set_ylim(-8, 8)

    # labels and legend
    for ax in axes:
        ax.set_xlabel('freq (MHz)')
    axes[0].set_ylabel('spectra')
    axes[0].legend()

    return fig

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

def plot_acf(nus, acf_on, m=None, nu_dc=None, y_offset=0., label=None, acf_offs=None, ax=None, alpha=0.5,
            logx=False, xlim=(1e-6,2), flag=None, plot_log_rebin=True, plot0s_std_threshold=1., **kwargs):
    '''
    Plot only one ACF.
    If m and nu_dc are not None, plot the fitted ACF as well.
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if acf_offs is not None:
        # randomly choose 3 off-pulse ACFs to plot
        inds_off = np.random.choice(acf_offs.shape[0], 3, replace=False)
        cs = ['b', 'g', 'm']

    # decide whether to plot 0s
    if flag is None:
        flag = np.std(acf_on[nus > 1.]) > plot0s_std_threshold

    # plot
    ax.plot(nus, 0*nus+y_offset, 'k--')
    # plot on-pulse ACF and fitted result
    l = ax.plot(nus, acf_on * (~flag) + y_offset, alpha=alpha, label=label)[0]
    c = l.get_color()
    if plot_log_rebin:
        # log binning
        xs, ys = log_rebin(nus, acf_on * (~flag), 0.02, 1)
        ax.plot(xs, ys+y_offset, 'o-', color=c)
    if m is not None and nu_dc is not None:
        ax.plot(nus, f_lorenzian(nus, m, nu_dc)+y_offset, 'r')
    # plot off-pulse ACFs
    if acf_offs is not None:
        for c, ind in zip(cs, inds_off):
            ax.plot(nus, acf_offs[ind] * (~flag) + y_offset, color=c, alpha=0.3)

    ax.set_xlabel(r'$\Delta\nu$ (MHz)')
    ax.set_ylabel('ACF')
    ax.set_ylim(y_offset-0.02, y_offset+1)
    if logx:
        ax.set_xscale('log')
    if label is not None:
        ax.legend(loc=1)
    ax.set_xlim(xlim)
    return ax

def plot_acfs(nus, acf_on_, freq_bins=None, acf_offs_=None, ms=None, nu_dcs=None, flags=None,
            ax=None, **kwargs):
    '''
    Plot all ACFs calculated for the frequency ranges given by freq_bins.
    If ms and nu_dcs are not None, plot the fitted ACFs as well.
    flags is a list of booleans, indicating whether the norm is too small so we plot 0s instead.
    If flags is not given, decide whether to plot 0s based on the std of the ACFs.
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for i in range(len(acf_on_)):
        m = nu_dc = flag = None
        if ms is not None:
            m, nu_dc = ms[i], nu_dcs[i]
        acf_offs = None
        if acf_offs_ is not None:
            acf_offs = acf_offs_[i]
        if flags is not None:
            flag = flags[i]
        label = None
        if freq_bins is not None:
            label = f'{freq_bins[i][0]}-{freq_bins[i][1]} MHz'
        plot_acf(nus, acf_on_[i], m=m, nu_dc=nu_dc, y_offset=i, label=label,
                acf_offs=acf_offs, ax=ax, flag=flag, **kwargs)
    ax.set_ylim([-0.5, len(acf_on_)+0.5])
    return ax

def plot_acfs_two_subplots(nus, acf_on_, freq_bins=None, acf_offs_=None, ms=None, nu_dcs=None, flags=None,
            xmax=2, **kwargs):
    '''
    Plot ACFs in two subplots, one with linear x-axis and one with log x-axis.
    '''
    fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3)
    plot_acfs(nus, acf_on_, freq_bins=freq_bins, acf_offs_=acf_offs_, ms=ms, nu_dcs=nu_dcs, flags=flags,
            xlim=(1e-6,xmax), ax=axes[0], logx=False, **kwargs)
    plot_acfs(nus, acf_on_, freq_bins=freq_bins, acf_offs_=acf_offs_, ms=ms, nu_dcs=nu_dcs, flags=flags,
            xlim=(None,xmax), ax=axes[1], logx=True, **kwargs)
    return fig
