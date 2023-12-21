import fitburst as fb
import json
from copy import deepcopy
import numpy as np
from chime_frb_constants import FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ
from baseband_analysis.core.sampling import scrunch
from scipy.interpolate import RegularGridInterpolator

def get_fb_model(fitburst_json, ds_factor=16):
    """
    fitburst_json is the file name of the output fitburst results json file.
    ds_factor is the downsampling factor used in the fitburst fitting.
    """
    data = json.load(open(fitburst_json, "r"))
    params = data["model_parameters"]
    numtime = data['fit_statistics']['num_time']
    numfreq = data['fit_statistics']['num_freq']
    num_components = len(params["amplitude"])
    new_params = deepcopy(params)
    freqs = np.linspace(FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, num=numfreq)
    # TODO: I didn't entirely believe in the way the times array was set originally so I changed it
    times = np.arange(numtime) * ds_factor * 2.56e-6
    model_obj = fb.analysis.model.SpectrumModeler(
        freqs,
        times,
        dm_incoherent = params["dm"][0],
        factor_freq_upsample=1,
        factor_time_upsample=1,
        is_dedispersed=True,
        verbose=True,
        num_components=num_components,
    )
    model_obj.update_parameters(new_params)
    model = model_obj.compute_model()

    return model, times

def create_full_fb_filter(ww, model, ds_factor=16):
    '''
    Create the fitburst filter that matches the size of the input waterfall.
    Match using the peak location after downsampling.
    '''
    power = scrunch(np.sum(np.abs(ww)**2, axis=1), tscrunch=ds_factor, fscrunch=1)
    ind = np.argmax(np.nanmean(power, axis=0))
    ind_model = np.argmax(np.mean(model, axis=0))
    # obtain a shited model first with time axis size matching that of of the downsampled waterfall
    model_shifted = np.zeros((model.shape[0], power.shape[1]))
    if ind >= ind_model:
        model_shifted[:,ind-ind_model:ind] = model[:,:ind_model]
    else:
        model_shifted[:,:ind] = model[:,ind_model-ind:ind_model]
    if model_shifted.shape[1] - ind >= model.shape[1] - ind_model:
        model_shifted[:,ind:ind+model.shape[1]-ind_model] = model[:,ind_model:]
    else:
        model_shifted[:,ind:] = model[:,ind_model:ind_model+model_shifted.shape[1]-ind]
    # interpolate the shifted model to match the full size of the waterfall
    freqs = np.linspace(0, ww.shape[0], model.shape[0]) # TODO: this is probably not totally correct
    times = np.arange(power.shape[1]) * ds_factor + ds_factor/2
    f = RegularGridInterpolator((freqs, times), model_shifted, fill_value=0., bounds_error=False)
    times_full, freqs_full = np.meshgrid(np.arange(ww.shape[-1]), np.arange(ww.shape[0]))
    model_full = f((freqs_full, times_full))
    return model_full
