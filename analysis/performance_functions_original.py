"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from typing import Dict, List, Tuple

def nse(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    SSerr = np.mean(np.square(df["qsim"][idex] - df["qobs"][idex]))
    SStot = np.mean(np.square(df["qobs"][idex] - np.mean(df["qobs"][idex])))
    return 1 - SSerr / SStot
    
def alpha_nse(df):
    # Ratio of std of observed and simulated flow
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df["qsim"][idex]
    qobs = df["qobs"][idex]
    # Calculate the standard deviations
    std_qsim = np.std(qsim)
    std_qobs = np.std(qobs)
    # Calculate the ratio of the standard deviations
    alpha = std_qsim / std_qobs
    return alpha
    
def beta_nse(df):
    #  Ratio of the means of observed and simulated flow
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df["qsim"][idex]
    qobs = df["qobs"][idex]
    # Calculate the means
    mean_qsim = np.mean(qsim)
    mean_qobs = np.mean(qobs)
    # Calculate the ratio of the means
    beta = (mean_qsim - mean_qobs)/ np.std(qobs)
    return beta

    
def kge(df): # yhwang 20240616
    # Select indices where both qsim and qobs are non-negative
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df["qsim"][idex]
    qobs = df["qobs"][idex]
    
    # Calculate correlation coefficient (r)
    r = np.corrcoef(qsim, qobs)[0, 1]
    
    # Calculate variability ratio
    alpha = np.std(qsim) / np.std(qobs)
    
    # Calculate bias ratio (ß)
    beta = np.mean(qsim) / np.mean(qobs)
    
    # Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge, r, alpha, beta

def get_quant(df, quant):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].quantile(quant)
    obs = df["qobs"][idex].quantile(quant)
    return obs, sim


def bias(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].mean()
    obs = df["qobs"][idex].mean()
    return (obs - sim) / obs * 100


def stdev_rat(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].std()
    obs = df["qobs"][idex].std()
    return sim / obs


def zero_freq(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = (df["qsim"][idex] == 0).astype(int).sum()
    obs = (df["qobs"][idex] == 0).astype(int).sum()
    return obs, sim


def flow_duration_curve(df):
    obs33, sim33 = get_quant(df, 0.33)
    obs66, sim66 = get_quant(df, 0.66)
    sim = (np.log(sim33) - np.log(sim66)) / (0.66 - 0.33)
    obs = (np.log(obs33) - np.log(obs66)) / (0.66 - 0.33)
    return obs, sim


def baseflow_index(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsQ = df["qobs"][idex].values
    simQ = df["qsim"][idex].values
    nTimes = len(obsQ)

    obsQD = np.full(nTimes, np.nan)
    simQD = np.full(nTimes, np.nan)
    obsQD[0] = obsQ[0]
    simQD[0] = simQ[0]

    c = 0.925
    for t in range(1, nTimes):
        obsQD[t] = c * obsQD[t - 1] + (1 + c) / 2 * (obsQ[t] - obsQ[t - 1])
        simQD[t] = c * simQD[t - 1] + (1 + c) / 2 * (simQ[t] - simQ[t - 1])

    obsQB = obsQ - obsQD
    simQB = simQ - simQD

    obs = np.mean(np.divide(obsQB[1:], obsQ[1:]))
    sim = np.mean(np.divide(simQB[1:], simQ[1:]))
    return obs, sim


def high_flows(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsMedian = df["qobs"][idex].median()
    obsFreq = len(df["qobs"][idex].index[(df["qobs"][idex] >= 9 * obsMedian)].tolist())
    simMedian = df["qsim"][idex].median()
    simFreq = len(df["qsim"][idex].index[(df["qsim"][idex] >= 9 * simMedian)].tolist())
    return obsFreq, simFreq


def low_flows(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsMedian = df["qobs"][idex].median()
    obsFreq = len(df["qobs"][idex].index[(df["qobs"][idex] <= 0.2 * obsMedian)].tolist())
    simMedian = df["qsim"][idex].median()
    simFreq = len(df["qsim"][idex].index[(df["qsim"][idex] <= 0.2 * simMedian)].tolist())
    return obsFreq, simFreq
    
def FHV(df, percentile_value):
    # percentile: 2 means top 2%, 10 means top 10%. 
    # Select indices where both qsim and qobs are non-negative
    idx = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df.loc[idx, "qsim"]
    qobs = df.loc[idx, "qobs"]
    
    # Calculate the 2% exceedance threshold
    qsim_threshold = np.percentile(qsim, 100 - percentile_value)
    qobs_threshold = np.percentile(qobs, 100 - percentile_value)
    
    # Identify high flow values
    qsim_high_flows = qsim[qsim > qsim_threshold]
    qobs_high_flows = qobs[qobs > qobs_threshold]
    
    # Calculate mean of high flow values
    qsim_high_mean = qsim_high_flows.mean()
    qobs_high_mean = qobs_high_flows.mean()
    
    # Calculate FHV
    fhv = ((qsim_high_mean - qobs_high_mean) / qobs_high_mean) 
    
    return fhv * 100
    
'''
def FLV(df, l: float = 0.3) -> float:
    idx = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df.loc[idx, "qsim"]
    qobs = df.loc[idx, "qobs"]

    qsim_sorted = np.sort(qsim)
    qobs_sorted = np.sort(qobs)
    
    # Calculate the number of flow values within the 70-100% exceedance probability range
    L = int(0.3 * len(qsim_sorted))
    
    # Identify the low flow values for the given exceedance range
    qsim_low_flows = qsim_sorted[-L:]
    qobs_low_flows = qobs_sorted[-L:]

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    qsim_low_flows[qsim_low_flows <= 0] = 1e-6
    qobs_low_flows[qobs_low_flows == 0] = 1e-6

    # transform values to log scale
    qobs_low_flows = np.log(qobs_low_flows)
    qsim_low_flows = np.log(qsim_low_flows)

    # calculate flv part by part
    qsl = np.sum(qsim_low_flows - qsim_low_flows.min())
    qol = np.sum(qobs_low_flows - qobs_low_flows.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100
'''

def _get_fdc(da: DataArray) -> np.ndarray:
    return da.sortby(da, ascending=False).values

def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError("Metrics only defined for time series (1d or 2d with second dimension 1)")

def _mask_valid(obs: DataArray, sim: DataArray) -> Tuple[DataArray, DataArray]:
    # mask of invalid entries. NaNs in simulations can happen during validation/testing
    idx = (~sim.isnull()) & (~obs.isnull())

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim

def FLV(df, l: float = 0.3) -> float:
    idx = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df.loc[idx, "qsim"]
    qobs = df.loc[idx, "qobs"]
    
    obs = xr.DataArray(qobs, dims=['time'])
    sim = xr.DataArray(qsim, dims=['time'])

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 1:
        return np.nan

    if (l <= 0) or (l >= 1):
        raise ValueError("l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    obs = obs[-np.round(l * len(obs)).astype(int):]
    sim = sim[-np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs)
    sim = np.log(sim)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100
    
def mass_balance(df):
    idx = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    qsim = df.loc[idx, "qsim"]
    qobs = df.loc[idx, "qobs"]
    massbias_total = abs(qsim.sum()-qobs.sum())/qobs.sum()
    if qsim.sum()-qobs.sum() > 0:
        massbias_pos = (qsim.sum()-qobs.sum()) / qobs.sum()
        massbias_neg = 0
    else:
        massbias_neg =  - (qsim.sum()-qobs.sum()) / qobs.sum()
        massbias_pos = 0
    return massbias_total * 100, massbias_pos * 100, massbias_neg * 100


