import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.stats import trim_mean
from .mcc import compute_rdc

def myMSE(Rr):
    d = np.min(Rr.shape)
    Rr = np.abs(Rr)
    D = np.zeros((d, d))
    for ii in range(d):
        for jj in range(d):
            tmp = - Rr[ii, :]
            tmp[jj] = tmp[jj] + 1
            D[ii, jj] = np.sum(tmp**2)
    r, c = linear_sum_assignment(D)
    mse = 2 - (2/d)*np.trace(Rr[:, c])

    Rr_sorted = Rr[r, c]
    mcc = np.array([trim_mean(Rr_sorted, 0.25), np.mean(Rr_sorted), np.min(Rr_sorted)])

    return mse, mcc

def MMSE(Y,Yh,u,S=None):
    n_segment = u.shape[1]
    n_source = Y.shape[1]
    n_modality = Y.shape[2]

    if S is None:
        S = [np.eye(n_source) for m in range(n_modality)]
    
    Rr_ps = np.zeros((n_segment,n_source,n_source))
    Rr_pm = np.zeros((n_modality,n_source,n_source))
    Rr_pmps = np.zeros((n_modality,n_segment,n_source,n_source))
    mse_ps, mcc_ps = np.zeros(n_segment), np.zeros((n_segment,3))
    mse_pm, mcc_pm = np.zeros(n_modality), np.zeros((n_modality,3))
    mse_pmps, mcc_pmps = np.zeros((n_modality,n_segment)), np.zeros((n_modality,n_segment,3))

    Ryyh = compute_rdc(Y, Yh, u)

    # Compute metrics per modality, per segment
    for seg in range(n_segment):
        for mm in range(n_modality):
            mk = np.split(S[mm] == 1, n_source, axis=0)
            for kr in range(n_source): # rows
                mkr = mk[kr]
                for kc in range(n_source): # columns
                    mkc = mk[kc]
                    Rr_pmps[mm,seg,kr,kc] = np.abs(Ryyh[mm][seg][mkr.T @ mkc]).max()
            mse_pmps[mm,seg], mcc_pmps[mm,seg] = myMSE(Rr_pmps[mm,seg])

    # Compute metrics per segment, aggregated over modalities
    for seg in range(n_segment):
        for mm in range(n_modality):
            mk = np.split(S[mm] == 1, n_source, axis=0)
            for kr in range(n_source): # rows
                mkr = mk[kr]
                for kc in range(n_source): # columns
                    mkc = mk[kc]
                    Rr_ps[seg,kr,kc] = np.abs(Ryyh[mm][seg][mkr.T @ mkc]).max()
        mse_ps[seg], mcc_ps[seg] = myMSE(Rr_ps[seg])

    # Compute metrics per modality, aggregated over segments
    for mm in range(n_modality):
        mk = np.split(S[mm] == 1, n_source, axis=0)
        for seg in range(n_segment):
            for kr in range(n_source): # rows
                mkr = mk[kr]
                for kc in range(n_source): # columns
                    mkc = mk[kc]
                    Rr_pm[mm,kr,kc] = np.abs(Ryyh[mm][seg][mkr.T @ mkc]).max()
        mse_pm[mm], mcc_pm[mm] = myMSE(Rr_pm[mm])

    Rr = np.max(np.abs(Rr_ps), axis=0)
    mse, mcc = myMSE(Rr)

    return mse_ps, mcc_ps, Rr_ps, mse_pm, mcc_pm, Rr_pm, mse, mcc, Rr
