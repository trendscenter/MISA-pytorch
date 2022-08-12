import numpy as np

def myISI(WAr):
    N = WAr.shape[0]
    WAr = np.abs(WAr)
    ISI = 0.
    ISI += np.sum(np.sum(WAr,axis=1)/np.max(WAr,axis=1) - 1)
    # np.max(WAr,axis=1)
    ISI += np.sum(np.sum(WAr,axis=0)/np.max(WAr,axis=0) - 1)
    # np.max(WAr,axis=0)
    ISI = ISI/(2*N*(N-1))
    return ISI

def MISI(W,A,S):

    WA = [W[cc].T @ A[cc].T for cc in range(len(W))]
    
    M = len(S)
    K = S[0].shape[0]

    WAr = np.zeros((K,K))

    for mm in range(M):
        mk = np.split(S[mm] == 1, K, axis=0)
        for kr in range(K):        # rows
            mkr = mk[kr]
            for kc in range(K):    # columns
                mkc = mk[kc]
                WAr[kr,kc] += np.abs(WA[mm][mkr.T @ mkc]).sum()
  
    out = myISI(WAr)

    return out, WAr