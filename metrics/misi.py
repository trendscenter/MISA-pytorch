import numpy as np

def myISI(WA):
    N = WA.shape[0]
    WA = np.abs(WA)
    ISI = 0.
    ISI = ISI + np.sum(np.sum(WA,axis=1)/np.max(WA,axis=1) - 1)
    # np.max(WA,axis=1)
    ISI = ISI + np.sum(np.sum(WA,axis=0)/np.max(WA,axis=0) - 1)
    # np.max(WA,axis=0)
    ISI = ISI/(2*N*(N-1));
    return ISI

def MISI(W,A,S):

    WA = [W[cc] @ A[cc] for cc in len(W)]
    
    M = len(S)
    K = S[0].shape[0]

    WAr = np.zeros((K,K))

    for mm in range(M):
        mk = np.split(S[mm] == 1, K, axis=0)
        for kr in range(K):        # rows
            mkr = mk[kr]
            for kc in range(K):    # columns
                mkc = mk[kc]
                WAr[kr,kc] += np.sum(WA[mm][mkr,mkc])
  
    out = myISI(WAr)

    return out, WAr