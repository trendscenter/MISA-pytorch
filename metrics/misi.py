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

def [out, WAr] = MISI(W,A,S):

    WA = cellfun(@(w,a) w*a, W, A, 'Un', 0)
    WA = blkdiag(WA{:})

    K = size(S{1},1)
    Stmp = [S{:}]
    idx = []
    cps = []
    for kk = 1:K
        tidx = find(Stmp(kk,:))
        idx = [idx tidx]
        cps(kk) = length(tidx)

    WA = WA(idx,idx)

    # cps = sum(Stmp,2) # This is bugged... crashes the server for large number of datasets...

    # figure, imagesc(WA)
    WAr = mat2cell(WA,cps,cps)
    #WAr = cell2mat(cellfun(@(wa) mean(abs(wa(:))), WAr, 'Un', 0))
    WAr = cell2mat(cellfun(@(wa) sum(abs(wa(:))), WAr, 'Un', 0))

    out = myISI(WAr)