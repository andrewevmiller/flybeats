import numpy as np, sys
cls=['kick','snare','hat_closed','hat_open','tom_low','tom_mid','tom_high','crash']
def r(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-8 and b.std()>1e-8 else np.nan
for f in sys.argv[1:]:
    d=np.load(f); H,V,Y=d['H'],d['V'],d['Y']
    C,T,K=Y.shape
    print(f)
    for c,n in enumerate(cls):
        y=Y[:,:,c]
        pk=(y>0.95)&(y>=np.roll(y,1,1))&(y>np.roll(y,-1,1))   # one step per hit
        cid=np.repeat(np.arange(C)[:,None],T,1)[pk]; h=H[:,:,c][pk]; t=V[:,:,c][pk]
        if len(t)<30: print(f"  {n:<11} {len(t)} hits, too few"); continue
        u=np.unique(cid); idx={k:np.where(cid==k)[0] for k in u}; rng=np.random.default_rng(0); bs=[]
        for _ in range(2000):
            p=np.concatenate([idx[k] for k in rng.choice(u,len(u))]); bs.append(r(h[p],t[p]))
        bs=np.array(bs); bs=bs[~np.isnan(bs)]
        lo,hi=np.percentile(bs,[2.5,97.5])
        v='RIGHT' if lo>0 else 'BACKWARDS' if hi<0 else "can't tell"
        print(f"  {n:<11} {len(t):>4} hits / {len(u):>3} clips  r {r(h,t):+.2f} [{lo:+.2f},{hi:+.2f}] {v}")
