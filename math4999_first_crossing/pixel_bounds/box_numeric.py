"""Conservative box-support arithmetic. No data is read by this module."""
from fractions import Fraction as F
import math
import numpy as np

INF=float('inf'); ETA=F(1,2**1074); U=F(1,2**53)

def fraction(x):
    x=float(x)
    if not math.isfinite(x): raise ValueError('nonfinite scalar')
    return F.from_float(x)

def finite(*xs):
    if any(not np.isfinite(x).all() for x in xs): raise ValueError('nonfinite interval arithmetic')

def w_interval(x,x0,mu):
    x=np.asarray(x); x0=np.asarray(x0); mu=np.asarray(mu)
    if x.shape!=x0.shape or x.shape!=mu.shape or x.dtype!=np.float32 or x0.dtype!=np.float32 or mu.dtype!=np.float64: raise ValueError('invalid shape or dtype')
    finite(x,x0,mu)
    if np.any((x<0)|(x>1)|(x0<0)|(x0>1)): raise ValueError('invalid pixel')
    v=x.astype(np.float64)-x0.astype(np.float64)
    vl=np.nextafter(v,-INF); vh=np.nextafter(v,INF)
    lo=np.nextafter(vl-mu,-INF); hi=np.nextafter(vh-mu,INF)
    finite(lo,hi)
    if np.any(lo>hi): raise ValueError('inverted w interval')
    return lo,hi

def coord_phi_upper(w,x,lam):
    """Upper-bound exact phi at exact binary64 w and float32 x, per coordinate."""
    w=np.asarray(w,dtype=np.float64); x=np.asarray(x)
    if w.shape!=x.shape or x.dtype!=np.float32: raise ValueError('shape/dtype')
    finite(w,x)
    if np.any((x<0)|(x>1)): raise ValueError('pixel outside box')
    lam=float(lam)
    if not math.isfinite(lam) or lam<0: raise ValueError('invalid lambda')
    lower=-x.astype(np.float64); upper=np.nextafter(1.0-x.astype(np.float64),INF)
    if lam==0:
        z=np.where(w>0,upper,lower)
        p=np.nextafter(w*z,INF)
        finite(p)
        if np.any(p<0): raise ValueError('negative support upper')
        return p
    den=2.0*lam
    if not math.isfinite(den) or den<=0: raise ValueError('invalid denominator')
    q=w/den; finite(q)
    zl=np.clip(np.nextafter(q,-INF),lower,upper); zh=np.clip(np.nextafter(q,INF),lower,upper)
    pl=np.nextafter(w*zl,INF); ph=np.nextafter(w*zh,INF)
    p=np.maximum(pl,ph)
    zsq=np.where((zl<=0)&(zh>=0),0.0,np.minimum(zl*zl,zh*zh))
    zsq=np.maximum(0.0,np.nextafter(zsq,-INF))
    penalty=np.maximum(0.0,np.nextafter(lam*zsq,-INF))
    result=np.nextafter(p-penalty,INF); finite(result)
    if np.any(result<0): raise ValueError('negative support upper')
    return result

def support_upper(wlo,whi,x,ehi,lam):
    wlo=np.asarray(wlo,dtype=np.float64); whi=np.asarray(whi,dtype=np.float64)
    if wlo.shape!=whi.shape or wlo.shape!=x.shape or wlo.size==0: raise ValueError('shape')
    finite(wlo,whi)
    if np.any(wlo>whi): raise ValueError('inverted w interval')
    ehi=F(ehi)
    if ehi<0: raise ValueError('negative error')
    lamf=fraction(lam)
    if lamf<0: raise ValueError('negative lambda')
    phi=np.maximum(coord_phi_upper(wlo,x,lam),coord_phi_upper(whi,x,lam))
    total=float(np.sum(phi,dtype=np.float64)); finite(np.asarray(total))
    n=phi.size; gamma=(n*U)/(1-n*U)
    if gamma>=1: raise ValueError('reduction too large')
    exact_upper=(fraction(total)+4*n*ETA)/(1-gamma)
    return lamf*ehi*ehi+exact_upper/n

def root_bound(q,upper):
    q=F(q)
    if q<0: raise ValueError('negative square')
    if not q: return F(0)
    try: x=math.sqrt(float(q))
    except OverflowError: raise ValueError('root overflow')
    if not math.isfinite(x): raise ValueError('root overflow')
    for _ in range(8):
        fx=fraction(x)
        if (fx*fx>=q if upper else fx*fx<=q): return fx
        x=math.nextafter(x, INF if upper else -INF)
    raise ValueError('root enclosure failed')

def box_bounds(wlo,whi,x,r,e):
    rl,rh=map(F,r); el,eh=map(F,e)
    if min(rl,el)<0 or rh<rl or eh<el: raise ValueError('invalid norm interval')
    rm=float((rl+rh)/2); em=float((el+eh)/2)
    ls=[0.0]
    try: l=rm/(2.0*em)
    except (ZeroDivisionError,OverflowError): l=0.0
    if math.isfinite(l) and l>0 and math.isfinite(2*l): ls.append(l)
    bp=min([rh*eh]+[support_upper(wlo,whi,x,eh,l) for l in ls])
    bm=min([rh*eh]+[support_upper(-whi,-wlo,x,eh,l) for l in ls])
    d=max(F(0),rl-eh,el-rh)
    low2=max(d*d,F(0),rl*rl+el*el-2*bm)
    high2=min((rh+eh)**2,rh*rh+eh*eh+2*bp)
    if high2<0 or low2>high2: raise ValueError('inconsistent bound')
    return root_bound(low2,False),root_bound(high2,True),ls
