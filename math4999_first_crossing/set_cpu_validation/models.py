"""Exact production algorithms for the restricted trajectory classes."""
from fractions import Fraction as F

def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, F, str)): raise ValueError('exact rational required')
    try: return F(x)
    except (ValueError, ZeroDivisionError): raise ValueError('invalid rational')

def inputs(a, d, tau):
    if not a or len(a) != len(d): raise ValueError('invalid horizon or input lengths')
    tau = rational(tau)
    if tau <= 0: raise ValueError('threshold must be positive')
    return tau

def interval_survival(gains, intervals, tau):
    tau=inputs(gains, intervals, tau); lo=hi=F(0); times=set(); trace=[]
    for t,(gain,bounds) in enumerate(zip(gains,intervals),1):
        gain=rational(gain); a,b=map(rational,bounds)
        if a>b: raise ValueError('empty interval')
        if lo is None: trace.append(None); continue
        p,q=sorted((gain*lo,gain*hi)); p+=a; q+=b
        if p < -tau or q > tau: times.add(t)
        lo,hi=max(p,-tau),min(q,tau)
        if lo>hi: lo=hi=None
        trace.append(None if lo is None else (lo,hi))
    if lo is not None: times.add(len(gains)+1)
    return times,trace

def vector_survival(matrices, alphabets, tau):
    tau=inputs(matrices,alphabets,tau); live={(F(0),F(0))}; times=set(); trace=[]
    for t,(matrix,alphabet) in enumerate(zip(matrices,alphabets),1):
        if len(matrix)!=2 or any(len(row)!=2 for row in matrix) or not alphabet: raise ValueError('invalid vector inputs')
        M=[[rational(v) for v in row] for row in matrix]
        ds=[tuple(map(rational,d)) for d in alphabet]
        if any(len(d)!=2 for d in ds): raise ValueError('invalid vector')
        nxt=set()
        for x,y in live:
            for dx,dy in ds:
                z=(M[0][0]*x+M[0][1]*y+dx,M[1][0]*x+M[1][1]*y+dy)
                if sum(v*v for v in z)>tau*tau: times.add(t)
                else: nxt.add(z)
        live=nxt; trace.append(sorted(live))
    if live: times.add(len(matrices)+1)
    return times,trace

def rectangle_min(xrange, yrange, k):
    a,b=map(rational,xrange); c,d=map(rational,yrange); k=rational(k)
    if min(a,c,k)<0 or a>b or c>d or k>1: raise ValueError('invalid rectangle')
    clamp=lambda v,l,h: max(l,min(v,h))
    candidates=[(x,clamp(k*x,c,d)) for x in (a,b)]+[(clamp(k*y,a,b),y) for y in (c,d)]
    q=lambda p: p[0]**2+p[1]**2-2*k*p[0]*p[1]
    p=min(candidates,key=q)
    return q(p),p

def old_g_squared(xrange,yrange,k):
    a,b=map(rational,xrange); c,d=map(rational,yrange); k=rational(k)
    if min(a,c,k)<0 or a>b or c>d or k>1: raise ValueError('invalid rectangle')
    if k==1: return F(0)
    if c<=k*a: return a*a*(1-k*k)
    if a<=k*c: return c*c*(1-k*k)
    return a*a+c*c-2*k*a*c

def radial_survival(m,L,a,d,tau,H):
    m,L,a,d,tau=map(rational,(m,L,a,d,tau))
    if min(m,L,a,d)<0 or tau<=0 or m>L or a>d or isinstance(H,bool) or not isinstance(H,int) or H<1: raise ValueError('invalid radial inputs')
    lo=hi=F(0); times=set(); trace=[]
    for t in range(1,H+1):
        if lo is None: trace.append(None); continue
        x,y=m*lo,L*hi
        lower=max(F(0),x-d,a-y); upper=y+d
        if upper>tau: times.add(t)
        lo,hi=lower,min(upper,tau)
        if lo>hi: lo=hi=None
        trace.append(None if lo is None else (lo,hi))
    if lo is not None: times.add(H+1)
    return times,trace
