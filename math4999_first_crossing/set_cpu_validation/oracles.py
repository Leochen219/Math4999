"""Independent exact path-space feasibility and full-path enumeration."""
from fractions import Fraction as F
from itertools import product
from math import gcd, lcm
from time import monotonic

class LimitError(RuntimeError): pass

def exact(x):
    if isinstance(x,bool) or not isinstance(x,(int,F,str)): raise ValueError('exact rational required')
    try: return F(x)
    except (ValueError, ZeroDivisionError): raise ValueError('invalid rational')

def check(deadline):
    if deadline is not None and monotonic() >= deadline: raise LimitError('deadline exceeded')

def fme(rows, cap=10000, deadline=None):
    if isinstance(cap,bool) or not isinstance(cap,int) or cap<1: raise ValueError('invalid cap')
    rows=[(tuple(exact(x) for x in a),exact(b),s) for a,b,s in rows]
    if any(not isinstance(s,bool) for a,b,s in rows): raise ValueError('strict flag must be bool')
    n=len(rows[0][0]) if rows else 0
    if any(len(a)!=n for a,b,s in rows): raise ValueError('unequal dimensions')
    def normalize(items):
        out={}
        for a,b,s in items:
            check(deadline)
            if not any(a):
                if b<0 or (b==0 and s): return None
                continue
            den=lcm(*(x.denominator for x in a)); ints=[int(x*den) for x in a]; divisor=gcd(*ints)
            scale=F(den,divisor); key=tuple(x*scale for x in a); bound=b*scale
            prior=out.get(key)
            if prior is None or bound<prior[0] or (bound==prior[0] and s and not prior[1]): out[key]=(bound,s)
            if len(out)>cap: raise LimitError('FM constraint cap exceeded')
        return [(a,b,s) for a,(b,s) in out.items()]
    rows=normalize(rows)
    if rows is None: return False
    for _ in range(n):
        check(deadline); pos=[]; neg=[]; zero=[]
        for a,b,s in rows:
            if a[0]>0: pos.append((a,b,s))
            elif a[0]<0: neg.append((a,b,s))
            else: zero.append((a[1:],b,s))
        def expanded():
            yield from zero
            for a,b,s in pos:
                for c,d,t in neg:
                    check(deadline); u=-c[0]; v=a[0]
                    yield (tuple(u*x+v*y for x,y in zip(a[1:],c[1:])),u*b+v*d,s or t)
        rows=normalize(expanded())
        if rows is None: return False
    return True

def interval_oracle(gains, intervals, tau, cap=10000, deadline=None):
    if not gains or len(gains)!=len(intervals): raise ValueError('invalid horizon')
    gains=[exact(g) for g in gains]
    tau=exact(tau); n=len(gains)
    if tau<=0: raise ValueError('invalid threshold')
    bounds=[]
    for i,(lo,hi) in enumerate(intervals):
        lo,hi=exact(lo),exact(hi)
        if lo>hi: raise ValueError('empty interval')
        unit=[F(0)]*n; unit[i]=1
        bounds.extend([(unit,hi,False),([-v for v in unit],-lo,False)])
    affine=[]
    for s in range(1,n+1):
        coeff=[F(0)]*n
        for j in range(s):
            coeff[j]=F(1)
            for k in range(j+1,s): coeff[j]*=exact(gains[k])
        affine.append(coeff)
    def safe(c): return [(c,tau,False),([-v for v in c],tau,False)]
    times=set()
    for t,c in enumerate(affine,1):
        prior=bounds+[r for old in affine[:t-1] for r in safe(old)]
        if any(fme(prior+[(list(sign*x for x in c),-tau,True)],cap,deadline) for sign in (1,-1)): times.add(t)
    if fme(bounds+[r for c in affine for r in safe(c)],cap,deadline): times.add(n+1)
    return times, {'method':'path_affine_fme'}

def vector_oracle(matrices,alphabets,tau,deadline=None):
    if not matrices or len(matrices)!=len(alphabets): raise ValueError('invalid horizon')
    tau=exact(tau)
    if tau<=0: raise ValueError('invalid threshold')
    witnesses={}
    for path in product(*alphabets):
        check(deadline); x=y=F(0); first=len(matrices)+1
        for t,(M,delta) in enumerate(zip(matrices,path),1):
            if len(M)!=2 or any(len(row)!=2 for row in M) or len(delta)!=2: raise ValueError('invalid vector inputs')
            x,y=exact(M[0][0])*x+exact(M[0][1])*y+exact(delta[0]),exact(M[1][0])*x+exact(M[1][1])*y+exact(delta[1])
            if x*x+y*y>tau*tau: first=t; break
        witnesses.setdefault(first,path)
    if any(not a for a in alphabets): raise ValueError('empty alphabet')
    return set(witnesses),witnesses
