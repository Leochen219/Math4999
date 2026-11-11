"""Strict first-crossing helpers; rational inputs give exact results."""
import math
from fractions import Fraction

def _valid(x):
    try:
        if not math.isfinite(x): raise ValueError('nonfinite input')
    except (TypeError, OverflowError):
        raise ValueError('invalid input')

def hit(values, tau):
    _valid(tau)
    if tau < 0 or not values: raise ValueError('invalid threshold or empty trajectory')
    for x in values:
        _valid(x)
        if x < 0: raise ValueError('negative error')
    for t, x in enumerate(values):
        if t and x > tau: return {'event': True, 'time': t}
    return {'event': False, 'time': len(values)}

def upper(L, d, H, initial=0):
    for x in (L,d,initial):
        _valid(x)
        if x < 0: raise ValueError('negative parameter')
    if not isinstance(H,int) or H<0: raise ValueError('invalid horizon')
    out=[initial]
    for _ in range(H): out.append(L*out[-1]+d)
    return out

def lower_squared(m,a,kappa,H,initial_squared=0):
    for x in (m,a,kappa,initial_squared):
        _valid(x)
        if x < 0: raise ValueError('negative parameter')
    if kappa>1 or not isinstance(H,int) or H<0: raise ValueError('invalid parameter')
    out=[initial_squared]; q=1-kappa
    for _ in range(H): out.append(q*(a*a+m*m*out[-1]))
    return out

def aligned(L,d,tau,H):
    for x in (L,d,tau):
        _valid(x)
        if x<0: raise ValueError('negative parameter')
    if not isinstance(H,int) or H<0: raise ValueError('invalid horizon')
    if d==0 or (L<1 and d <= tau*(1-L)):
        return {'kind':'infinite','time':None}
    r=hit(upper(L,d,H),tau)
    return {'kind':'finite' if r['event'] else 'censored','time':r['time']}

def conformal_index(n, alpha):
    if not isinstance(n,int) or n<0 or not isinstance(alpha,Fraction) or not 0<alpha<1: raise ValueError('invalid rank parameters; alpha must be Fraction')
    return math.ceil((n+1)*(1-alpha))

def truncated_time(values,tau):
    return hit(values,tau)['time']



def sharp_step_squared(a, b, k):
    """Exact squared one-step minimum, for nonnegative magnitudes and 0<=k<=1."""
    for x in (a,b,k):
        _valid(x)
        if x < 0: raise ValueError('negative parameter')
    if k > 1: raise ValueError('invalid cancellation parameter')
    if k == 1: return a*0
    if b <= k*a: return a*a*(1-k*k)
    if a <= k*b: return b*b*(1-k*k)
    return a*a+b*b-2*k*a*b

def sharp_lower_squared(m, a, k, H, initial_squared=None):
    """Decimal iteration of the sharper lower envelope, with context rounding."""
    from decimal import Decimal
    if initial_squared is None: initial_squared=Decimal(0)
    if not all(isinstance(x,Decimal) for x in (m,a,k,initial_squared)):
        raise ValueError('Decimal parameters required')
    for x in (m,a,k,initial_squared):
        _valid(x)
        if x < 0: raise ValueError('negative parameter')
    if k > 1 or not isinstance(H,int) or H<0: raise ValueError('invalid parameter')
    values=[initial_squared]
    for _ in range(H):
        values.append(sharp_step_squared(a,m*values[-1].sqrt(),k))
    return values
