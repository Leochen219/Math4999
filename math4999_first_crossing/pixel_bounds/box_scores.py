"""Exact rational score composition for the frozen three-arm comparison."""
from fractions import Fraction as F
TAU=F.from_float(.2); CENSOR=81

def sqpos(x): return max(F(0),x)**2

def one_frame(r,e,ab,s):
    rl,rh=map(F,r); el,eh=map(F,e); al,ah=map(F,ab); s=F(s)
    if min(rl,el,al)<0 or rh<rl or eh<el or ah<al or s<=0: raise ValueError('invalid interval')
    d=max(F(0),rl-eh,el-rh)
    radial_safe=min(rh*rh,sqpos(rh+eh-TAU))/(s*s)
    radial_cross=min(rh*rh,sqpos(TAU-d))/(s*s)
    box_safe=min(radial_safe,rh*rh/(s*s),sqpos(ah-TAU)/(s*s))
    box_cross=min(radial_cross,rh*rh/(s*s),sqpos(TAU-al)/(s*s))
    return radial_safe,radial_cross,box_safe,box_cross, sqpos(al-TAU)/(s*s),sqpos(TAU-ah)/(s*s)

def run_scores(frames,label,old_floor):
    if len(frames)!=80 or label not in range(1,82): raise ValueError('invalid target')
    radial=safe=two=lower=F(0)
    for i,(rs,rc,bs,bc,lf,lc) in enumerate(frames,1):
        if i==label:
            radial+=rc; safe+=rc; two+=bc; lower+=lc; break
        radial+=rs; safe+=bs; two+=bs; lower+=lf
    lower=max(F(old_floor),lower)
    if not lower<=two<=safe<=radial: raise ValueError('score bounds inconsistent')
    return lower,two,safe,radial

def candidate_intervals(a,scales):
    if len(a)!=48 or len(scales)<48: raise ValueError('bad prefix length')
    lo=[]; hi=[]; slo=shi=F(0)
    for (x,y),s in zip(a,scales[:48],strict=True):
        x,y,s=F(x),F(y),F(s)
        if x<0 or y<x or s<=0: raise ValueError('invalid candidate interval')
        lo.append(slo+sqpos(TAU-y)/(s*s)); hi.append(shi+sqpos(TAU-x)/(s*s))
        slo+=sqpos(x-TAU)/(s*s); shi+=sqpos(y-TAU)/(s*s)
    lo.extend([slo]*33); hi.extend([shi]*33)
    if any(a>b for a,b in zip(lo,hi,strict=True)): raise ValueError('candidate inverted')
    return lo,hi

def summarize(labels,y,radial):
    labels=set(labels); base=set(range(6,72))|{CENSOR}; f=sorted(labels-{CENSOR})
    hull=(f[-1]-f[0]+1) if f else None
    covered=y in labels
    only_censor=labels=={CENSOR} and y==CENSOR
    primary=covered and labels<=base and (only_censor or (bool(f) and hull<66))
    return {'labels':sorted(labels),'finite_count':len(f),'finite_min':f[0] if f else None,'finite_max':f[-1] if f else None,'finite_span':hull,'censor':CENSOR in labels,'total':len(labels),'covered':covered,'primary':primary,'censor_only_success':only_censor,'secondary_covered_count_lt67':covered and len(labels)<67,'strict_subset_C0':labels<base,'deleted_common':sorted(base-labels),'added_outside_common':sorted(labels-base),'proper_reduction_radial_covered':covered and labels<set(radial)}
