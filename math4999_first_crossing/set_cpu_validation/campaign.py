"""Frozen 87-case campaign; invocation creates a new audit directory."""
import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from . import models as m, oracles as o

PROTOCOL_SHA='a18b6c28c1556a6e3f60ebd68fb04f43c39f98710a15f13fb98f2a1756507bfc'
TAU=F(1,5)

def exact_tree(value):
    if isinstance(value,(tuple,list)): return type(value)(exact_tree(x) for x in value)
    if isinstance(value,int) and not isinstance(value,bool): return F(value)
    return value

def build_cases():
    cases=[]
    def add(kind, **data): cases.append(dict(id=f'{kind}{sum(c["kind"]==kind for c in cases)+1:02}',kind=kind,**data))
    for gain,interval,H in itertools.product((-1,F(1,2),1,F(3,2)),((0,F(1,10)),(F(-1,10),F(1,10)),(F(1,10),F(1,10))),range(1,5)):
        add('A',gain=F(gain),interval=exact_tree(interval),H=H)
    matrices=(((1,0),(0,1)),((F(3,5),F(-4,5)),(F(4,5),F(3,5))),((F(4,5),1),(0,F(4,5))))
    for matrix,H in itertools.product(matrices,range(1,4)): add('B',matrix=exact_tree(matrix),alphabet=exact_tree(((0,F(-1,10)),(0,F(1,10)))),H=H)
    rectangles=(((0,F(1,10)),(0,F(1,5))),((F(1,10),F(1,5)),(0,F(1,5))),((0,F(1,10)),(F(1,10),F(3,10))),((F(1,10),F(1,5)),(F(1,10),F(3,10))),((F(3,10),F(3,10)),(0,0)),((0,0),(0,0)))
    for k,rect in itertools.product((0,F(3,5),1),rectangles): add('G',k=F(k),rectangle=exact_tree(rect))
    for gain,a,expected in ((0,F(1,10),(5,)),(0,F(3,10),(1,)),(F(1,2),F(1,10),(5,)),(1,F(1,10),(3,4,5)),(2,F(1,10),(2,3,4,5)),(4,F(1,10),(2,))): add('R',gain=F(gain),a=a,H=4,expected=expected)
    examples=[dict(gains=[1]*3,intervals=[(F(9,100),F(1,10))]*3,tau=TAU),dict(gains=[0]*3,intervals=[(TAU,TAU)]*3,tau=TAU),dict(gains=[0,-1],alphabets=[(0,1),(1,)],tau=F(1,2)),dict(gains=[0]*3,intervals=[(0,1),(0,0),(1,1)],tau=F(1,2)),dict(theta=(-1,1),fixed_across_steps=True,tau=F(3,2),H=2),dict(gains=[1]*4,alphabets=[(F(-1,10),F(1,10))]*4,tau=TAU)]
    for i,(expected,data) in enumerate(zip(((3,),(4,),(1,2),(1,3),(3,),(3,5)),examples),1):
        data={k: (v if k in ('H','fixed_across_steps') else exact_tree(v)) for k,v in data.items()}
        add('P',example=i,expected=expected,parameters=data)
    return cases

def metrics(times,H,truth):
    values=sorted(times); events=[t for t in values if t<=H]
    return dict(times=values,event_times=events,censored=H+1 in times,cardinality=len(values),holes=[t for t in range(values[0],values[-1]+1) if t not in times],hull_width=values[-1]-values[0],safe_prefix=values[0]-1,finite_right_endpoint=values[-1] if values[-1]<=H else None,unique=len(values)==1,unique_event=len(values)==1 and bool(events),censor_only=values==[H+1],abstain=len(values)!=1,wrong_unique_identification=len(values)==1 and set(values)!=set(truth))

def rectangle_certificate(rect,k,q,p):
    (a,b),(c,d)=rect; x,y=p; dx=2*(x-k*y); dy=2*(y-k*x)
    def kkt(v,lo,hi,derivative):
        return lo<=v<=hi and (lo==hi or (derivative>=0 if v==lo else derivative<=0 if v==hi else derivative==0))
    root={F(0):F(1),F(3,5):F(4,5),F(1):F(0)}[F(k)]
    beta=(x,F(0)); perturb=(-k*y,root*y)
    dot=sum(u*v for u,v in zip(beta,perturb)); total=tuple(u+v for u,v in zip(beta,perturb))
    return dict(feasible_kkt=kkt(x,a,b,dx) and kkt(y,c,d,dy),positive_semidefinite=0<=k<=1,objective=q==x*x+y*y-2*k*x*y,beta_norm_squared=sum(v*v for v in beta)==x*x,perturb_norm_squared=sum(v*v for v in perturb)==y*y,inner_product=dot==-k*x*y,attained=sum(v*v for v in total)==q,old_g_dominated=q>=m.old_g_squared(*rect,k),beta=beta,perturbation=perturb)

def scalar_paths(gains,alphabets,tau):
    times=set(); witnesses={}
    for path in itertools.product(*alphabets):
        x=F(0); t=len(gains)+1
        for i,(g,d) in enumerate(zip(gains,path),1):
            x=g*x+d
            if abs(x)>tau: t=i; break
        times.add(t); witnesses.setdefault(t,exact_tree(path))
    return times,witnesses

def validate_case(case,budget):
    o.check(budget); kind=case['kind']; details={}
    if kind=='A':
        H=case['H']; gain=case['gain']; interval=case['interval']; gains=[gain]*H; intervals=[interval]*H
        times,trace=m.interval_survival(gains,intervals,TAU); oracle,_=o.interval_oracle(gains,intervals,TAU,deadline=budget)
        a,b=interval; low=0 if a<=0<=b else min(abs(a),abs(b)); high=max(abs(a),abs(b))
        free,free_trace=m.radial_survival(abs(gain),abs(gain),low,high,TAU,H)
        ok=times==oracle and times<=free
        details=dict(trace=trace,oracle=oracle,free_direction_times=free,free_direction_trace=free_trace,inclusion=times<=free)
    elif kind=='B':
        H=case['H']; matrices=[case['matrix']]*H; alphabets=[case['alphabet']]*H
        times,trace=m.vector_survival(matrices,alphabets,TAU); oracle,witnesses=o.vector_oracle(matrices,alphabets,TAU,deadline=budget)
        ok=times==oracle; details=dict(trace=trace,oracle=oracle,witnesses=witnesses)
    elif kind=='G':
        q,p=m.rectangle_min(*case['rectangle'],case['k']); cert=rectangle_certificate(case['rectangle'],case['k'],q,p)
        ok=all(v for k,v in cert.items() if isinstance(v,bool)); return dict(id=case['id'],kind=kind,passed=ok,min_squared=q,argmin=p,old_g_squared=m.old_g_squared(*case['rectangle'],case['k']),certificate=cert)
    elif kind=='R':
        H=case['H']; times,trace=m.radial_survival(case['gain'],case['gain'],case['a'],case['a'],TAU,H)
        ok=times==set(case['expected']); details=dict(trace=trace,expected=case['expected'])
    else:
        i=case['example']; p=case['parameters']; tau=p['tau']; H=p['H'] if 'H' in p else len(p['gains'])
        if i in (1,2,4):
            times,trace=m.interval_survival(p['gains'],p['intervals'],tau); details={'trace':trace}
        elif i==3:
            times,w=scalar_paths(p['gains'],p['alphabets'],tau); details={'witnesses':w}
        elif i==5:
            times=set()
            for theta in p['theta']:
                e1=theta; e2=e1-theta
                times.add(1 if abs(e1)>tau else 2 if abs(e2)>tau else H+1)
            negative={1 if abs(t1)>tau else 2 if abs(t1-t2)>tau else H+1 for t1,t2 in itertools.product(p['theta'],repeat=2)}
            details={'fixed_theta':p['theta'],'stepwise_reselection_negative_control':negative,'negative_control_detected':p['fixed_across_steps'] and negative=={2,3} and negative>times}
        else:
            times,w=scalar_paths(p['gains'],p['alphabets'],tau)
            magnitude=abs(p['alphabets'][0][0]); free,_=m.radial_survival(abs(p['gains'][0]),abs(p['gains'][0]),magnitude,magnitude,tau,H)
            details={'witnesses':w,'free_direction_times':free,'different_information_classes':True}
        ok=times==set(case['expected']) and (i!=5 or details['negative_control_detected']) and (i!=6 or free=={3,4,5} and times<free)
        details['expected']=case['expected']
    o.check(budget)
    return dict(id=case['id'],kind=kind,passed=ok,metrics=metrics(times,H,oracle if kind in ('A','B') else case['expected']),details=details)

def serial(value):
    if isinstance(value,F): return f'{value.numerator}/{value.denominator}'
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)): return [serial(v) for v in (sorted(value) if isinstance(value,set) else value)]
    return value

def dump(value): return json.dumps(serial(value),ensure_ascii=False,sort_keys=True,indent=2)+'\n'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(dump(value),encoding='utf-8')

def run(protocol,output,max_seconds=600):
    protocol=Path(protocol); output=Path(output)
    if isinstance(max_seconds,bool) or not isinstance(max_seconds,(int,float)) or not 0<max_seconds<=600: raise ValueError('invalid deadline')
    if sha(protocol)!=PROTOCOL_SHA: raise ValueError('protocol SHA mismatch')
    cases=build_cases(); code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    output.mkdir(parents=False,exist_ok=False)
    start=time.monotonic(); deadline=start+max_seconds
    manifest=dict(protocol_sha256=PROTOCOL_SHA,code_sha256=code,roster=cases,default_tau=TAU,initial_state=F(0),limits={'seconds':max_seconds,'fm_constraints_per_stage':10000},planned=87)
    write(output/'manifest.json',manifest)
    passed=failed=technical=errors=attempted=written=0; status='PASS'; error=None
    try:
        with (output/'cases.jsonl').open('x',encoding='utf-8') as stream:
            for case in cases:
                attempted+=1; category='passed'
                try:
                    result=validate_case(case,deadline)
                    if not result['passed']: category='failed'; status='STOPPED_FAILURE'
                except o.LimitError as exc:
                    category='technical'; status='STOPPED_TECHNICAL_FAILURE'; error=f'{type(exc).__name__}: {exc}'
                    result=dict(id=case['id'],kind=case['kind'],passed=False,technical_failure=error)
                except Exception as exc:
                    category='error'; status='STOPPED_ERROR'; error=f'{type(exc).__name__}: {exc}'
                    result=dict(id=case['id'],kind=case['kind'],passed=False,error=error)
                line=json.dumps(serial(result),ensure_ascii=False,sort_keys=True)+'\n'
                stream.write(line); stream.flush(); written+=1
                if category=='passed': passed+=1
                elif category=='failed': failed+=1
                elif category=='technical': technical+=1
                else: errors+=1
                if status!='PASS': break
        if passed!=87 and status=='PASS': status='STOPPED_INCOMPLETE'
    except Exception as exc:
        errors+=1; status='STOPPED_ERROR'; error=f'{type(exc).__name__}: {exc}'
    summary=dict(status=status,planned=87,passed=passed,failed=failed,technical_failures=technical,errors=errors,attempted=attempted,written=written,not_run=87-attempted,stopped=status!='PASS',elapsed_seconds=time.monotonic()-start,error=error)
    write(output/'summary.json',summary)
    write(output/'hashes.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='hashes.json'})
    return status=='PASS'

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--protocol',required=True); parser.add_argument('--output',required=True); parser.add_argument('--max-seconds',type=float,default=600)
    args=parser.parse_args()
    try: return 0 if run(args.protocol,args.output,args.max_seconds) else 1
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr); return 1
if __name__=='__main__': sys.exit(main())
