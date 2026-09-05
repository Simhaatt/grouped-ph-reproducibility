"""Strict, keyed parsers for frozen aggregate output, never training data."""
from pathlib import Path
import re, csv, json
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results_raw'
NUM=r'[-+]?\d+(?:\.\d+)?'
def read(name): return (RAW/name).read_text(encoding='utf-8',errors='strict')
def main_rows():
    out=[]; row=None; full=False
    for lineno,line in enumerate(read('experiments/protocol_decomp.txt').splitlines(),1):
        m=re.match(r'^(\S+) @ T=(\d+)\s+n=(\d+).*modal bin mass=([\d.]+)',line)
        if m:
            row=dict(cohort=m[1],T=int(m[2]),n=int(m[3]),modal_all=float(m[4]),source_line=lineno); out.append(row); full=False
        if 'FULL COHORT' in line:
            full=True
            row['splits']=int(re.search(r'\((\d+) splits\)',line)[1])
        if line.strip().startswith('SUBSAMPLE'): full=False
        if row and full:
            m=re.match(r'\s+cox:efron\+(breslow|kalbfleisch-prentice)\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')',line)
            if m:
                key='B' if m[1]=='breslow' else 'KP'
                if 'D_'+key in row: raise ValueError('Duplicate arm in full cohort')
                row['D_'+key]=float(m[2]); row['naive_SE_'+key]=float(m[3]); row['SE_'+key]=float(m[4]); row['line_'+key]=lineno
    ordering={}
    for i,line in enumerate(read('experiments/ordering_variable.txt').splitlines(),1):
        f=line.split()
        if len(f)==8 and f[2] in ('R1','R2'):
            key=(f[0],int(f[1])); assert key not in ordering
            ordering[key]=(f[2],float(f[4]),i,float(f[7]))
    for r in out:
        reg,mass,line,dt=ordering[(r['cohort'],r['T'])]
        assert abs(r['D_B']-dt)<1e-9
        r.update(regime=reg,modal_event=mass,ordering_line=line)
        assert all(k in r for k in ['D_B','D_KP','SE_B','SE_KP','splits'])
    assert len({(r['cohort'],r['T']) for r in out})==len(out)==34
    return out
def event_rows():
    out=[]
    for i,line in enumerate(read('experiments/simulations_e8.txt').splitlines(),1):
        f=line.split()
        if len(f)==11 and f[3] in ['low','mid','high']:
            out.append(dict(n=int(f[0]),T=int(f[1]),rows_per_bin=int(f[2]),target=f[3],modal_all=float(f[4]),modal_event=float(f[5]),events_per_bin=int(f[6]),D_B=float(f[7]),SE_B=float(f[8]),D_base=float(f[9]),D_coef=float(f[10]),source_line=i))
    assert len(out)==27
    return out
def weibull_rows():
    out=[]
    for i,line in enumerate(read('experiments/simulations_e5.txt').splitlines(),1):
        f=line.split()
        if len(f)==9 and re.fullmatch(r'\d+%',f[1]) and f[2].isdigit():
            out.append(dict(shape=float(f[0]),censor_pct=int(f[1][:-1]),n=int(f[2]),T=int(f[3]),modal_all=float(f[4]),D_B=float(f[5]),SE_B=float(f[6]),bias_cox_pct=float(f[7][:-1]),bias_grouped_pct=float(f[8][:-1]),source_line=i))
    assert len(out)==135
    return out
def validation_rows():
    out=[]
    for i,line in enumerate(read('experiments/kp_out_of_sample.txt').splitlines(),1):
        f=line.split()
        if f and re.fullmatch(r'sparcs/drg\d+@T\d+',f[0]):
            out.append(dict(cohort=f[0].split('@')[0],n=int(f[1]),T=int(f[2]),modal_all=float(f[3]),D_B=float(f[4]),SE_B=float(f[5]),D_KP=float(f[7]),SE_KP=float(f[8]),source_line=i))
    assert len(out)==12
    return out
def flexibility_rows():
    out=[]
    for cohort in ['rotgbsg','metabric','nwtco','flchain','support-pycox']:
        txt=read(f'experiments/flexibility_{cohort}.txt')
        line=next(l for l in txt.splitlines() if l.strip().startswith('KAN(matched)') and '+-' in l and re.search(r'[+-]\d+\.\d{5}\+-',l))
        pairs=re.findall(r'('+NUM+r')\+-('+NUM+r')',line)
        assert len(pairs)==5
        for metric,(value,se) in zip(['nll','c_antolini','c_uno','ibs','ici'],pairs):
            out.append(dict(cohort=cohort,metric=metric,difference=float(value),SE=float(se)))
    return out
def write_csv(path,rows):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summaries():
    import numpy as np
    from scipy.stats import spearmanr
    m=main_rows();e=event_rows();v=validation_rows()
    def legacy_rank(key):
        a=np.array([r[key] for r in e]);b=np.array([r['D_B'] for r in e])
        return float(np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1])
    return dict(configurations=len(m),positive_B=sum(r['D_B']>0 for r in m),nonpositive_KP=sum(r['D_KP']<=0 for r in m),resolved_positive_KP=sum(r['D_KP']>1.96*r['SE_KP'] for r in m),validation_resolved_KP=sum(abs(r['D_KP'])>1.96*r['SE_KP'] for r in v),rho_all_legacy=legacy_rank('modal_all'),rho_events_legacy=legacy_rank('modal_event'),rho_all_tie_aware=float(spearmanr([r['modal_all'] for r in e],[r['D_B'] for r in e]).statistic),rho_events_tie_aware=float(spearmanr([r['modal_event'] for r in e],[r['D_B'] for r in e]).statistic))
