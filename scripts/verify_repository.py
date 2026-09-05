"""Verify frozen provenance separately from publication readiness."""
from frozen_results import *
import argparse,hashlib,math,sys
from pypdf import PdfReader
ap=argparse.ArgumentParser();ap.add_argument('--release',action='store_true');args=ap.parse_args()
failures=[];checks=[]
def check(name,condition):
    checks.append({'check':name,'pass':bool(condition)})
    print(('PASS: ' if condition else 'FAIL: ')+name)
    if not condition:failures.append(name)
def near(a,b,tol=5.01e-6):return abs(a-b)<=tol
try:
    required=['README.md','LICENSE','CITATION.cff','.zenodo.json','environment.yml','requirements.txt','.gitignore','CHANGELOG.md','MANIFEST.md','manuscript/template.tex','manuscript/supplement.tex','manuscript/supplement.pdf','manuscript/references.bib','data/README.md','data/DATA_SOURCES.md','docs/REPRODUCIBILITY.md','docs/EXPERIMENT_MAP.md','docs/DATA_DICTIONARY.md','docs/RELEASE_CHECKLIST.md']
    check('Required files exist',all((ROOT/p).is_file() for p in required))
    inv=json.loads((ROOT/'docs/FILE_INVENTORY.json').read_text())
    check('All 93 frozen aggregate files match recorded SHA-256',len(inv)==93 and all((ROOT/r['release']).is_file() and hashlib.sha256((ROOT/r['release']).read_bytes()).hexdigest()==r['release_sha256'] for r in inv))
    m=main_rows();e=event_rows();w=weibull_rows();v=validation_rows();f=flexibility_rows();s=summaries()
    expected=json.loads((ROOT/'experiments/protocol_decomp.json').read_text())
    check('All 34 distinct configurations match original protocol',set((r['cohort'],r['T']) for r in m)==set((r['cohort'],r['T']) for r in expected))
    check('20 successful full-cohort splits stored for all 34 configurations',all(r['splits']==20 for r in m))
    src=(ROOT/'experiments/protocol_decomp.py').read_text()
    check('Original main seed policy 0..19 and test fraction 0.3 remain',all(x in src for x in ['N_SPLITS = 20','TEST_FRAC = 0.3','for s in range(N_SPLITS):','seed=s']))
    factor=math.sqrt(1+20*.3/.7)
    # Both columns are rounded to five decimal places; propagate rounding uncertainty.
    check('Nadeau-Bengio SE agrees with naive SE within printed precision',all(abs(r['SE_'+k]-factor*r['naive_SE_'+k]) <= (1+factor)*5.01e-6 for r in m for k in ['B','KP']))
    for key,value in [('positive_B',29),('nonpositive_KP',31),('resolved_positive_KP',0),('validation_resolved_KP',3)]:check(key+' matches manuscript',s[key]==value)
    sparcs=next(r for r in m if r['cohort']=='sparcs/drg302' and r['T']==6)
    check('SPARCS T=6 four reported values match exact cohort/arm',all(near(sparcs[k],x) for k,x in zip(['D_B','SE_B','D_KP','SE_KP'],[.29336,.00261,.00135,.00128])))
    manuscript=(ROOT/'manuscript/template.tex').read_text(encoding='utf-8')
    maincount=0;validationcount=0;tablesok=True
    for line in manuscript.splitlines():
        mm=re.match(r'(\S+), \$T=(\d+)\$ & (.*)\\\\',line)
        if mm:
            vals=[float(x.strip()) for x in mm[3].split('&')]
            r=next(r for r in m if r['cohort']==mm[1] and r['T']==int(mm[2]))
            tablesok &= len(vals)==5 and all(near(x,r[k],5.01e-4 if k=='modal_event' else 5.01e-6) for x,k in zip(vals,['modal_event','D_B','SE_B','D_KP','SE_KP']));maincount+=1
        mm=re.match(r'(drg\d+) & (\d+)\s*& (.*)\\\\',line)
        if mm:
            vals=[float(x.strip()) for x in mm[3].split('&')]
            r=next(r for r in v if r['cohort']=='sparcs/'+mm[1] and r['T']==int(mm[2]))
            tablesok &= len(vals)==5 and all(near(x,r[k],5.01e-5 if k=='modal_all' else 5.01e-6) for x,k in zip(vals,['modal_all','D_B','SE_B','D_KP','SE_KP']));validationcount+=1
    check('All numerical cells in manuscript main-examples and independent-SPARCS tables match keyed logs',maincount==4 and validationcount==12 and tablesok)
    check('27 E8 and 135 Weibull cells present',len(e)==27 and len(w)==135)
    check('Published E8 correlations match original ordinal-ranking algorithm',near(s['rho_all_legacy'],-.1422,5.01e-5) and near(s['rho_events_legacy'],.8742,5.01e-5))
    pdf=PdfReader(ROOT/'manuscript/supplement.pdf')
    text='\n'.join(p.extract_text(extraction_mode='layout') for p in pdf.pages).replace('\u2212','-')
    bykey={(r['cohort'],r['T']):r for r in m};count=0;ok=True
    for line in text.splitlines():
        mm=re.match(r'\s*(\S+),\s*T\s*=\s*(\d+)\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')\s*$',line)
        if mm:
            count+=1;r=bykey[(mm[1],int(mm[2]))]
            ok &= all(near(float(mm[j]),r[k],5.01e-5 if k=='modal_event' else 5.01e-6) for j,k in enumerate(['modal_event','D_B','SE_B','D_KP','SE_KP'],3))
    check('All 170 numerical cells in supplied supplement S1/S2 match keyed frozen results',count==34 and ok)
    # Extract each PDF table separately to prevent matching a value in the wrong row.
    flexok=True;flexcount=0
    for table,nexttable,metrics in [('Table S3','Table S4',['nll','c_antolini','c_uno']),('Table S4','The matched comparisons',['ibs','ici'])]:
        block=text.split(table)[1].split(nexttable)[0]
        for line in block.splitlines():
            cells=re.findall(r'('+NUM+r')\s*±\s*('+NUM+r')',line)
            cohort=line.split()[0] if line.split() else ''
            if cohort in ['rotgbsg','metabric','nwtco','flchain','support-pycox'] and cells:
                flexcount+=len(cells);flexok &= len(cells)==len(metrics)
                for metric,(val,se) in zip(metrics,cells):
                    r=next(r for r in f if r['cohort']==cohort and r['metric']==metric)
                    flexok &= near(float(val),r['difference']) and near(float(se),r['SE'])
    check('Supplement S3/S4 all 25 paired differences and SEs match',flexcount==25 and flexok)
    for file,expectedrows in [('main/baseline_decomposition',m),('simulations/event_concentration',e),('simulations/grouped_weibull',w),('main/independent_sparcs',v),('supplementary/flexibility',f)]:
        p=ROOT/'results'/f'{file}.csv'
        with p.open(encoding='utf-8',newline='') as stream:actual=list(csv.DictReader(stream))
        check('Generated '+file+' matches frozen values in every cell',len(actual)==len(expectedrows) and all(a=={k:str(value) for k,value in b.items()} for a,b in zip(actual,expectedrows)))
    check('Generated summary JSON matches recomputation',json.loads((ROOT/'results/main/summary.json').read_text())==s)
    check('All three figures present in PDF and PNG',all((ROOT/'results/figures'/f'F{i}_{name}.{ext}').is_file() for i,name in [(1,'baseline_decomposition'),(2,'event_concentration'),(3,'grouped_weibull')] for ext in ['pdf','png']))
    sensitive=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['.git','.venv','__pycache__','reruns']):continue
        if p.suffix.lower() in ['.feather','.yzbx','.pt','.pth','.ckpt','.key','.pem','.zip']:sensitive.append(p.relative_to(ROOT).as_posix())
        if p.suffix.lower() in ['.py','.txt','.log','.md','.json','.yml','.cff','.tex']:
            txt=p.read_text(encoding='utf-8',errors='replace')
            if re.search(r'(?i)(?:[A-Z]:[\\/](?:Users|kanreliability)[\\/]|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN '+r'PRIVATE KEY-----)',txt):sensitive.append(p.relative_to(ROOT).as_posix())
    check('No detected private machine paths, common secret signatures or excluded file types',not sensitive)
    if sensitive:print('Detected:',sensitive)
except Exception as exc:
    check('Verification completed without parser or dependency errors: '+str(exc),False)
print('NOTE: Aggregate verification does not refit models or recover missing per-split observations.')
print('NOTE: E8 standard tie-aware Spearman differs from the archived ordinal-rank statistic; see RELEASE_BLOCKERS.md.')
if args.release:
    metadata=json.loads((ROOT/'.zenodo.json').read_text())
    check('Author-approved licence inserted',metadata.get('license') not in ['AUTHOR_APPROVAL_REQUIRED',None] and 'PENDING AUTHOR APPROVAL' not in (ROOT/'LICENSE').read_text(encoding='utf-8'))
    check('Real GitHub URL inserted', 'YOUR_USERNAME' not in json.dumps(metadata))
    check('Unresolved release blockers addressed in review record',(ROOT/'docs/RELEASE_APPROVAL.json').exists() and all(json.loads((ROOT/'docs/RELEASE_APPROVAL.json').read_text()).get(k) is True for k in ['licence_review','data_and_cache_review','seed_limitation_disclosed','correlation_definition_review','supplement_source_review','manuscript_review','clean_environment_test','author_metadata_review']))
report={'checks':checks,'failures':failures,'scope':'frozen-output consistency' if not args.release else 'release-readiness'}
(ROOT/'results'/('verification-release.json' if args.release else 'verification.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(('FAIL' if failures else 'PASS')+f': {len(checks)-len(failures)}/{len(checks)} checks; '+report['scope'])
raise SystemExit(bool(failures))
