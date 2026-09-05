"""Explicit fresh experiment in a new isolated directory; never overwrites frozen logs."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,datetime
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('script',help='Repository-relative experiments/*.py or theory/*.py')
ap.add_argument('args',nargs=argparse.REMAINDER)
a=ap.parse_args(); rel=Path(a.script)
if rel.is_absolute() or '..' in rel.parts or len(rel.parts)!=2 or rel.parts[0] not in ['experiments','theory'] or rel.suffix!='.py':ap.error('Choose an existing experiment or theory script')
if not (ROOT/rel).is_file():ap.error('Script missing')
run=ROOT/'results/reruns'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f');run.mkdir(parents=True)
for folder in ['src','kanrel','experiments','theory','paper']:
    shutil.copytree(ROOT/folder,run/folder,ignore=shutil.ignore_patterns('__pycache__'))
for folder in ['experiments','theory']:
    for p in (ROOT/'results_raw'/folder).glob('*'):
        if p.is_file():shutil.copy2(p,run/folder/p.name)
# Explicit external data only. No raw data are copied into the release tree.
external=os.environ.get('KANREL_DATA')
if external:
    external=Path(external).resolve()
    if not external.is_dir():ap.error('KANREL_DATA must be an existing external data directory')
    datafile=run/'src/kanrel/data.py';s=datafile.read_text(encoding='utf-8')
    s=s.replace('root: str | Path = "data/drsa-data"','root: str | Path = os.environ.get("KANREL_DRSA_DATA", "data/drsa-data")')
    datafile.write_text(s,encoding='utf-8')
env=os.environ.copy();env['PYTHONHASHSEED']='0';env['PYTHONUTF8']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
if external:env.setdefault('KANREL_DRSA_DATA',str(external/'drsa-data'))
(run/'run_metadata.json').write_text(__import__('json').dumps({'script':rel.as_posix(),'arguments':a.args,'python':sys.version,'PYTHONHASHSEED':'0','note':'Fresh run; original E8 hash seed was not recorded. Frozen values are not overwritten.'},indent=2))
print('Fresh run directory:',run,flush=True)
result=subprocess.run([sys.executable,'-u',str(run/rel),*a.args],cwd=run,env=env)
raise SystemExit(result.returncode)
