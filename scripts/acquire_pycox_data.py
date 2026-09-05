"""Explicit opt-in acquisition from pycox providers; never writes raw data into the repository."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
# Preserve the original loader's package-local raw cache, not processed=True output.
if os.environ.get('PYCOX_DATA_DIR'):
    raise SystemExit('Unset PYCOX_DATA_DIR for this original package-cache workflow; see data/DATA_SOURCES.md.')
from pycox import datasets
observed=[]
expected={'metabric':1904,'support':8873,'gbsg':2232,'flchain':7874,'nwtco':4028}
for name,n in expected.items():
    loader=getattr(datasets,name);p=Path(loader.path).resolve()
    if p.is_relative_to(ROOT):
        raise SystemExit('Dataset cache is inside the repository (including its .venv). Use an external Python/Conda environment for full data experiments.')
    df=loader.read_df(processed=False) if name in ['flchain','nwtco'] else loader.read_df()
    if len(df)!=n:raise SystemExit(f'{name}: expected {n} raw-cache rows, received {len(df)}; do not silently substitute.')
    observed.append({'dataset':name,'rows':len(df),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    print(f'{name}: {len(df)} rows; cached outside repository')
print(json.dumps(observed,indent=2))
