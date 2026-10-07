"""Compress large evidence tables while retaining their exact relative paths."""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--evidence',required=True); ap.add_argument('--local')
    args=ap.parse_args(); root=Path(args.evidence)
    names={'splits.csv','preprocessing.json','predictions.csv','fit-ids.csv','ros-lineage.csv'}
    members=sorted(p for p in root.rglob('*') if p.is_file() and p.name in names)
    archive=root/'evidence-tables.tar.gz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for p in members:
                    info=tar.gettarinfo(str(p),arcname=str(p.relative_to(root)))
                    info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
                    with p.open('rb') as f: tar.addfile(info,f)
    index={'archive':archive.name,'sha256':digest(archive),'bytes':archive.stat().st_size,
           'entries':[{'path':str(p.relative_to(root)),'sha256':digest(p),'bytes':p.stat().st_size} for p in members],
           'contains_raw_features_or_checkpoints':False,
           'restore':'Extract this archive into the experiment directory to restore the full tables; keep raw datasets and weights separately.'}
    (root/'archive-index.json').write_text(json.dumps(index,indent=2)+'\n')
    prep=json.loads((root/'preprocessing.json').read_text()); ids=prep.pop('fit_ids')
    prep.update(fit_record_count=len(ids),complete_fit_ids='preprocessing.json inside evidence-tables.tar.gz')
    (root/'preprocessing-summary.json').write_text(json.dumps(prep,indent=2)+'\n')
    # Verify archive payload bytes, not just its top-level digest.
    with tarfile.open(archive,'r:gz') as tar:
        for entry in index['entries']:
            content=tar.extractfile(entry['path']).read()
            assert hashlib.sha256(content).hexdigest()==entry['sha256']
    report=root/'report.md'
    if report.exists():
        s=report.read_text().replace('[split manifest](splits.csv)','[complete split/prediction tables](evidence-tables.tar.gz)')
        s+='\n## Full evidence tables\n\nThe [compressed table archive](evidence-tables.tar.gz) contains the exact split manifest, training-fit IDs/scaler parameters, GAN fit IDs, oversampling lineage, and all per-record predictions. The [archive index](archive-index.json) lists every member and hash. Extract it into this experiment directory to restore the original relative paths. Summary metrics and figures are directly viewable in GitHub. Raw features, generated feature pools, and checkpoints remain in the separately retained local artifact directory.\n'
        if '## Full evidence tables' in report.read_text(): s=report.read_text()
        report.write_text(s)
    # Normalize only directly published small tables; preserve archived exact bytes.
    for p in root.rglob('*.csv'):
        if p.name not in names: p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    index_path=root/'artifacts.sha256'
    previous_local=[line for line in index_path.read_text().splitlines() if '  local/' in line] if index_path.exists() else []
    with index_path.open('w') as f:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!='artifacts.sha256': f.write(f'{digest(p)}  public/{p.relative_to(root)}\n')
        if args.local:
            local=Path(args.local)
            for p in sorted(local.rglob('*')):
                if p.is_file(): f.write(f'{digest(p)}  local/{p.relative_to(local)}\n')
        else:
            for line in previous_local: f.write(line+'\n')
    print(f'Archived {len(members)} full evidence tables; {archive.stat().st_size/1024/1024:.1f} MiB; member hashes verified')


if __name__=='__main__': main()
