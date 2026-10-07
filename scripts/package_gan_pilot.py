"""Publish exact ID/prediction evidence without raw features or model objects."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(data): return hashlib.sha256(data).hexdigest()


def package(root):
    large_names = {'splits.csv','preprocessing.json','fit-ids.csv','gan-fit-lineage.csv','ros-lineage.csv','predictions.csv'}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name in large_names)
    plan = json.loads((root/'plan.json').read_text())
    expected = 26 if plan['gan'].get('family_balanced_fit',False) else 23
    if len(paths) != expected:
        raise ValueError(f'Expected {expected} exact evidence tables')
    archive = root/'evidence-tables.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=9) as tar:
        for path in paths:
            tar.add(path,arcname=str(path.relative_to(root)))
    with tarfile.open(archive,'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == len(paths)
        for path in paths:
            assert digest(tar.extractfile(str(path.relative_to(root))).read()) == digest(path.read_bytes())
    # Normalize the small published tables, keeping exact large files in the archive.
    for p in root.rglob('*.csv'):
        if p.name not in large_names:
            p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    items = [dict(path=str(p.relative_to(root)),sha256=digest(p.read_bytes()),bytes=p.stat().st_size)
             for p in sorted(root.rglob('*')) if p.is_file() and p.name not in large_names|{'artifacts.json'}]
    index = dict(archive_sha256=digest(archive.read_bytes()),archive_bytes=archive.stat().st_size,
                 archive_members=[dict(path=str(p.relative_to(root)),sha256=digest(p.read_bytes()),bytes=p.stat().st_size) for p in paths],
                 directly_published=items,excludes=['raw feature records','model weights','synthetic feature arrays','pickled CTGAN objects'])
    (root/'artifacts.json').write_text(json.dumps(index,indent=2)+'\n')
    report = root/'report.md'; text = report.read_text()
    text = text.split('\n## Download exact evidence tables')[0]
    text += ('\n## Download exact evidence tables\n\n'
             f'[Evidence archive](evidence-tables.tar.gz) contains all {expected} exact split, fit, lineage and prediction tables. '
             '[Artifact index](artifacts.json) records each member\'s hash and the archive hash. '
             'Extract into this directory before rerunning verification. No raw features, generated records or models are included.\n')
    report.write_text(text)
    # report changed above: refresh its direct hash.
    for item in index['directly_published']:
        if item['path'] == 'report.md':
            item.update(sha256=digest(report.read_bytes()),bytes=report.stat().st_size)
    (root/'artifacts.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(archive=str(archive),members=len(paths),bytes=archive.stat().st_size)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--evidence',required=True,type=Path)
    package(ap.parse_args().evidence)
