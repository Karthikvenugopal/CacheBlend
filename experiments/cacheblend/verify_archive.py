"""Verify the separate closed edit-repair archive without modifying it."""
import hashlib
import json
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'archives/kvrepair-final-2026-11-09'
m=json.loads((p/'manifest.json').read_text())
archive=p/'implementation-and-results.tar.gz'
with archive.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
assert digest == m['archive_sha256']
with tarfile.open(archive) as tf:
    index={x.name:x for x in tf.getmembers() if x.isfile()}
    checked=0
    for name,expected in m['files_sha256'].items():
        content=tf.extractfile(index[name]).read()
        assert hashlib.sha256(content).hexdigest()==expected,name
        checked+=1
out={'archive':str(archive.relative_to(root)), 'archive_sha256':digest,
     'verified_file_count':checked,'passed':True,'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()}
(root/'artifacts/cacheblend/edit-repair-archive-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
