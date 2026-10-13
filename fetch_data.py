"""Download the data record of this repository from Zenodo and unpack it into data/ (or $DFW_DATA).

python3 fetch_data.py [part ...]      parts: signals catalogue measurement selfgravity masking inference results
                                      (default: all; 74 MB compressed, 204 MB unpacked)
Each part is a tarball dfw_<part>.tar.gz whose MD5 checksum is verified against the record before unpacking.
"""
import hashlib
import json
import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import paths  # noqa: E402

RECORD = os.environ.get('DFW_ZENODO_RECORD', '23343105')     # Zenodo record id of the data DOI
PARTS = ['signals', 'catalogue', 'measurement', 'selfgravity', 'masking', 'inference', 'results']


def main(parts):
    if not RECORD.isdigit():
        sys.exit('Set the Zenodo record id (RECORD in fetch_data.py or $DFW_ZENODO_RECORD).')
    meta = json.load(urllib.request.urlopen('https://zenodo.org/api/records/%s' % RECORD))
    files = {f['key']: f for f in meta['files']}
    tmp = os.path.join(paths.DATA, '_download')
    os.makedirs(tmp, exist_ok=True)
    for part in parts:
        key = 'dfw_%s.tar.gz' % part
        f = files[key]
        url = f.get('links', {}).get('self') or 'https://zenodo.org/records/%s/files/%s?download=1' % (RECORD, key)
        out = os.path.join(tmp, key)
        print('downloading %s (%.1f MB)' % (key, f['size'] / 1e6), flush=True)
        urllib.request.urlretrieve(url, out)
        md5 = hashlib.md5(open(out, 'rb').read()).hexdigest()
        if md5 != f['checksum'].split(':')[-1]:
            sys.exit('%s: checksum mismatch' % key)
        with tarfile.open(out) as t:
            t.extractall(paths.DATA)
        os.remove(out)
    os.rmdir(tmp)
    print('data in %s' % paths.DATA)


if __name__ == '__main__':
    bad = [p for p in sys.argv[1:] if p not in PARTS]
    if bad:
        sys.exit('unknown part(s) %s; choose from %s' % (bad, PARTS))
    main(sys.argv[1:] or PARTS)
