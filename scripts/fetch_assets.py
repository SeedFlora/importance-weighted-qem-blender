"""Download the external research meshes into ``dataset/external/``.

    python scripts/fetch_assets.py            # download or verify, write SOURCES.json
    python scripts/fetch_assets.py --check    # verify local files only, no network

Standard library only. Models come from Alec Jacobson's *common-3d-test-models*
collection. Credits and first-known-appearance links are copied from that
repository's README table; they are the collection's attributions, not a licence
grant. Expected SHA256 values pin the bytes used by this project: a changed
upstream file is reported and not accepted silently. The script is idempotent:
files whose hash already matches are not downloaded again.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'dataset' / 'external'
REPOSITORY = 'https://github.com/alecjacobson/common-3d-test-models'
RAW = 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/{ref}/data/{name}.obj'
TABLE = 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/{ref}/README.md'
COMMITS_API = 'https://api.github.com/repos/alecjacobson/common-3d-test-models/commits/{ref}'
USAGE = 'research use; verify the original licence before publishing renders or redistributing'

# name: (sha256, bytes, README "Model Name", README "Source" text, source link or None,
#        README "First Known Appearance" link or None, extra note).
MODELS = {
    'stanford-bunny': ('1eb35d1e21ce99e5ce911353b6be278990713448dd9e8f5c9387f9de39b32205', 2408417,
                       'Stanford Bunny', 'Stanford University', 'http://graphics.stanford.edu/data/3Dscanrep/',
                       'https://dblp.uni-trier.de/rec/bibtex/conf/siggraph/TurkL94', ''),
    'armadillo': ('d2fd040334c12b16804fbcb9962391232cc24708244617f931baaa871210563e', 4634780,
                  'Armadillo', 'Stanford', 'http://graphics.stanford.edu/data/3Dscanrep/',
                  'https://dblp.uni-trier.de/rec/bibtex/conf/siggraph/KrishnamurthyL96', ''),
    'horse': ('d2e1beaa7e366b61746192345f552f09056c97bbd096cb31eac4b9fe467fa7aa', 4818311,
              'Horse', 'CyberWare',
              'http://web.archive.org/web/20010126102500/https://www.cc.gatech.edu/projects/large_models/horse.html',
              'https://dblp.uni-trier.de/rec/bibtex/conf/siggraph/PraunFH00', ''),
    'max-planck': ('282b278f40cbf6f6054e95d9b7704e51f49c7a2c8d74a2d6977412e6f455ef06', 4573334,
                   'Max Planck', 'MPI', 'https://gfx.cs.princeton.edu/proj/sugcon/models/', None,
                   'README lists the original file as missing ("not full res?").'),
    'igea': ('13dd57979295049c73830313772c379da4508e035551921686e2edc7de9bff0e', 9593281,
             'Igea', 'Cyberware', None, None, ''),
    'ogre': ('b72e270246ec3bd91bd8e082c4acf462352680f09067813322903b4e1a3f6967', 3723236,
             'Ogre', 'Kiaran Ritchie', None, 'https://dblp.uni-trier.de/rec/bibtex/journals/tog/LipmanLC08', ''),
    'beetle': ('46ee9dcbec84b8d1b8f0d743ab1c91b1ed81378672fc0516e7afff5172d7c9b7', 124863,
               'Beetle', 'Ivan Sutherland', 'https://blenderartists.org/t/ivan-sutherlands-1972-mesh/1162769', None,
               'Low-poly car body shell; README lists the original file as missing (quad mesh?).'),
    'fandisk': ('ea5bab2fbf545b1915f0d9faf6cc61ff8c18e0d8174ad61f8e35de15d8f6e3f8', 379559,
                'Fandisk', 'CAD part Pratt & Whitney/Hughes Hoppe', None,
                'https://dblp.org/rec/conf/siggraph/HoppeDDHJMSS94.html', ''),
    'rocker-arm': ('590257e6d01e689354821d807854306c3bfbb02d9cca73404d5a4924f105c287', 665711,
                   'Rocker Arm', 'INRIA', 'http://visionair.ge.imati.cnr.it:8080/ontologies/shapes/search.jsp', None,
                   'README marks the source link with an asterisk ("does scan data exist?").'),
    'spot': ('0738b5e8608fed74e5e8c7aa8dd0af97b4b74f9f6cbf7aac84cd7e40b2e44a75', 330624,
             'Spot', 'Keenan Crane', 'https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/#spot', None,
             'Keenan Crane\'s model repository states "Model and associated data is dedicated under CC0 1.0 '
             'Universal" (page checked 2026-09-21).'),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, timeout=120):
    request = urllib.request.Request(url, headers={'User-Agent': 'smart-lod-dataset/1'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def resolve_commit(ref):
    """Commit SHA that ``ref`` points to now, or None when the API is unavailable."""
    try:
        return json.loads(fetch(COMMITS_API.format(ref=ref), timeout=30)).get('sha')
    except Exception as error:  # rate limits or offline use must not block downloads
        print(f'  note: could not resolve {ref!r} to a commit ({error})')
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ref', default='master', help='branch, tag or commit of the upstream repository')
    parser.add_argument('--only', help='comma-separated subset of model names')
    parser.add_argument('--check', action='store_true', help='verify existing files only (no network)')
    parser.add_argument('--accept-changed', action='store_true',
                        help='keep a file whose upstream hash differs from the pinned hash (breaks comparability)')
    args = parser.parse_args(argv)
    names = args.only.split(',') if args.only else list(MODELS)
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        parser.error('unknown model(s): ' + ', '.join(unknown))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    sources_path = DESTINATION / 'SOURCES.json'
    previous = {}
    if sources_path.exists():
        previous = {entry['name']: entry for entry in json.loads(sources_path.read_text('utf-8')).get('models', [])}
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    commit = None if args.check else resolve_commit(args.ref)
    records, failures = [], []
    for name in names:
        expected, size, title, credit, link, first, note = MODELS[name]
        path = DESTINATION / f'{name}.obj'
        url = RAW.format(ref=args.ref, name=name)
        old = previous.get(name, {})
        status = 'missing'
        if path.exists() and sha256(path) == expected:
            status = 'verified'
        elif not args.check:
            data = fetch(url)
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected and not args.accept_changed:
                failures.append(f'{name}: upstream sha256 {actual} differs from pinned {expected}')
                continue
            temporary = path.with_suffix('.obj.part')
            temporary.write_bytes(data)
            os.replace(temporary, path)
            status = 'downloaded' if actual == expected else 'downloaded-changed'
        if status == 'missing':
            failures.append(f'{name}: missing or hash mismatch ({path})')
            continue
        digest = sha256(path)
        print(f'  {status:18s} {name:15s} {path.stat().st_size:>9,d} bytes  {digest[:12]}')
        if status == 'verified':
            # Keep the provenance of the download that produced these bytes; unknown if none recorded.
            known = old if old.get('sha256') == digest else {}
            origin = dict(url=known.get('url', url), retrieved_at=known.get('retrieved_at'),
                          upstream_ref=known.get('upstream_ref'), upstream_commit=known.get('upstream_commit'))
        else:
            origin = dict(url=url, retrieved_at=now, upstream_ref=args.ref, upstream_commit=commit)
        records.append(dict(
            name=name, file=f'external/{name}.obj', url=origin['url'],
            readme_model_name=title, source_credit=credit, source_link=link,
            first_known_appearance=first, readme_table=TABLE.format(ref='master'),
            collection=REPOSITORY, usage_note=USAGE, note=note or None,
            bytes=path.stat().st_size, sha256=digest, pinned_sha256=expected,
            retrieved_at=origin['retrieved_at'], upstream_ref=origin['upstream_ref'],
            upstream_commit=origin['upstream_commit'], last_verified_at=now,
        ))
    if records:
        kept = {entry['name']: entry for entry in previous.values() if entry['name'] not in names}
        kept.update({record['name']: record for record in records})
        ordered = [kept[name] for name in MODELS if name in kept]
        sources_path.write_text(json.dumps(dict(
            schema_version=1, generated_by='scripts/fetch_assets.py', collection=REPOSITORY,
            usage_note=USAGE, models=ordered), indent=2) + '\n', encoding='utf-8')
        print(f'wrote {sources_path.relative_to(ROOT)} ({len(ordered)} models)')
    for failure in failures:
        print('ERROR', failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
