#!/usr/bin/env python3
"""R826 exact identity checks, CPU only. Archived env maps are independent inputs."""
import argparse
import hashlib
from pathlib import Path
import re

PINS = {
    'OLD': ('6429dfa2035a37dba51bb09651d374e8', 'tabbyapi:merge-tok-r1',
            'sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454', 41),
    'NEW': ('262e9c31f714724409635fbac9df8ac2', 'tabbyapi:r825c-hostprepare',
            'sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9', 46),
}


def env_map(path):
    entries = [s.split('=', 1) for s in Path(path).read_text().splitlines() if s.startswith('EXL3_')]
    assert all(len(x) == 2 for x in entries), 'malformed EXL3 entry'
    assert len({x[0] for x in entries}) == len(entries), 'duplicate EXL3 entry'
    return dict(entries)


def check(arm, launcher, actual, expected):
    md5, image, _, nkeys = PINS[arm]
    raw = Path(launcher).read_bytes()
    assert hashlib.md5(raw).hexdigest() == md5, 'launcher md5 mismatch'
    text = raw.decode()
    assert re.search(r'^DAILY_IMG=(\S+)$', text, re.M)[1] == image, 'DAILY_IMG mismatch'
    selectors = re.search(r'^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$', text, re.M)[1].split()
    assert len(selectors) == nkeys, 'selector count mismatch'
    defaults = dict(s.split('=', 1) for s in selectors)
    assert len(defaults) == nkeys, 'duplicate selector'
    want, got = env_map(expected), env_map(actual)
    assert not any(k.startswith('EXL3_NVME_TIER') for k in got), 'tier enabled'
    assert all(want.get(k) == v for k, v in defaults.items()), 'archived env differs from launcher selectors'
    assert got == want, ('EXL3 env mismatch: ' + ', '.join(
        f'{k}: got {got.get(k)!r}, want {want.get(k)!r}' for k in sorted(set(got) | set(want))
        if got.get(k) != want.get(k)))
    return f'{arm} identity PASS: {nkeys} selectors; {len(got)} container EXL3 entries'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=PINS, required=True)
    for name in ('launcher', 'env', 'expected'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    try:
        print(check(a.arm, a.launcher, a.env, a.expected))
    except (AssertionError, OSError, TypeError, ValueError) as e:
        print(f'{a.arm} identity FAIL: {e}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
