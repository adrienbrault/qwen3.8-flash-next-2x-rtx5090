#!/usr/bin/env python3
"""CPU-only, duplicate-free MTP graft for the served ExLlamaV3 header reader."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def header(path):
    with open(path, 'rb') as f:
        n, = struct.unpack('<Q', f.read(8))
        assert 0 < n <= 100_000_000 and n + 8 <= path.stat().st_size, path
        h = json.loads(f.read(n))
    for k, v in h.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        assert 0 <= a <= b <= path.stat().st_size - n - 8, (path, k)
    return n, h


def inventory(root):
    keys = {}
    for path in sorted(root.glob('*.safetensors')):
        _, h = header(path)
        for k, v in h.items():
            if k == '__metadata__':
                continue
            assert k not in keys, ('duplicate tensor', k, keys.get(k), path)
            keys[k] = (path.name, v)
    assert keys, root
    return keys


def verify(base, dest, sidecar):
    a, b = inventory(base), inventory(dest)
    _, sh = header(sidecar)
    sk = {k for k in sh if k != '__metadata__'}
    assert sk and all(k.startswith('mtp.') for k in sk), 'non-MTP sidecar key'
    assert sk == {k for k in a if k.startswith('mtp.')}, 'MTP key sets differ'
    assert set(a) == set(b), 'graft key set differs from base'
    for k in a:
        if k.startswith('mtp.'):
            assert b[k][0] == 'mtp-2.6bpw.safetensors' and b[k][1] == sh[k], k
        else:
            assert a[k] == b[k], ('target tensor descriptor changed', k)
    return dict(keys=len(b), mtp_keys=len(sk), duplicates=0,
                base_mtp_bytes=sum(v['data_offsets'][1] - v['data_offsets'][0]
                                   for k, (_, v) in a.items() if k.startswith('mtp.')))


def build(base, sidecar, dest, pins, expected_mtp_bytes=887398952):
    assert not dest.exists() and not dest.is_symlink(), f'refusing existing {dest}'
    assert base.parent == dest.parent == sidecar.parent.parent, 'all links must stay within models mount'
    assert sidecar.stat().st_size == pins['size'], 'sidecar size mismatch'
    assert digest(sidecar) == pins['sha256'], 'sidecar SHA256 mismatch'
    original = inventory(base)
    _, sh = header(sidecar)
    assert {k for k in sh if k != '__metadata__'} == {k for k in original if k.startswith('mtp.')}, 'MTP key sets differ'
    stage = dest.with_name(dest.name + '.new')
    assert not stage.exists() and not stage.is_symlink(), stage
    stage.mkdir()
    copied = []
    for src in sorted(base.iterdir()):
        out = stage / src.name
        if src.name == 'model.safetensors.index.json':
            continue  # regenerated below, although the served loader ignores it
        if src.suffix == '.safetensors':
            n, h = header(src)
            tensors = {k: v for k, v in h.items() if k != '__metadata__'}
            keep = {k: v for k, v in tensors.items() if not k.startswith('mtp.')}
            if not keep:
                continue
            if len(keep) != len(tensors):
                # Preserve the data origin and every target offset. The served reader permits gaps.
                # This is a private ExLlamaV3 checkpoint, not a safetensors-library export.
                if '__metadata__' in h:
                    keep['__metadata__'] = h['__metadata__']
                encoded = json.dumps(keep, separators=(',', ':')).encode()
                assert len(encoded) <= n
                shutil.copyfile(src, out)
                with out.open('r+b') as f:
                    f.seek(8)
                    f.write(encoded.ljust(n, b' '))
                # Full streaming hash verifies ALL payload bytes, including target tensors.
                def payload_hash(p):
                    result = hashlib.sha256()
                    with p.open('rb') as f:
                        f.seek(n + 8)
                        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                            result.update(chunk)
                    return result.hexdigest()
                assert payload_hash(src) == payload_hash(out), 'payload copy mismatch'
                copied.append(src.name)
                continue
        # Relative paths survive /srv/qwen5090/models -> /models in Docker.
        out.symlink_to(os.path.relpath(src, stage))
    (stage / 'mtp-2.6bpw.safetensors').symlink_to(os.path.relpath(sidecar, stage))
    result = verify(base, stage, sidecar)
    assert result['base_mtp_bytes'] == expected_mtp_bytes, result
    keys = inventory(stage)
    (stage / 'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata=dict(total_size=sum(v['data_offsets'][1] - v['data_offsets'][0] for _, v in keys.values())),
        weight_map={k: name for k, (name, _) in keys.items()}), sort_keys=True) + '\n')
    # Preserve config verbatim: packing is described by each tensor, not mtp_bits metadata.
    result.update(sidecar_sha256=pins['sha256'], rewritten_shards=copied,
                  layout='unchanged payload, MTP keys excluded from mixed-shard headers')
    (stage / 'mtphead-build.json').write_text(json.dumps(result, indent=2) + '\n')
    stage.rename(dest)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    command, base, sidecar, dest, pinfile = sys.argv[1:]
    args = [Path(x) for x in (base, sidecar, dest)]
    pins = json.loads(Path(pinfile).read_text())
    if command == 'build':
        build(*args, pins)
    elif command == 'verify':
        assert args[1].stat().st_size == pins['size'] and digest(args[1]) == pins['sha256']
        print(json.dumps(verify(args[0], args[2], args[1]), sort_keys=True))
    else:
        raise SystemExit('expected build or verify')
