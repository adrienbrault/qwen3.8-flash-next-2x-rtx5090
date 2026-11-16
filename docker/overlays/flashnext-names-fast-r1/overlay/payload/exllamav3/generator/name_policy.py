"""Shared exact aliases and root-only cache domains. No adapter is implemented here."""
from dataclasses import dataclass
import hashlib
import json
import os

RAW = "qwen3.8-flash-next-exl3-2.68bpw-selfcal"
FAST = RAW + "-fast"
ADAPTED = RAW + "-abliterated"
COMBINED = RAW + "-abliterated-fast"
ALIASES = (RAW, FAST, ADAPTED, COMBINED)
ABI = "flashnext-names-r1/root-v1/fly-w1-h3-.3-native-kept-v1"
POLICY_DIGEST = hashlib.sha256(ABI.encode()).hexdigest()


@dataclass(frozen=True)
class NamePolicy:
    fast: bool = False
    adapter_identity: str = ""
    cache_domain: bytes = b""
    alias: str = RAW


NATIVE = NamePolicy()


def domain(alias, adapter_identity="", policy_digest=POLICY_DIGEST):
    if alias == RAW and not adapter_identity:
        return b""
    return json.dumps([ABI, alias, adapter_identity, policy_digest],
                      ensure_ascii=False, separators=(",", ":")).encode()


def fast_enabled():
    # Candidate-image opt-in only. Does not change a request's sampler settings.
    return os.environ.get("EXL3_FLASHNEXT_FAST", "0") == "1"


def resolve(name, loaded_name, enabled=None):
    if name not in ALIASES or name == RAW:
        return NATIVE
    if name != FAST or not (fast_enabled() if enabled is None else enabled):
        raise ValueError("Reserved Flash-Next alias is unavailable: " + name)
    if loaded_name != RAW:
        raise ValueError("Flash-Next alias requires the loaded " + RAW)
    return NamePolicy(True, "", domain(FAST), FAST)


def advertised(loaded_name):
    return (FAST,) if loaded_name == RAW and fast_enabled() else ()


def response_name(requested, loaded):
    return requested if requested in (FAST,) else loaded


def content_hash(tokens, prev_hash, cache_domain=b""):
    # Keep the incumbent byte order and digest, including real None roots.
    h = hashlib.blake2b(digest_size=16)
    if prev_hash is not None:
        h.update(prev_hash)
    elif cache_domain:
        h.update(b"\x00flashnext-domain\x00")
        h.update(len(cache_domain).to_bytes(4, "big"))
        h.update(cache_domain)
    h.update(tokens.numpy().tobytes())
    return h.digest()
