"""ple-ckpt-clone r1: PLELayerState.stash() returned id_state[slot, :ctx].cpu(); id_state is a host tensor, so .cpu() is a
no-op and every stored PLE checkpoint aliased the live slot (later prefill/decode writes and the eos fill of the next job in
that slot changed it). Replace it with .clone(). Refuses any ple.py other than the served baseline."""
import hashlib, importlib.util, os, sys
BASE = "c18233c0265b8d9d22ecbcb3e2e727f9552ddc417c98f8e207b5ae15c3a5bc29"
POST = "cbd597f6507f0d82f0cf59608340985cab339f6029a88ee75d7c04128164db44"
OLD = '        return (self.conv_state[slot, :, :self.win].cpu(), self.id_state[slot, :self.ctx].cpu())\n'
NEW = '        # id_state lives on the host, where .cpu() returns a view of the live slot: clone so a stored checkpoint cannot\n        # follow later writes to the slot (ple-ckpt-clone r1)\n        return (self.conv_state[slot, :, :self.win].cpu(), self.id_state[slot, :self.ctx].clone())\n'
spec = importlib.util.find_spec("exllamav3")
path = os.path.join(list(spec.submodule_search_locations)[0], "modules", "ple.py")
s = open(path).read()
h = hashlib.sha256(s.encode()).hexdigest()
if h == POST:
    print("ple-ckpt-clone: already applied"); sys.exit(0)
if h != BASE:
    sys.exit(f"ple-ckpt-clone: {path} sha256 {h} is not the served baseline {BASE}")
s = s.replace(OLD, NEW)
assert hashlib.sha256(s.encode()).hexdigest() == POST
open(path, "w").write(s)
print("ple-ckpt-clone: patched", path)
