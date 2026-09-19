"""Builds the installed PLELayerState class alone (ast, no package import: the package needs CUDA) and checks that a stored
checkpoint no longer follows writes to the live slot, and that unstash restores it."""
import ast, importlib.util, os, types, torch
spec = importlib.util.find_spec("exllamav3")
path = os.path.join(list(spec.submodule_search_locations)[0], "modules", "ple.py")
tree = ast.parse(open(path).read())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PLELayerState")
ns = {"torch": torch, "PLELayer": object}   # annotations only
exec(compile(ast.Module(body = [cls], type_ignores = []), path, "exec"), ns)
S = ns["PLELayerState"]
emb = types.SimpleNamespace(context_len = 3, eos_token_id = 7)
mod = types.SimpleNamespace(conv_state_len = 2, ple_embedding = emb, hc_mult = 1, hidden_size = 4)
st = S(mod, max_batch_size = 2, max_history = 4, cache_id = 0)
st.alloc("cpu")
st.id_state[0, :3] = torch.tensor([11, 12, 13])
ck = st.stash(0)
st.id_state[0].fill_(99)                     # later writes / the next job's eos fill
assert ck[1].tolist() == [11, 12, 13], f"stored checkpoint followed the live slot: {ck[1].tolist()}"
st.unstash(1, ck)
assert st.id_state[1, :3].tolist() == [11, 12, 13]
print("ple-ckpt-clone test OK")
