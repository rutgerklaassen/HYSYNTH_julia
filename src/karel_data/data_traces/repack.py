# pack_for_julia.py
import numpy as np
import os

# --- load your existing split (unchanged) ---
src_path = "val.npz"   # change to train/test as needed
d = np.load(src_path, allow_pickle=True)

inputs   = d["inputs"]     # [N,H,W,C] numeric
outputs  = d["outputs"]    # [N,H,W,C] numeric
traces   = d["traces"]     # ragged: list of [H,W,C] frames
codes    = d.get("codes", None)            # ragged: list of int tokens (optional)
lengths  = d.get("code_lengths", None)     # [N] (optional)
num_examples_per_code = d.get("num_examples_per_code", None)

# --- pack traces: ragged -> flat tensor + offsets ---
N = len(traces)
lens = np.fromiter((len(ts) for ts in traces), dtype=np.int32, count=N)
starts = np.empty(N, dtype=np.int32)
starts[0] = 0
if N > 1:
    np.cumsum(lens[:-1], out=starts[1:])
T = int(lens.sum())

# infer shape/dtype from first frame
H, W, C = traces[0][0].shape
dtype = inputs.dtype  # keep same dtype as inputs/outputs
trace_frames = np.empty((T, H, W, C), dtype=dtype)
pos = 0
for ts in traces:
    L = len(ts)
    trace_frames[pos:pos+L] = ts
    pos += L
trace_offsets = np.stack([starts, lens], axis=1)  # [N,2], 0-based (start, length)

# --- pack codes: ragged -> padded int matrix (optional) ---
codes_padded = None
code_lengths = None
if codes is not None and lengths is not None:
    code_lengths = np.asarray(lengths, dtype=np.int32)  # [N]
    maxlen = int(max((len(c) for c in codes), default=0))
    codes_padded = np.zeros((len(codes), maxlen), dtype=np.int32)
    for i, seq in enumerate(codes):
        L = len(seq)
        if L:
            codes_padded[i, :L] = seq

# --- write Julia-friendly NPZ (numeric only) ---
dst_path = os.path.splitext(src_path)[0] + "_packed.npz"
save_kwargs = dict(
    inputs=inputs,                   # [N,H,W,C]
    outputs=outputs,                 # [N,H,W,C]
    trace_frames=trace_frames,       # [T,H,W,C]
    trace_offsets=trace_offsets,     # [N,2]
)
if codes_padded is not None:
    save_kwargs["codes_padded"] = codes_padded   # [N,max_len]
    save_kwargs["code_lengths"] = code_lengths   # [N]
if num_examples_per_code is not None:
    # ensure scalar int32 to keep it plain numeric
    save_kwargs["num_examples_per_code"] = np.int32(num_examples_per_code)

np.savez_compressed(dst_path, **save_kwargs)
print(f"Wrote {dst_path}")

# --- (optional) the human-readable print you already had ---
if False:  # flip to True if you still want to print one example
    from karel import KarelForSynthesisParser
    from karel.karel import Karel
    from karel.utils import beautify

    i = 999
    parser = KarelForSynthesisParser()
    id2tok = parser.idx_to_token_details
    tok_seq = [id2tok[t] for t in d["codes"][i]]
    print("\n=== Program ==="); print(beautify(" ".join(tok_seq)))
    print("\n=== Input world ==="); Karel(state=inputs[i]).draw()
    print("\n=== Trace frames (including initial) ===")
    for t, st in enumerate(traces[i]):
        print(f"\n--- frame {t} ---"); Karel(state=st).draw()
    print("\n=== Output world ==="); Karel(state=outputs[i]).draw()
