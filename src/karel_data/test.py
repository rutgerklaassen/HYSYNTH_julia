import numpy as np

# import your modules
from karel import KarelForSynthesisParser          # to map token IDs back to tokens :contentReference[oaicite:0]{index=0}
from karel.karel import Karel                      # to reconstruct and draw worlds from a state tensor :contentReference[oaicite:1]{index=1}
from karel.utils import beautify                   # to pretty-print synthesis tokens as curly-brace code :contentReference[oaicite:2]{index=2}

# 1) Load one split (adjust path if needed)
data = np.load("data_traces/val.npz", allow_pickle=True)  # traces/codes are ragged → allow_pickle=True :contentReference[oaicite:3]{index=3}

inputs   = data["inputs"]     # (H, W, C) state tensors
outputs  = data["outputs"]    # (H, W, C) state tensors
codes    = data["codes"]      # object array of variable-length int lists (token IDs) :contentReference[oaicite:4]{index=4}
lengths  = data["code_lengths"]
traces   = data["traces"]     # object array: list of per-step (H,W,C) states, includes initial state :contentReference[oaicite:5]{index=5}

# pick an example to display
i = 999

# 2) Reconstruct and pretty-print the program
parser = KarelForSynthesisParser()
id2tok = parser.idx_to_token_details               # index → token (e.g., MOVE, IF, m(, R=3, …) :contentReference[oaicite:6]{index=6}
tok_seq = [id2tok[t] for t in codes[i]]
program_raw = " ".join(tok_seq)
program_pretty = beautify(program_raw)             # converts m(…)/c(…)/r(…) to { } and ( ) and formats nicely :contentReference[oaicite:7]{index=7}
print("=== Program ===")
print(program_pretty)

# 3) Show the input world (ASCII)
print("\n=== Input world ===")
Karel(state=inputs[i]).draw()                      # parse_state + draw() renders ASCII grid with hero/markers/walls :contentReference[oaicite:8]{index=8}

# 4) Step through the trace frames
print("\n=== Trace frames (including initial) ===")
for t, st in enumerate(traces[i]):
    print(f"\n--- frame {t} ---")
    Karel(state=st).draw()

# 5) Show the output world
print("\n=== Output world ===")
Karel(state=outputs[i]).draw()
  
