# MiniDwarf Leaderboard

| Model | compile% | correct% | fast_p@1 | fast_p@2 | fast_p@5 |
|---|---|---|---|---|---|

## Per-dwarf breakdown

| Model | Dwarf | correct% | fast_p@1 |
|---|---|---|---|

`fast_p@p`: share of problems solved correctly with speedup >= p * (1 + eps), where eps is that problem's own timing noise floor (eps_global = 0.022 for unmeasured problems, eps_max = 0.222; harness v3).
Unstable timing: sddmm (eps 0.22), spmm_csr (eps 0.04) — fast_p for these is reported but low-confidence.
_Skipped 2 run(s) scored with another harness version._
