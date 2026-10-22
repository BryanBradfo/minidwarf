# Pilot: qwen2.5-coder 1.5B / 3B on MiniDwarf v3 (valid split)

Date: 2026-10-21 · Harness v3 (`main` @ 760acf4 + `feature/pilot-fixes`) · 1 sample per problem · RTX 5070 Laptop.
Purpose: (1) check the v3 harness for false positives on real LLM output, (2) open-code real failures before
fixing the failure taxonomy of sub-project 2.

## Harness findings (fixed on `feature/pilot-fixes`)

- **Extractor false negatives.** 4/12 of the 1.5B "compile errors" came from our extractor, not the model:
  the model parroted the system prompt and opened its block with "```cuda fenced block"; the old fence
  regex only accepted a bare language tag, so the fence line leaked into the source. Fixed (info string =
  first word; unclosed fences from truncated responses also handled).
- **Diagnostics were discarded.** `ProblemResult` kept no compiler/runtime message, which the taxonomy
  needs. Now stored in `ProblemResult.error` (truncated to 4000 chars).
- No `forbidden_api` false positive on the 24 real kernels.

## Results (after the fixes)

| model | ok | wrong_output | compile_error |
|---|---|---|---|
| qwen2.5-coder-1.5b | 0 | 2 | 10 |
| qwen2.5-coder-3b | 2 | 2 | 8 |

Correct kernels were slow (spmm_csr 0.05×, transpose 0.88× of baseline). One wrong kernel was fast
(spmv_csr 3B: 5.97× and wrong on 24/28 checks): speed without correctness, caught by every-call
verification.

## Open coding of the 18 compile errors (each checked: one fenced block per response)

| code | count | example |
|---|---|---|
| use-before-declaration (kernel/helper defined after the host code that calls it) | 5 | `minidwarf_kernel` called in `minidwarf_solve`, defined below |
| missing definition (called, never written) | 5 | `gemv`, `sgemm`, `transpose` |
| missing standard header (`stderr`/`fprintf` without `<cstdio>`) | 4 | coulomb_1d, nbody_potential |
| const-correctness / type errors on the `void*` ABI | 3 | casting `const void*` to `float*` |
| undefined math helper | 1 | `dot` |

## Implications for sub-project 2 (failure taxonomy)

- Small models fail mostly **before** the HPC pattern matters: 18/24 kernels never compile, and the dominant
  causes (declaration order, incomplete code, headers, const-correctness on the ABI) are generic C++ errors,
  not dwarf-specific. A dwarf-level taxonomy needs models that compile most of the time → frontier models.
- The taxonomy needs a **compile-error layer** (the codes above, assignable from nvcc diagnostics with
  regexes) **below** the semantic layer (indexing, races, reductions, performance).
- The const-correctness errors suggest the `void*` ABI itself is a stumbling block; worth reporting
  separately from the computational pattern (and possibly worth a prompt clarification in a later version).
- Wrong outputs fail on **both** check and eval shapes here (not only tails), i.e. logic errors, not
  boundary errors — the check/eval split is informative and should be kept per failure.
