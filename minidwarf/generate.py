# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import yaml
from .evalconfig import EvalConfig
from .extract import extract_kernel
from .models.base import Model
from .models.registry import create_model

SYSTEM_PROMPT = (
    "You are writing a CUDA kernel for a benchmark. Output a single, self-contained "
    "CUDA source file implementing exactly this C-linkage entry point:\n"
    '    extern "C" void minidwarf_solve(const void* const* inputs, void* const* outputs, '
    "const long* dims, int n_dims);\n"
    "Cast each input/output pointer to the element type documented below. Use only "
    "<cuda_runtime.h> plus standard C++ headers; do NOT use cuBLAS, cuSPARSE, cuFFT, cuRAND, "
    "cuSOLVER, Thrust, CUB or any other library unless the problem says otherwise. No file I/O, "
    "process spawning, environment access, dynamic loading, inline host syscalls, constructor "
    "attributes or raw string literals (these are rejected). minidwarf_solve must launch your "
    "kernel(s), fully write all outputs, and synchronize the device before returning; it is "
    "called many times and every call is checked, so it must not depend on being called a "
    "particular number of times or in a particular order. "
    "Return ONLY the CUDA code in a single ```cuda fenced block.\n"
    "\n--- PROBLEM ---\n"
)

def resolve_split(split, problems_root=Path("problems"), splits_file=Path("SPLITS.yaml")):
    """Return the problem directories named under SPLITS.yaml[split]."""
    names = set(yaml.safe_load(Path(splits_file).read_text())[split])
    dirs = [p.parent for p in Path(problems_root).glob("*/*/spec.yaml")
            if p.parent.name in names]
    return sorted(dirs, key=lambda d: d.name)

def _is_daily_quota(res) -> bool:
    """Errors that no retry today can fix: daily quota exhausted, or the model is gone/forbidden."""
    e = (res.error or "")
    return res.status == "error" and ("PerDay" in e or "per day" in e.lower()
                                      or any(f"Error code: {c}" in e for c in (401, 403, 404)))

def _write_rows(path, rows):
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); tmp.replace(path)

def run_generation(config: EvalConfig, problem_dirs, out_dir, run_id, model: Model | None = None, resume=False):
    """Generate n_samples kernels per problem; write results.jsonl + kernels/.

    resume=True keeps existing successful samples and only (re)generates missing or failed ones.
    Results are rewritten after every sample, and generation stops at a daily-quota error (nothing is
    recorded for that sample, so a later resume picks it up)."""
    model = model or create_model(config)
    run_dir = Path(out_dir) / run_id
    (run_dir / "kernels").mkdir(parents=True, exist_ok=True)
    path = run_dir / "results.jsonl"
    done = {}
    if resume and path.exists():
        for l in path.read_text().splitlines():
            r = json.loads(l)
            if r.get("generation_status") == "success": done[(r["problem"], r["sample"])] = r
    rows = []
    for pdir in problem_dirs:
        pdir = Path(pdir)
        prompt = SYSTEM_PROMPT + (pdir / "prompt.md").read_text()
        for i in range(config.n_samples):
            if (pdir.name, i) in done:
                rows.append(done[(pdir.name, i)]); continue
            res = model.generate(prompt)
            if _is_daily_quota(res):
                rows += [r for (p, s), r in done.items() if r not in rows]
                _write_rows(path, rows); return run_dir
            kpath = run_dir / "kernels" / f"{pdir.name}__s{i}.cu"
            kpath.write_text(extract_kernel(res.text))
            rows.append({"problem": pdir.name, "dwarf": pdir.parent.name, "sample": i,
                         "generation_status": res.status, "error": res.error,
                         "kernel_path": str(kpath.relative_to(run_dir)),
                         "raw_response": res.text, "model": config.model_name})
            _write_rows(path, rows + [r for r in done.values() if r not in rows])
    _write_rows(path, rows)
    return run_dir
