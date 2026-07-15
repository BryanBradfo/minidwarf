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
    "<cuda_runtime.h>; do NOT call cuBLAS, cuSPARSE, Thrust, or any other library. "
    "minidwarf_solve must launch your kernel(s), fully write all outputs, and synchronize "
    "the device before returning. Return ONLY the CUDA code in a single ```cuda fenced block.\n"
    "\n--- PROBLEM ---\n"
)

def resolve_split(split, problems_root=Path("problems"), splits_file=Path("SPLITS.yaml")):
    """Return the problem directories named under SPLITS.yaml[split]."""
    names = set(yaml.safe_load(Path(splits_file).read_text())[split])
    dirs = [p.parent for p in Path(problems_root).glob("*/*/spec.yaml")
            if p.parent.name in names]
    return sorted(dirs, key=lambda d: d.name)

def run_generation(config: EvalConfig, problem_dirs, out_dir, run_id, model: Model | None = None):
    """Generate n_samples kernels per problem; write results.jsonl + kernels/."""
    model = model or create_model(config)
    run_dir = Path(out_dir) / run_id
    (run_dir / "kernels").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w") as jf:
        for pdir in problem_dirs:
            pdir = Path(pdir)
            dwarf = pdir.parent.name
            prompt = SYSTEM_PROMPT + (pdir / "prompt.md").read_text()
            for i in range(config.n_samples):
                res = model.generate(prompt)
                kernel = extract_kernel(res.text)
                kpath = run_dir / "kernels" / f"{pdir.name}__s{i}.cu"
                kpath.write_text(kernel)
                jf.write(json.dumps({
                    "problem": pdir.name, "dwarf": dwarf, "sample": i,
                    "generation_status": res.status, "error": res.error,
                    "kernel_path": str(kpath.relative_to(run_dir)),
                    "raw_response": res.text,
                    "model": config.model_name,
                }) + "\n")
    return run_dir
