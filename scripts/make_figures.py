# SPDX-License-Identifier: Apache-2.0
"""Paper figures from runs/*/scores.json (harness v3).

  python scripts/make_figures.py [--runs-dir runs] [--out paper/figures]

Fig. 1 overview (correct and fast_p@1 per model), Fig. 2 fast_p@p curves, Fig. 3 per-dwarf heatmap,
Fig. 4 outcome breakdown over every sample. Writes .pdf (vector, for LaTeX) and .png (300 dpi).
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from minidwarf.score import HARNESS_VERSION, load_noise_floor

# Reference categorical palette, fixed slot order (validated light: CVD dE >= 9.1, normal dE >= 19.6);
# slots 3-5 are below 3:1 on the surface, so every series also gets a direct label and a marker.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
OUTCOMES = ["ok", "wrong_output", "runtime_error", "timeout", "forbidden_api", "compile_error"]
# fixed colour per outcome (palette slots), chosen so no error class wears green or the "correct" blue
OUTCOME_COLORS = {"ok": SERIES[0], "wrong_output": SERIES[1], "runtime_error": SERIES[3], "timeout": SERIES[4],
                  "forbidden_api": SERIES[6], "compile_error": SERIES[7]}
OUTCOME_LABELS = {"ok": "correct", "wrong_output": "wrong output", "runtime_error": "runtime error",
                  "timeout": "timeout", "forbidden_api": "forbidden API", "compile_error": "compile error"}
DWARFS = ["dense", "sparse", "nbody", "structured_grids"]
DWARF_LABELS = {"dense": "Dense LA", "sparse": "Sparse LA", "nbody": "N-body", "structured_grids": "Structured\ngrids"}
P_GRID = [round(0.05 * i, 2) for i in range(101)]  # 0 .. 5

def collect(runs_dir):
    """The most complete current-harness scores.json per model (most graded problems, then newest)."""
    best = {}
    for sj in Path(runs_dir).glob("*/scores.json"):
        d = json.loads(sj.read_text())
        if d.get("harness_version") != HARNESS_VERSION or not d.get("per_problem"): continue
        key = (len(d["per_problem"]), sj.stat().st_mtime)
        if d["model"] not in best or key > best[d["model"]][0]: best[d["model"]] = (key, d)
    return [d for _, d in sorted(best.values(), key=lambda t: t[1]["model"])]

def summarize_run(scores, eps_global, per):
    rows = scores["per_problem"]; n = len(rows)
    ok = lambda r, p: r["correct"] and (r.get("speedup") or 0) >= p * (1 + per.get(r["name"], eps_global))
    samples = [s for r in rows for s in r.get("samples") or [r]]
    by_dwarf = defaultdict(list)
    for r in rows: by_dwarf[r["dwarf"]].append(r["correct"])
    return {"model": scores["model"], "n_problems": n,
            "correct": sum(r["correct"] for r in rows) / n,
            "pass1": sum(sum(s["correct"] for s in r.get("samples") or [r]) / len(r.get("samples") or [r]) for r in rows) / n,
            "fast_p": {p: sum(ok(r, p) for r in rows) / n for p in P_GRID},
            "by_dwarf": {d: sum(v) / len(v) for d, v in by_dwarf.items()},
            "outcomes": dict(Counter(s["status"] for s in samples))}

def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "pdf.fonttype": 42, "savefig.bbox": "tight"})

def _save(fig, out, name):
    for ext in ("pdf", "png"): fig.savefig(out / f"{name}.{ext}", dpi=300)

def _label(s): return f"{s['model']} (n={s['n_problems']})"

def fig_overview(plt, S, out):
    S = sorted(S, key=lambda s: (s["correct"], s["fast_p"][1.0]))
    fig, ax = plt.subplots(figsize=(3.4, 0.38 * len(S) + 0.9))
    h = 0.36
    for i, s in enumerate(S):
        for j, (key, val) in enumerate((("correct (best-of-n)", s["correct"]), ("fast_p@1", s["fast_p"][1.0]))):
            y = i + (h / 2 + 0.02) * (1 if j == 0 else -1)
            ax.barh(y, 100 * val, height=h, color=SERIES[j], label=key if i == 0 else None, edgecolor=SURFACE, linewidth=0.8)
            ax.text(100 * val + 1.5, y, f"{100 * val:.0f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(S)), [_label(s) for s in S]); ax.set_xlim(0, 112); ax.set_xlabel("% of problems")
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    _save(fig, out, "fig1_overview"); plt.close(fig)

def fig_fastp(plt, S, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, s in enumerate(S):
        ys = [100 * s["fast_p"][p] for p in P_GRID]
        ax.plot(P_GRID, ys, color=SERIES[i % 8], linewidth=1.5, label=_label(s))
        marks = [0, 20, 40, 60, 80, 100]
        ax.plot([P_GRID[k] for k in marks], [ys[k] for k in marks], linestyle="none", marker=MARKERS[i % 8],
                markersize=4.5, color=SERIES[i % 8], markeredgecolor=SURFACE, markeredgewidth=0.6)
    ax.axvline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)))
    ax.text(1.06, 3, "baseline\nspeed", fontsize=6.5, color=INK2, va="bottom")
    ax.set_xlim(0, 5); ax.set_ylim(0, 105); ax.set_xlabel("speedup threshold p (×, noise-floor adjusted)")
    ax.set_ylabel("fast_p (% of problems)"); ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _save(fig, out, "fig2_fast_p"); plt.close(fig)

def fig_dwarfs(plt, S, out):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP); cmap.set_bad("#f0efec")
    import numpy as np
    M = np.array([[s["by_dwarf"].get(d, np.nan) for d in DWARFS] for s in S], dtype=float)
    fig, ax = plt.subplots(figsize=(3.4, 0.36 * len(S) + 0.9))
    im = ax.imshow(np.ma.masked_invalid(100 * M), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(S)):
        for j in range(len(DWARFS)):
            v = M[i, j]
            txt, col = ("n/a", INK2) if np.isnan(v) else (f"{100 * v:.0f}", "#ffffff" if v > 0.55 else INK)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)
    ax.set_xticks(range(len(DWARFS)), [DWARF_LABELS[d] for d in DWARFS]); ax.set_yticks(range(len(S)), [_label(s) for s in S])
    ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks(np.arange(-.5, len(DWARFS)), minor=True); ax.set_yticks(np.arange(-.5, len(S)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.05); cb.set_label("% correct (best-of-n)", color=INK2); cb.outline.set_visible(False)
    _save(fig, out, "fig3_dwarfs"); plt.close(fig)

def fig_outcomes(plt, S, out):
    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(S) + 0.9))
    for i, s in enumerate(S):
        tot = sum(s["outcomes"].values()); left = 0.0
        for k, o in enumerate(OUTCOMES):
            w = 100 * s["outcomes"].get(o, 0) / tot
            if w == 0: continue
            ax.barh(i, w, left=left, height=0.62, color=OUTCOME_COLORS[o], edgecolor=SURFACE, linewidth=1.0,
                    label=OUTCOME_LABELS[o] if o not in ax.get_legend_handles_labels()[1] else None)
            if w >= 9: ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=6.5,
                              color="#ffffff" if o in ("ok", "compile_error", "forbidden_api") else INK)
            left += w
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(OUTCOME_LABELS[o]) for o in OUTCOMES if OUTCOME_LABELS[o] in labels]
    ax.legend([handles[k] for k in order], [labels[k] for k in order], ncol=6, fontsize=6.5,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_yticks(range(len(S)), [f"{s['model']} ({sum(s['outcomes'].values())} samples)" for s in S])
    ax.set_xlim(0, 100); ax.set_xlabel("% of graded samples"); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _save(fig, out, "fig4_outcomes"); plt.close(fig)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs"); ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args(argv)
    runs = collect(a.runs_dir)
    if not runs:
        print(f"no harness-v{HARNESS_VERSION} scores under {a.runs_dir}", file=sys.stderr); return 2
    eps_global, per, _ = load_noise_floor()
    S = [summarize_run(r, eps_global, per) for r in runs]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_overview(plt, S, out); fig_fastp(plt, S, out); fig_dwarfs(plt, S, out); fig_outcomes(plt, S, out)
    print("\n".join(f"{s['model']:24s} n={s['n_problems']:2d} correct={100*s['correct']:5.1f}% "
                    f"pass@1={100*s['pass1']:5.1f}% fast_p@1={100*s['fast_p'][1.0]:5.1f}%" for s in S))
    print(f"figures written to {out}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
