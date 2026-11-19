"""Figures of the article (vector PDF + PNG preview).

Usage: python -I make_figures.py <analysis_dir> <repo_results_dir> <fig_dir>
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

AN, RES, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(FIG, exist_ok=True)
X = json.load(open(os.path.join(AN, "extra_analysis.json")))
P = json.load(open(os.path.join(RES, "analysis_primary_40.json")))

# Reference categorical order (light mode), slots 1..5
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})
ROWS = [("qwen2.5:3b", "A0", "3B  A0 free"), ("qwen2.5:3b", "A1", "3B  A1 schema"), ("qwen2.5:3b", "A2", "3B  A2 grammar"),
        ("qwen2.5:7b", "A0", "7B  A0 free"), ("qwen2.5:7b", "A1", "7B  A1 schema"), ("qwen2.5:7b", "A2", "7B  A2 grammar")]


def text_on(color):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


def stacked(ax, rows, parts, colors, labels, as_pct, min_label):
    ys = list(range(len(rows)))[::-1]
    for y, vals in zip(ys, parts):
        tot = sum(vals)
        left = 0.0
        for v, c in zip(vals, colors):
            w = 100 * v / tot if as_pct else v
            if w > 0:
                ax.barh(y, w, left=left, height=0.62, color=c, edgecolor=SURF, linewidth=1.2)
                if w >= min_label:
                    ax.text(left + w / 2, y, f"{w:.0f}" + ("%" if as_pct else ""), ha="center", va="center",
                            fontsize=7.5, color=text_on(c))
            left += w
        if not as_pct:
            ax.text(left + 1.5, y, f"{tot}  ({', '.join(str(v) for v in vals)})", va="center", ha="left",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(rows)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(labels), frameon=False,
              fontsize=7.5, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# Figure: provenance of identifiers behind "not found" errors (primary 40)
parts = []
for m, a, _ in ROWS:
    p = X[m]["primary_40"][a]["provenance"]
    ai, ph = p.get("agent_introduced", 0), p.get("agent_introduced_placeholder", 0)
    parts.append([ph, ai - ph, p.get("customer_unfaithful", 0), p.get("customer_faithful", 0),
                  p.get("format", 0) + p.get("real_wrong_place", 0) + p.get("tool_output", 0)])
fig, ax = plt.subplots(figsize=(6.3, 2.35))
stacked(ax, [r[2] + f"  (n={sum(v)})" for r, v in zip(ROWS, parts)], parts, S,
        ["Agent, placeholder", "Agent, other", "Customer, not in instruction",
         "Customer, in instruction", "Format slip or misplaced ID"], True, 6)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of identifiers behind “not found” errors, by who introduced them (%)")
save(fig, "fig_provenance")

# Figure: fate of write calls that reached execution (primary 40)
parts = []
for m, a, _ in ROWS:
    d = P[m][a]
    correct = d["gate_total_writes"] - d["gate_wrong_writes"]
    parts.append([correct, d["gate_wrong_executed"], d["gate_wrong_rejected_by_shop"]])
fig, ax = plt.subplots(figsize=(6.3, 2.2))
stacked(ax, [r[2] for r in ROWS], parts, S[:3],
        ["Correct (matches a gold write)", "Wrong, executed", "Wrong, rejected by the shop"], False, 4)
ax.set_xlim(0, 135)
ax.set_xlabel("Write calls that reached execution (count, 40 episodes per row)")
save(fig, "fig_writes")

# Figure: per-episode funnel on the 38 tasks that require a state change
stages = [("identified", "Customer\nidentified"), ("identified_and_one_correct_write", "One correct\nwrite"),
          ("database_correct", "Database\ncorrect"), ("success", "Success")]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
for ax, model, title in zip(axes, ("qwen2.5:3b", "qwen2.5:7b"), ("Qwen2.5-3B", "Qwen2.5-7B")):
    w = 0.26
    for i, (arm, lab) in enumerate((("A0", "A0 free"), ("A1", "A1 schema"), ("A2", "A2 grammar"))):
        f = X[model]["primary_40"][arm]["funnel"]
        vals = [f.get(k, 0) for k, _ in stages]
        xs = [j + (i - 1) * w for j in range(len(stages))]
        ax.bar(xs, vals, width=w - 0.03, color=S[i], edgecolor=SURF, linewidth=0.8, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.6, str(v), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s for _, s in stages], fontsize=7.5)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_ylim(0, 40)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Episodes (of 38)")
axes[1].legend(loc="upper right", frameon=False, fontsize=7.5)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_funnel")
print("figures written to", FIG)
