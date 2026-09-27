"""Matplotlib figures. No model logic lives here: every function takes results.

Palette: a colour-vision-deficiency-checked categorical order (blue, orange,
aqua, yellow, magenta, green, violet, red), recessive grey chrome, text in ink
colours rather than series colours. Series identity is always also given by a
legend or a direct label.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STATUS = {"Within appetite": "#0ca30c", "Near threshold": "#fab219", "Outside appetite": "#d03b3b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 9, "legend.labelcolor": INK2, "font.family": "DejaVu Sans", "lines.linewidth": 2,
    "figure.dpi": 110,
})


def musd(x, _=None):
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.0f}M" if abs(x) >= 1e7 else f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:.0f}"


MONEY = FuncFormatter(musd)


def _note(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def loss_histogram(losses, metrics: dict, title="Simulated annual loss distribution", ax=None):
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.2))
    x = np.asarray(losses)
    pos = x[x > 0]
    bins = np.geomspace(max(pos.min(), 1e5), pos.max() * 1.05, 70)
    ax.hist(pos, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(MONEY)
    ymax = ax.get_ylim()[1]
    marks = [("Median", metrics["median"]), ("EAL", metrics["eal"]), ("P95", metrics["p95"]), ("P99", metrics["p99"])]
    for i, (lab, v) in enumerate(marks):
        ax.axvline(v, color=INK if lab in ("EAL",) else INK2, lw=1.2, ls="-" if lab == "EAL" else "--")
        ax.text(v * 1.04, ymax * (0.95 - 0.09 * i), f"{lab} {musd(v)}", color=INK, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_xlabel("Annual loss (log scale)")
    ax.set_ylabel("Simulated years")
    ax.set_title(title)
    return fig


def exceedance_curves(curves: dict, title="Loss exceedance curve", ax=None, annotate=(0.05, 0.01)):
    """curves: {label: losses array}."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.5))
    for i, (lab, L) in enumerate(curves.items()):
        xs = np.sort(np.asarray(L))
        p = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
        keep = (xs > 0) & (p > 0)
        ax.plot(xs[keep], p[keep], color=SERIES[i % len(SERIES)], label=lab, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(MONEY)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for a in annotate:
        ax.axhline(a, color=AXIS, lw=0.8)
        ax.text(ax.get_xlim()[0], a, f" 1-in-{int(round(1 / a))} yr", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(bottom=max(1.0 / max(len(L) for L in curves.values()), 1e-5))
    ax.set_xlabel("Annual loss")
    ax.set_ylabel("P(annual loss > x)")
    ax.set_title(title)
    if len(curves) > 1:
        ax.legend(loc="lower left")
    return fig


def exceedance_band(nested, pooled_losses=None, title="Loss exceedance: epistemic uncertainty band"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    g, lec = nested.lec_grid, nested.lec
    lo, med, hi = (np.quantile(lec, q, axis=0) for q in (0.05, 0.5, 0.95))
    ax.fill_between(g, np.maximum(lo, 1e-6), np.maximum(hi, 1e-6), color=SERIES[0], alpha=0.18, lw=0,
                    label="5-95 % of worlds (epistemic)")
    ax.plot(g, np.maximum(med, 1e-6), color=SERIES[0], label="Median world")
    if pooled_losses is not None:
        xs = np.sort(pooled_losses)
        p = 1 - np.arange(1, len(xs) + 1) / len(xs)
        k = (xs > 0) & (p > 0)
        ax.plot(xs[k], p[k], color=SERIES[1], lw=1.6, ls="--", label="Pooled (predictive)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1.0 / nested.n_inner, 1.05)
    ax.set_xlim(g[0], g[-1])
    ax.xaxis.set_major_formatter(MONEY)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Annual loss")
    ax.set_ylabel("P(annual loss > x)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return fig


def epistemic_metric_hist(nested, metric="eal", title=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    v = nested.worlds[metric]
    ax.hist(v, bins=30, color=SERIES[0], edgecolor=SURFACE)
    for q, ls in ((0.05, "--"), (0.5, "-"), (0.95, "--")):
        x = v.quantile(q)
        ax.axvline(x, color=INK2, ls=ls, lw=1.1)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" P{int(q * 100)} {musd(x)}", fontsize=8.5, color=INK, va="top")
    ax.xaxis.set_major_formatter(MONEY)
    ax.set_xlabel(f"{metric.upper()} of a 'world' (one draw of the uncertain parameters)")
    ax.set_ylabel("Worlds")
    ax.set_title(title or f"Epistemic uncertainty in {metric.upper()}")
    return fig


def tornado(df: pd.DataFrame, metric="eal", top=12, title=None, labels=None):
    d = df.sort_values(f"{metric}_swing", ascending=False).head(top).iloc[::-1]
    base = float(d[f"{metric}_base"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    lo, hi = d[f"{metric}_low"].to_numpy(), d[f"{metric}_high"].to_numpy()
    ax.barh(y, lo - base, left=base, color=SERIES[0], height=0.62, label="Parameter at P10")
    ax.barh(y, hi - base, left=base, color=SERIES[1], height=0.62, label="Parameter at P90")
    ax.axvline(base, color=INK, lw=1)
    names = [labels.get(p, p) if labels else p for p in d["parameter"]]
    ax.set_yticks(y, names, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(MONEY)
    ax.set_xlabel(f"{metric.upper()} (other parameters at their medians; base {musd(base)})")
    ax.set_title(title or f"Tornado: one-at-a-time sensitivity of {metric.upper()}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def importance_bars(df: pd.DataFrame, value="S1", label="parameter", title="", xlabel="", top=12, fmt="{:.2f}"):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.3))
    ax.barh(np.arange(len(d)), d[value], color=SERIES[0], height=0.62)
    for i, v in enumerate(d[value]):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(d)), d[label], color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def bayes_update(prior, posterior, likelihood_pdf, grid, title, xlabel, truth=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(grid, prior.pdf(grid), color=SERIES[0], label="Prior")
    ax.plot(grid, likelihood_pdf(grid), color=SERIES[1], ls="--", label="Likelihood (normalised)")
    ax.plot(grid, posterior.pdf(grid), color=SERIES[2], label="Posterior")
    if truth is not None:
        ax.axvline(truth, color=MUTED, lw=1)
        ax.text(truth, ax.get_ylim()[1] * 0.95, " synthetic truth", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return fig


def sequential_posterior(df: pd.DataFrame, title, ylabel, truth=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(df))
    ax.fill_between(x, df["lo"], df["hi"], color=SERIES[0], alpha=0.2, lw=0, label="90 % credible interval")
    ax.plot(x, df["mean"], color=SERIES[0], marker="o", ms=4, label="Posterior mean")
    if truth is not None:
        ax.axhline(truth, color=MUTED, lw=1, ls="--", label="Synthetic truth")
    ax.set_xticks(x, [str(p) for p in df["period"]], rotation=0)
    ax.set_xlabel("Years of records included")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def convergence(df: pd.DataFrame, metrics=("eal", "p95", "p99", "es99")):
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.4), sharex=True)
    for ax, m in zip(axes, metrics):
        for s, g in df.groupby("seed"):
            ax.plot(g["n_years"], g[m], color=AXIS, lw=1, marker="o", ms=3)
        mean = df.groupby("n_years")[m].mean()
        ax.plot(mean.index, mean.values, color=SERIES[0], marker="o", ms=5, label="Mean over seeds")
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(MONEY)
        ax.set_title(m.upper(), fontsize=11)
        ax.set_xlabel("Simulated years")
        ax.set_xticks([10_000, 50_000, 100_000], ["10k", "50k", "100k"])
        ax.minorticks_off()
    axes[0].set_ylabel("Estimate (grey: individual seeds)")
    fig.suptitle("Monte Carlo convergence", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def mitigation_effects(df: pd.DataFrame):
    d = df.sort_values("reduction_eal").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(d) + 1.6), sharey=True)
    y = np.arange(len(d))
    axes[0].barh(y, d["reduction_eal"], xerr=1.96 * d["reduction_eal_se"], color=SERIES[0], height=0.62,
                 error_kw={"ecolor": INK2, "lw": 1})
    axes[0].set_title("EAL reduction (±95 % MC)", fontsize=11)
    axes[1].barh(y, d["reduction_es99"], color=SERIES[1], height=0.62)
    axes[1].set_title("ES99 reduction", fontsize=11)
    axes[2].barh(y, d["annualised_cost_usd"], color=SERIES[2], height=0.62)
    axes[2].set_title("Annualised cost", fontsize=11)
    axes[0].set_yticks(y, d["name"], color=INK2)
    for ax in axes:
        ax.xaxis.set_major_formatter(MONEY)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=AXIS, lw=0.8)
    fig.suptitle("Risk controls: measured reduction vs cost", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def scenario_comparison(df: pd.DataFrame):
    d = df[df["scenario"] != "baseline"].iloc[::-1]
    base = df[df["scenario"] == "baseline"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.hlines(y, d["annual_median"], d["annual_p99"], color=AXIS, lw=2)
    ax.scatter(d["annual_median"], y, color=SERIES[0], s=40, zorder=3, label="Median year")
    ax.scatter(d["annual_eal"], y, color=SERIES[1], s=40, zorder=3, label="Mean (conditional EAL)")
    ax.scatter(d["annual_p99"], y, color=SERIES[6], s=40, zorder=3, label="P99")
    ax.axvline(base["annual_eal"], color=INK2, ls="--", lw=1)
    ax.text(base["annual_eal"], -0.75, f" baseline EAL {musd(base['annual_eal'])}", color=INK2, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(MONEY)
    ax.set_yticks(y, d["title"], color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Annual loss given the scenario (log scale)")
    ax.set_title("Stress scenarios: conditional annual loss", pad=26)
    ax.set_ylim(-0.8, len(d) - 0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5)
    fig.tight_layout()
    return fig


def contribution_bars(df: pd.DataFrame, es_col="share_of_es99"):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    y = np.arange(len(d))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, d["share_of_eal"], height=h, color=SERIES[0], label="Share of EAL (average year)")
    ax.barh(y - h / 2 - 0.02, d[es_col], height=h, color=SERIES[1], label="Share of ES99 (worst 1 % of years)")
    ax.set_yticks(y, [s.replace("_", " ") for s in d["source"]], color=INK2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("What drives the average year vs the tail")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def component_bars(comp_means: pd.Series, direct: list, title="Mean annual loss by component"):
    d = comp_means[comp_means > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(d) + 1.4))
    cols = [SERIES[0] if c in direct else SERIES[1] for c in d.index]
    ax.barh(np.arange(len(d)), d.values, color=cols, height=0.62)
    for i, v in enumerate(d.values):
        ax.text(v, i, " " + musd(v), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(d)), [c.replace("_", " ") for c in d.index], color=INK2)
    ax.xaxis.set_major_formatter(MONEY)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=SERIES[0], label="Direct"), Patch(color=SERIES[1], label="Indirect")], loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def reliability_curves(shape, scale, lam_equiv, age_max=6.0):
    from ..reliability.models import exp_reliability, weibull_hazard, weibull_reliability

    t = np.linspace(0.01, age_max, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    axes[0].plot(t, weibull_reliability(t, shape, scale), color=SERIES[0], label=f"Weibull (β={shape:.1f}, η={scale:.1f} y)")
    axes[0].plot(t, exp_reliability(lam_equiv, t), color=SERIES[1], ls="--", label=f"Exponential (same mean life)")
    axes[0].set_xlabel("Years since overhaul")
    axes[0].set_ylabel("R(t)")
    axes[0].set_title("Reliability of one compressor train", fontsize=11)
    axes[0].legend()
    axes[1].plot(t, weibull_hazard(t, shape, scale), color=SERIES[0], label="Weibull hazard")
    axes[1].axhline(lam_equiv, color=SERIES[1], ls="--", label="Constant hazard")
    axes[1].set_xlabel("Years since overhaul")
    axes[1].set_ylabel("Failures per year")
    axes[1].set_title("Hazard rate: wear-out vs constant", fontsize=11)
    axes[1].legend()
    fig.tight_layout()
    return fig


def fault_tree_importance(imp: pd.DataFrame, title="Fire-protection fault tree: Fussell-Vesely importance"):
    d = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.4))
    ax.barh(np.arange(len(d)), d["fussell_vesely"], color=SERIES[0], height=0.62)
    for i, v in enumerate(d["fussell_vesely"]):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(d)), d["basic_event"], color=INK2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of top-event probability in cut sets containing the event")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def asset_diagram(asset, single_outage: pd.DataFrame | None = None):
    """Left: physical layout (bridges). Right: functional dependency matrix (impact fractions)."""
    pos = {"WHP": (0, 1), "PPA": (1.6, 1), "PPB": (3.2, 1), "UCP": (4.8, 1), "LQ": (6.4, 1), "SWI": (4.8, -0.3)}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    ax.set_axis_off()
    for a, b in asset.bridges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color=AXIS, lw=6, solid_capstyle="round", zorder=1)
    lost = {} if single_outage is None else dict(zip(single_outage["platform"], single_outage["fraction_of_complex"]))
    for c, (x, y) in pos.items():
        ax.add_patch(plt.Rectangle((x - 0.62, y - 0.32), 1.24, 0.64, fc="white", ec=INK2, lw=1.2, zorder=2))
        ax.text(x, y + 0.1, c, ha="center", va="center", fontsize=12, fontweight="bold", color=INK, zorder=3)
        ax.text(x, y - 0.13, asset.names[c].replace(" Platform", "").replace("Production", "Prod.").replace(" / ", "/"),
                ha="center", va="center", fontsize=7, color=INK2, zorder=3)
        q = asset.production_bopd[asset.index(c)]
        if q > 0:
            ax.text(x, y + 0.45, f"{q / 1000:.0f} kbbl/d", ha="center", fontsize=8, color=INK2)
        if c in lost:
            ax.text(x, y - 0.5, f"alone down: −{lost[c]:.0%} output", ha="center", fontsize=7.5, color=INK)
    ax.set_xlim(-0.8, 7.2)
    ax.set_ylim(-0.9, 1.7)
    ax.set_title("Layout: bridge links (fire/explosion escalation paths)")

    codes = asset.codes
    M = np.zeros((len(codes), len(codes)))
    for d in asset.dependencies:
        M[codes.index(d["platform"]), codes.index(d["provider"])] = d["impact"]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    ax2.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(codes)):
        for j in range(len(codes)):
            if M[i, j] > 0:
                ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 0.5 else INK)
    ax2.set_xticks(range(len(codes)), codes)
    ax2.set_yticks(range(len(codes)), codes)
    ax2.set_xlabel("Provider (if unavailable…)")
    ax2.set_ylabel("Dependent platform")
    ax2.grid(False)
    ax2.set_title("…fraction of dependent's capability lost")
    fig.tight_layout()
    return fig


def appetite_table_fig(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(df) + 0.9))
    ax.set_axis_off()
    for i, r in enumerate(df.iloc[::-1].itertuples()):
        y = i
        w = min(r.utilisation, 1.6)
        ax.barh(y, w, color=STATUS[r.status], height=0.55)
        ax.text(w + 0.03, y, f"{r.status} ({r.utilisation:.0%} of limit)", va="center", fontsize=8.5, color=INK)
        ax.text(-0.03, y, r.description, va="center", ha="right", fontsize=8.5, color=INK2)
    ax.axvline(1.0, color=INK, lw=1)
    ax.axvline(0.8, color=MUTED, lw=0.8, ls="--")
    ax.text(1.0, len(df) - 0.45, "limit", ha="center", fontsize=8, color=INK)
    ax.text(0.8, len(df) - 0.45, "80 %", ha="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.6, len(df) - 0.2)
    ax.set_title("Risk appetite status (hypothetical limits)", loc="left")
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
