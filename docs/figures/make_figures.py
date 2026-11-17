#!/usr/bin/env python3
"""Render the figures used in README.md and docs/.

    python3 docs/figures/make_figures.py                      # draw from committed data
    python3 docs/figures/make_figures.py --collect live_runs  # refresh docs/data from local runs

The track and raceline figures read the maps and trajectories in
f1tenth_planning/. The closed-loop figures read the small CSV extracts in
docs/data/, which --collect rebuilds from the trial directories written by
tools/sim/run_trial.sh.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]
PLANNING = ROOT / "f1tenth_planning"
DATA = ROOT / "docs/data"
OUT = ROOT / "docs/images"

TRACKS = {
    "competition": dict(
        title="Competition track",
        map=PLANNING / "maps/competition",
        raceline=PLANNING / "trajectories/competition/raceline.csv",
        reference_s=6.925,
    ),
    "practice": dict(
        title="Practice track",
        map=PLANNING / "maps/practice",
        raceline=PLANNING / "trajectories/practice/raceline.csv",
        reference_s=4.800,
    ),
}

# Final-configuration validation runs: sweep_final_config_N (competition) and
# sweep_pracfinal_config_N (practice). Closed-loop traces come from one complete run of each.
RUN_GLOBS = {"competition": "sweep_final_config_*", "practice": "sweep_pracfinal_config_*"}
TRACE_RUN = {"competition": "sweep_final_config_2", "practice": "sweep_pracfinal_config_0"}

INK = "#1f2933"
MUTED = "#6b7785"
ACCENT = "#d1495b"
BLUE = "#2e6f95"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 100,
})


def read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    parts = raw.split(maxsplit=4)
    width, height = int(parts[1]), int(parts[2])
    return np.frombuffer(parts[4], dtype=np.uint8, count=width * height).reshape(height, width)


def read_map(stem: Path):
    meta = dict(line.split(":", 1) for line in stem.with_suffix(".yaml").read_text().splitlines() if ":" in line)
    resolution = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"].strip(" []").split(",")]
    image = read_pgm(stem.with_suffix(".pgm"))
    extent = (origin[0], origin[0] + image.shape[1] * resolution,
              origin[1], origin[1] + image.shape[0] * resolution)
    return image, extent


def read_raceline(path: Path) -> dict[str, np.ndarray]:
    names = ["s", "x", "y", "psi", "kappa", "v", "a", "d_left", "d_right"]
    data = np.loadtxt(path, delimiter=",", comments="#", usecols=range(len(names)))
    return dict(zip(names, data.T))


def draw_track(ax, spec, vmax):
    image, extent = read_map(spec["map"])
    shade = np.where(image < 100, 0.15, np.where(image > 240, 1.0, 0.82))
    ax.imshow(shade, cmap="gray", vmin=0, vmax=1, extent=extent, origin="upper", interpolation="nearest")
    line = read_raceline(spec["raceline"])
    points = np.column_stack([line["x"], line["y"]]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", norm=plt.Normalize(2, vmax), linewidth=2.4)
    lc.set_array(line["v"][:-1])
    ax.add_collection(lc)
    ax.plot(line["x"][0], line["y"][0], marker="o", color=INK, markersize=5, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    length = line["s"][-1]
    ax.set_title(f"{spec['title']}\n{length:.1f} m lap, {spec['reference_s']:.3f} s reference",
                 fontsize=10.5, color=INK)
    return lc


def figure_tracks():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.4), gridspec_kw={"width_ratios": [1, 1.15]})
    vmax = 12.0
    for ax, key in zip(axes, ("practice", "competition")):
        lc = draw_track(ax, TRACKS[key], vmax)
    cbar = fig.colorbar(lc, ax=axes, orientation="horizontal", fraction=0.045, pad=0.03, aspect=45)
    cbar.set_label("reference speed (m/s)")
    cbar.outline.set_visible(False)
    fig.savefig(OUT / "tracks.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_profiles():
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.2), sharex=False)
    for ax, key in zip(axes, ("competition", "practice")):
        line = read_raceline(TRACKS[key]["raceline"])
        ax.fill_between(line["s"], 0, line["v"], color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(line["s"], line["v"], color=BLUE, linewidth=1.6)
        ax2 = ax.twinx()
        ax2.plot(line["s"], np.abs(line["kappa"]), color=ACCENT, linewidth=0.9, alpha=0.85)
        ax2.set_ylabel("|curvature| (1/m)", color=ACCENT)
        ax2.tick_params(axis="y", colors=ACCENT)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(ACCENT)
        ax.set_ylabel("speed (m/s)")
        ax.set_ylim(0, 13)
        ax.set_xlim(0, line["s"][-1])
        ax.set_title(TRACKS[key]["title"], loc="left", fontsize=10, color=INK)
    axes[-1].set_xlabel("distance along the lap (m)")
    fig.tight_layout()
    fig.savefig(OUT / "speed_profiles.png", dpi=170, facecolor="white")
    plt.close(fig)


def load_trace(key):
    rows = list(csv.DictReader((DATA / f"closed_loop_{key}.csv").open()))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def figure_closed_loop(key):
    trace = load_trace(key)
    ref = read_raceline(TRACKS[key]["raceline"])
    length = ref["s"][-1]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.6), sharex=True)
    laps = np.unique(trace["lap"]).astype(int)
    for lap in laps:
        m = trace["lap"] == lap
        s = trace["s"][m]
        gap = np.where(np.diff(s) < 0)[0] + 1  # the lap crosses 0 m: break the line there
        s = np.insert(s, gap, np.nan)
        axes[0].plot(s, np.insert(trace["u"][m], gap, np.nan), color=BLUE, alpha=0.3, linewidth=0.8)
        axes[1].plot(s, np.insert(trace["e_y"][m] * 100, gap, np.nan), color=BLUE, alpha=0.3, linewidth=0.8)
    axes[0].plot(ref["s"], ref["v"], color=INK, linewidth=1.4, label="reference")
    axes[0].plot([], [], color=BLUE, linewidth=1.2, label=f"measured, {len(laps)} laps")
    axes[0].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    axes[0].set_ylabel("speed (m/s)")
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].set_ylabel("lateral error (cm)")
    bins = np.arange(0, length + 0.5, 0.5)
    index = np.digitize(trace["s"], bins)
    centers, mean_iter = [], []
    for i in range(1, len(bins)):
        sel = index == i
        if sel.any():
            centers.append(0.5 * (bins[i - 1] + bins[i]))
            mean_iter.append(trace["iterations"][sel].mean())
    axes[2].fill_between(centers, 0, mean_iter, color=ACCENT, alpha=0.25, linewidth=0)
    axes[2].plot(centers, mean_iter, color=ACCENT, linewidth=1.4)
    axes[2].set_ylabel("SQP iterations\nper 25 ms cycle")
    axes[2].set_xlabel("distance along the lap (m)")
    axes[2].set_xlim(0, length)
    axes[0].set_title(f"{TRACKS[key]['title']}: closed loop in the simulator", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"closed_loop_{key}.png", dpi=170, facecolor="white")
    plt.close(fig)


def figure_lap_times():
    rows = list(csv.DictReader((DATA / "lap_times.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax, key in zip(axes, ("practice", "competition")):
        times = np.array([float(r["time_s"]) for r in rows if r["track"] == key])
        ref = TRACKS[key]["reference_s"]
        lo, hi = np.floor(times.min() * 50) / 50, np.ceil(times.max() * 50) / 50
        ax.hist(times, bins=np.arange(lo, hi + 0.02, 0.02), color=BLUE, alpha=0.85, edgecolor="white")
        ax.axvline(ref, color=ACCENT, linewidth=1.6)
        ax.text(ref, ax.get_ylim()[1] * 0.97, f" reference {ref:.3f} s", color=ACCENT, va="top", ha="left", fontsize=9)
        ax.axvline(times.mean(), color=INK, linewidth=1.0, linestyle="--")
        ax.set_title(f"{TRACKS[key]['title']}: mean {times.mean():.3f} s, best {times.min():.3f} s",
                     fontsize=10, loc="left", color=INK)
        ax.set_xlabel(f"lap time (s), {len(times)} scored laps")
        ax.set_ylabel("laps")
    fig.tight_layout()
    fig.savefig(OUT / "lap_times.png", dpi=170, facecolor="white")
    plt.close(fig)


def collect(live_runs: Path):
    DATA.mkdir(parents=True, exist_ok=True)
    lap_rows = []
    for key, pattern in RUN_GLOBS.items():
        for run in sorted(live_runs.glob(pattern)):
            log = run / "watcher.log"
            if not log.exists():
                continue
            for lap, seconds in re.findall(r"LAP_TIME lap=(\d+) time_s=([\d.]+)", log.read_text(errors="ignore")):
                if int(lap) >= 2:  # lap 1 is the warm-up lap
                    lap_rows.append((key, run.name, int(lap), float(seconds)))
    with (DATA / "lap_times.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track", "run", "lap", "time_s"])
        writer.writerows(lap_rows)
    for key, run_name in TRACE_RUN.items():
        length = read_raceline(TRACKS[key]["raceline"])["s"][-1]
        cycles = list(csv.DictReader((live_runs / run_name / "analysis/cycles.csv").open()))
        samples = [(float(r["progress_m"]), float(r["u_mps"]), float(r["e_y_m"]),
                    float(r["sqp_iterations_attempted"] or 0))
                   for r in cycles if r["score_eligible"] == "True" and r["progress_m"] and r["u_mps"]]
        start = min(s[0] for s in samples)
        # Laps are counted from the start position; lap 0 is the warm-up lap and is left out.
        rows = [(int((p - start) // length), p % length, u, ey, it)
                for p, u, ey, it in samples if p - start >= length]
        first = 1
        with (DATA / f"closed_loop_{key}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["lap", "s", "u", "e_y", "iterations"])
            for lap, s, u, ey, it in rows:
                writer.writerow([lap - first, f"{s:.3f}", f"{u:.3f}", f"{ey:.4f}", int(it)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", type=Path, metavar="LIVE_RUNS")
    args = parser.parse_args()
    if args.collect:
        collect(args.collect)
    OUT.mkdir(parents=True, exist_ok=True)
    figure_tracks()
    figure_profiles()
    figure_lap_times()
    for key in TRACKS:
        figure_closed_loop(key)


if __name__ == "__main__":
    main()
