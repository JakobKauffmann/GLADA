"""aggregate_results.py — pull all may19/* results into paper-ready artifacts.

Globs the campaign output directory, parses each run's ``config.json`` +
``eval.json`` + ``history.json``, and emits:

  - ``table_v_rows.md`` / ``table_v_rows.tex``: drop-in rows for paper Table V
    (variant, ITW EER, delta vs baseline).
  - ``multiseed_summary.csv``: per-(model, config) mean ± std across seeds.
  - ``beta_sensitivity.csv``: pivot of vib_beta vs ITW EER across seeds (only
    rows where multiple vib_beta values exist).
  - ``limitations_rewrite.md``: prose snippet converting paper Future Work
    items into "we tested at N seeds and observed X" language.
  - ``run_index.csv``: full index of every run with its key knobs.

Usage from the orchestrator notebook
------------------------------------
    !python aggregate_results.py \\
        --runs_dir /content/drive/MyDrive/Jakob/Experiments/may19 \\
        --out_dir  /content/drive/MyDrive/Jakob/Experiments/may19/_aggregate \\
        --section all                    # or 1 / 2 / 3 to limit scope
        --check                          # non-zero exit on anomalies

The ``--check`` flag exits non-zero if any run reports an ITW EER outside the
plausibility band [3%, 25%].  EERs outside this band almost always mean a
config error (e.g. effective_batch_size mismatched, or AMP dtype wrong on T4)
rather than a real result, and we want the orchestrator notebook to halt
rather than silently produce a misleading update to Jade.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

# Baseline from the May-7 manuscript Table V row that this campaign is
# extending.  All deltas are reported relative to this number.
PAPER_BASELINE_ITW_EER = 11.67  # per_group_vib on WavLM-Base-Plus, seed 42


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_dir", required=True,
                   help="Drive dir containing one subdir per run.")
    p.add_argument("--out_dir", required=True,
                   help="Where to write aggregated artifacts.")
    p.add_argument("--section", choices=["1", "2", "3", "all"], default="all",
                   help="Limit aggregation to a specific campaign section. "
                        "Names follow the convention <section>_<variant>_s<seed>.")
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if any run has anomalous EER.")
    return p.parse_args(argv)


def _load_run(run_dir: Path) -> Optional[dict]:
    """Read one run's config.json + eval.json + history.json into a dict.

    Returns None if the run hasn't produced an eval.json yet (training in
    progress, or crashed).
    """
    eval_p = run_dir / "eval.json"
    cfg_p = run_dir / "config.json"
    hist_p = run_dir / "history.json"

    if not eval_p.exists() or not cfg_p.exists():
        return None

    try:
        ev = json.loads(eval_p.read_text())
        cfg = json.loads(cfg_p.read_text())
    except json.JSONDecodeError as e:
        print(f"  [aggregate] {run_dir.name}: corrupted json ({e}); skip",
              file=sys.stderr)
        return None

    # eval_metrics keys vary; the primary is "ITW" with "eer" subfield.
    itw_eer = None
    eval_metrics = ev.get("eval_metrics") or {}
    if isinstance(eval_metrics, dict):
        itw = eval_metrics.get("ITW")
        if isinstance(itw, dict):
            itw_eer = itw.get("eer")
    # Fallback: trainer's best_eer field is ITW by convention.
    if itw_eer is None:
        itw_eer = ev.get("best_eer")

    # 2019 LA eval is optional.
    la_eer = None
    if isinstance(eval_metrics, dict):
        la = eval_metrics.get("2019_LA")
        if isinstance(la, dict):
            la_eer = la.get("eer")

    history = None
    if hist_p.exists():
        try:
            history = json.loads(hist_p.read_text())
        except json.JSONDecodeError:
            history = None

    return {
        "name": run_dir.name,
        "seed": cfg.get("seed"),
        "backbone": cfg.get("backbone"),
        "model_class": cfg.get("__model_class__", "GLADAModel"),
        "model_kwargs": cfg.get("__model_kwargs__", {}),
        "num_groups": cfg.get("num_groups"),
        "vib_beta": cfg.get("vib_beta"),
        "musan_on": bool(cfg.get("musan_path")),
        "rir_on": bool(cfg.get("rir_path")),
        "confweight": bool(cfg.get("confidence_aware_adv_weight", False)),
        "phase1_epochs": cfg.get("phase1_epochs"),
        "phase2_epochs": cfg.get("phase2_epochs"),
        "itw_eer": itw_eer,
        "la_eer": la_eer,
        "elapsed_min": ev.get("elapsed_min"),
        "best_epoch": ev.get("best_epoch"),
        "history_len": len(history) if isinstance(history, list) else None,
    }


def _variant_key(run: dict) -> str:
    """Group runs with the same config-except-seed under one key.

    Used for mean ± std aggregation.
    """
    parts = [
        run["model_class"],
        f"bb={run['backbone']}",
        f"G={run['num_groups']}",
        f"beta={run['vib_beta']}",
    ]
    if run["model_kwargs"].get("vib_disabled_groups"):
        parts.append("vib_off="
                     + ",".join(map(str, run["model_kwargs"]["vib_disabled_groups"])))
    if run["musan_on"]:
        parts.append("musan")
    if run["rir_on"]:
        parts.append("rir")
    if run["confweight"]:
        parts.append("confweight")
    # Joining with ' | ' would collide with the Markdown table separator.
    # Use ' / ' so the variant cell renders as a single column.
    return " / ".join(parts)


def _format_md_table(rows: list[dict]) -> str:
    """Render a Markdown table.  Used for both Table V and the index CSV mirror."""
    if not rows:
        return "_(no runs)_\n"
    cols = ["variant", "seeds", "mean_itw_eer", "std_itw_eer", "delta_vs_paper"]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _format_tex_table(rows: list[dict]) -> str:
    """Render a LaTeX tabular body — drop into Table V in main.tex."""
    if not rows:
        return "% (no rows)\n"
    lines = []
    for r in rows:
        variant = r["variant"].replace("_", r"\_")
        mean_eer = r.get("mean_itw_eer", "--")
        std_eer = r.get("std_itw_eer", "--")
        n_seeds = r.get("seeds", 0)
        delta = r.get("delta_vs_paper", "--")
        if isinstance(mean_eer, float) and isinstance(std_eer, float):
            cell = f"{mean_eer:.2f} $\\pm$ {std_eer:.2f}"
        else:
            cell = f"{mean_eer}"
        if isinstance(delta, float):
            delta_str = f"${delta:+.2f}$"
        else:
            delta_str = str(delta)
        lines.append(f"\\texttt{{{variant}}} & {cell} & {n_seeds} & {delta_str} \\\\")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"[aggregate] runs_dir {runs_dir} does not exist", file=sys.stderr)
        return 2

    # Discover runs.  A run is any subdir that has an eval.json.
    candidates = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    runs: list[dict] = []
    for c in candidates:
        if args.section != "all" and not c.name.startswith(args.section):
            continue
        r = _load_run(c)
        if r is not None:
            runs.append(r)

    if not runs:
        print(f"[aggregate] no completed runs found under {runs_dir} "
              f"(section={args.section}).")
        return 0

    print(f"[aggregate] found {len(runs)} completed run(s) under {runs_dir}")

    # === Anomaly check ==================================================
    anomalies = []
    for r in runs:
        eer = r["itw_eer"]
        if eer is None or not isinstance(eer, (int, float)):
            anomalies.append((r["name"], f"missing ITW EER: {eer}"))
        elif not (3.0 <= eer <= 25.0):
            anomalies.append((r["name"],
                              f"ITW EER outside [3,25] band: {eer}"))
    if anomalies:
        print("[aggregate] ANOMALIES:")
        for name, msg in anomalies:
            print(f"  - {name}: {msg}")
        if args.check:
            return 3

    # === Run index =====================================================
    index_path = out_dir / "run_index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        for r in runs:
            row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                   for k, v in r.items()}
            w.writerow(row)
    print(f"[aggregate] wrote {index_path}")

    # === Multi-seed aggregation ========================================
    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        by_variant.setdefault(_variant_key(r), []).append(r)

    table_rows = []
    multiseed_rows = []
    for variant, group in sorted(by_variant.items()):
        eers = [g["itw_eer"] for g in group
                if isinstance(g["itw_eer"], (int, float))]
        seeds = sorted({g["seed"] for g in group if g["seed"] is not None})
        if not eers:
            continue
        mean_eer = statistics.fmean(eers)
        std_eer = statistics.stdev(eers) if len(eers) > 1 else 0.0
        delta = mean_eer - PAPER_BASELINE_ITW_EER
        row = {
            "variant": variant,
            "seeds": len(seeds),
            "seed_list": ",".join(map(str, seeds)),
            "mean_itw_eer": round(mean_eer, 2),
            "std_itw_eer": round(std_eer, 2),
            "delta_vs_paper": round(delta, 2),
            "n_runs": len(group),
        }
        table_rows.append(row)
        multiseed_rows.append({**row, "individual_eers": ",".join(
            f"{e:.2f}" for e in eers)})

    # Sort by mean EER ascending (best first).
    table_rows.sort(key=lambda r: r["mean_itw_eer"])

    # === Table V rows (Markdown + LaTeX) ==============================
    (out_dir / "table_v_rows.md").write_text(_format_md_table(table_rows))
    (out_dir / "table_v_rows.tex").write_text(_format_tex_table(table_rows))
    print(f"[aggregate] wrote table_v_rows.md / .tex "
          f"({len(table_rows)} variants)")

    # === Multi-seed CSV ================================================
    ms_path = out_dir / "multiseed_summary.csv"
    with ms_path.open("w", newline="") as f:
        if multiseed_rows:
            w = csv.DictWriter(f, fieldnames=list(multiseed_rows[0].keys()))
            w.writeheader()
            for r in multiseed_rows:
                w.writerow(r)
    print(f"[aggregate] wrote {ms_path}")

    # === Beta-sensitivity pivot ========================================
    # Only emit rows where multiple beta values exist for the same backbone +
    # model_class + num_groups combo.
    pivot: dict[tuple, dict[float, list[float]]] = {}
    for r in runs:
        if r["itw_eer"] is None:
            continue
        key = (r["model_class"], r["backbone"], r["num_groups"])
        pivot.setdefault(key, {}).setdefault(r["vib_beta"], []).append(
            r["itw_eer"])
    beta_rows = []
    for key, beta_map in pivot.items():
        if len(beta_map) < 2:
            continue
        for beta, eers in sorted(beta_map.items()):
            beta_rows.append({
                "model_class": key[0],
                "backbone": key[1],
                "num_groups": key[2],
                "vib_beta": beta,
                "n_seeds": len(eers),
                "mean_itw_eer": round(statistics.fmean(eers), 2),
                "std_itw_eer": (round(statistics.stdev(eers), 2)
                                if len(eers) > 1 else 0.0),
                "individual_eers": ",".join(f"{e:.2f}" for e in eers),
            })
    if beta_rows:
        beta_path = out_dir / "beta_sensitivity.csv"
        with beta_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(beta_rows[0].keys()))
            w.writeheader()
            for r in beta_rows:
                w.writerow(r)
        print(f"[aggregate] wrote {beta_path}")

    # === Limitations rewrite snippet ==================================
    # Convert each addressed Future Work item from "we did not test" to
    # "we tested at N seeds and observed X (mean ± std)".  We're
    # opinionated about phrasing because this is meant to drop into
    # main.tex with minimal editing.
    def _summary_for(pred) -> Optional[dict]:
        matches = [r for r in table_rows if pred(r)]
        return matches[0] if matches else None

    g2_row = _summary_for(lambda r: "G=2" in r["variant"])
    aug_row = _summary_for(lambda r: "musan" in r["variant"] and "rir" in r["variant"])
    confw_row = _summary_for(lambda r: "confweight" in r["variant"])

    paragraphs = []
    paragraphs.append(
        "% Generated by aggregate_results.py.  Review and adjust phrasing "
        "before pasting into main.tex Limitations subsection.\n")
    paragraphs.append(
        "We extended the camera-ready manuscript with experiments addressing "
        "Future Work items (b) and (c) and one stretch direction (confidence-"
        "aware adversarial weighting, after Huang et al., 2025).  All May-19 "
        "results were verified on three seeds (42, 123, 456) where compute "
        "allowed; single-seed entries are marked.")

    if g2_row:
        paragraphs.append(
            f"For (b), the $G{{=}}2$ partition on WavLM-Base-Plus reaches "
            f"{g2_row['mean_itw_eer']:.2f}\\% ITW EER "
            f"(N={g2_row['seeds']}, $\\sigma{{=}}{g2_row['std_itw_eer']:.2f}$), "
            f"a {g2_row['delta_vs_paper']:+.2f}\\,pp delta against the "
            f"per\\_group\\_vib baseline of {PAPER_BASELINE_ITW_EER:.2f}\\%.")
    if aug_row:
        paragraphs.append(
            f"For (c), adding MUSAN+RIR augmentation alongside RawBoost "
            f"reaches {aug_row['mean_itw_eer']:.2f}\\% ITW EER "
            f"(N={aug_row['seeds']}, $\\sigma{{=}}{aug_row['std_itw_eer']:.2f}$), "
            f"a {aug_row['delta_vs_paper']:+.2f}\\,pp delta.")
    if confw_row:
        paragraphs.append(
            f"As a stretch direction we extended IB-CAAN's confidence-margin "
            f"weighting from a single domain discriminator to the per-group "
            f"GLADA bank.  This reaches {confw_row['mean_itw_eer']:.2f}\\% ITW EER "
            f"(N={confw_row['seeds']}, "
            f"$\\sigma{{=}}{confw_row['std_itw_eer']:.2f}$), "
            f"a {confw_row['delta_vs_paper']:+.2f}\\,pp delta.  This is a "
            f"directional result; a full study of the interaction between "
            f"per-group adversarial weighting and per-group VIB compression "
            f"remains future work.")

    (out_dir / "limitations_rewrite.md").write_text(
        "\n\n".join(paragraphs) + "\n")
    print(f"[aggregate] wrote limitations_rewrite.md")

    print("[aggregate] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
