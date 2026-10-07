"""build_notebook.py — generator for pgvib_strengthening_a100.ipynb.

Mirrors the Coding/notebooks/april13/ pattern: a Python script emits the
notebook so the cell structure is auditable and regenerable.  Re-run after
editing the cell list to refresh the .ipynb.

Run with::

    python Coding/notebooks/may19/build_notebook.py

Writes ``pgvib_strengthening_a100.ipynb`` next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(*lines: str) -> dict:
    """Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l if l.endswith("\n") else l + "\n" for l in lines][:-1] +
                  [lines[-1] if not lines[-1].endswith("\n") else lines[-1][:-1]]
                  if lines else [""],
    }


def md_block(body: str) -> dict:
    """Markdown cell from a single body string (one trailing newline stripped)."""
    body = dedent(body).strip("\n")
    lines = body.splitlines(keepends=False)
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(body: str) -> dict:
    """Code cell from a single body string."""
    body = dedent(body).strip("\n")
    lines = body.splitlines(keepends=False)
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


CELLS = []


# ============================================================================
# Section 0 — Setup
# ============================================================================
CELLS.append(md_block(r"""
# pg-VIB Paper Strengthening Campaign — A100 Notebook (May 19)

This notebook patches the May-7 ICTAI manuscript's Future Work items (b)
"number-of-groups sweep" and (c) "MUSAN+RIR augmentation", solidifies the
single-seed 11.67% ITW EER paper headline via multi-seed replication, and
extends IB-CAAN's confidence-margin adversarial weighting to the per-group
GLADA disc bank.


Every section produces paper-ready artifacts under
`/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/`:
- `table_v_rows.md` / `table_v_rows.tex` — drop into Table V
- `multiseed_summary.csv` — mean ± std per variant
- `limitations_rewrite.md` — Markdown for the manuscript Limitations subsection
- `run_index.csv` — full audit trail

All training logic lives in `Coding/glada/`; this notebook is an orchestrator
that launches `run_pgvib.py` subprocesses (Rule 13: no model/trainer code
inlined in notebooks).  Re-running any section is safe: scripts skip runs
whose `history.json` already shows 20 completed epochs.

## Section Map

| Section | Experiments | Deliverable | Wall-time (A100) |
|---------|-------------|-------------|------------------|
| 0  | Mount Drive, install deps, copy glada pkg, extract datasets | Sanity gate | ~15 min |
| 1A | MUSAN+RIR + pg-VIB g0-only, seed 42 | Future Work (c) for Jade | ~4-5 h |
| 1B | G=2 num-groups + pg-VIB g0-only, seed 42 | Future Work (b) for Jade | ~4-5 h |
| 2A | pg-VIB g0-only headline, seeds 123 + 456 | Multi-seed for Abri | ~8-10 h |
| 2B | Section-1 winner, seeds 123 + 456 | Multi-seed confirmation | ~8-10 h |
| 2C | β-sweep multi-seed (optional, compute permitting) | β-sensitivity figure | ~9-12 h |
| 3  | Confidence-aware adversarial weighting on best config | Novel-extension row | ~4-5 h |
| 4  | Aggregate all results → Table V + Limitations rewrite | Paper deltas | ~30 s |

> **Re-entrancy:** the launcher cells use `subprocess.Popen` with stdout tee'd
> to Drive log files.  On Colab disconnect, re-mount Drive and re-run the
> notebook from the top — completed runs are auto-skipped, partial runs
> resume from the latest checkpoint.
"""))


CELLS.append(md_block("## 0. Setup"))

CELLS.append(code(r"""
# 0.1 — pip deps + Drive mount + GPU check
!pip install -q transformers soundfile librosa scikit-learn tqdm safetensors

from google.colab import drive
drive.mount('/content/drive')

import torch
assert torch.cuda.is_available(), "No GPU — switch runtime to A100"
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  ({gpu.total_memory/1e9:.1f} GB)")
if 'A100' not in gpu.name:
    print(f"[WARN] expected A100, got {gpu.name}. Consider lowering batch_size to 24.")
"""))

CELLS.append(code(r"""
# 0.2 — Copy glada package to /content (matches the april13 batch_size=64 pattern)
import os, shutil, sys, glob

CANDIDATES = [
    "/content/drive/Othercomputers/Jakob's MacBook Air/Thesis/298-deliverables/glada-code/glada",
    "/content/drive/MyDrive/Jakob/glada_pkg/glada",
    "/content/drive/MyDrive/Jakob/glada",
]
DRIVE_PKG = next((p for p in CANDIDATES if os.path.isdir(p)), None)
assert DRIVE_PKG is not None, (
    "Could not find glada package on Drive.  Run sync_glada_pkg.sh locally "
    "first to ship Coding/glada/ to Drive, then re-run this cell.")

LOCAL_PKG_PARENT = "/content"
LOCAL_PKG = os.path.join(LOCAL_PKG_PARENT, "glada")
if os.path.isdir(LOCAL_PKG):
    shutil.rmtree(LOCAL_PKG)
shutil.copytree(DRIVE_PKG, LOCAL_PKG)
if LOCAL_PKG_PARENT not in sys.path:
    sys.path.insert(0, LOCAL_PKG_PARENT)

import glada
print(f"glada loaded from {glada.__file__}")
# Sanity-check new symbols are present (May-19 package edits).
from glada.models import PerGroupVIBModel  # noqa: F401
from glada.config import GLADAConfig
_c = GLADAConfig(backbone='wavlm-base-plus',
                 confidence_aware_adv_weight=True,
                 num_groups_override=2)
assert _c.num_groups == 2
assert _c.confidence_aware_adv_weight is True
print("OK: PerGroupVIBModel import, num_groups_override=2, confidence_aware_adv_weight=True")
"""))

CELLS.append(code(r"""
# 0.3 — Copy may19 .py scripts from Drive to /content/may19 alongside the package
# Mirrors the april13 pattern: scripts live next to the notebook but are
# copied to the runtime SSD for fast imports.
import os, shutil

NB_CANDIDATES = [
    "/content/drive/Othercomputers/Jakob's MacBook Air/Thesis/Coding/notebooks/may19",
    "/content/drive/MyDrive/Jakob/notebooks/may19",
    "/content/drive/MyDrive/Jakob/glada_pkg/notebooks/may19",
]
NB_DRIVE = next((p for p in NB_CANDIDATES if os.path.isdir(p)), None)
assert NB_DRIVE is not None, (
    "Could not find Coding/notebooks/may19 on Drive.  Sync it before running.")

LOCAL_NB = "/content/may19"
if os.path.isdir(LOCAL_NB):
    shutil.rmtree(LOCAL_NB)
shutil.copytree(NB_DRIVE, LOCAL_NB)
print("scripts:", sorted(os.listdir(LOCAL_NB)))
"""))

CELLS.append(code(r"""
# 0.4 — Extract datasets to runtime SSD.  Idempotent: skips if already extracted.
# We pull ASVspoof19 LA + ITW unconditionally (used by every section) and
# MUSAN + RIR upfront so Section 1A doesn't block on extraction.
import os, shutil, tarfile

def extract_one(drive_archive, extract_to, verify_path, name):
    if os.path.isdir(verify_path):
        print(f"[{name}] already extracted at {verify_path}"); return
    os.makedirs(extract_to, exist_ok=True)
    local_archive = os.path.join("/content", os.path.basename(drive_archive))
    print(f"[{name}] copying archive from Drive...")
    shutil.copy2(drive_archive, local_archive)
    print(f"[{name}] extracting...")
    with tarfile.open(local_archive) as t:
        t.extractall(extract_to)
    os.remove(local_archive)
    print(f"[{name}] done")

# ASVspoof 2019 LA
extract_one(
    drive_archive="/content/drive/MyDrive/Jakob/Datasets/LA.tar.gz",
    extract_to="/content/data/asvspoof",
    verify_path="/content/data/asvspoof/LA/ASVspoof2019_LA_train",
    name="asvspoof19_la",
)

# In-the-Wild
extract_one(
    drive_archive="/content/drive/MyDrive/Jakob/Datasets/release_in_the_wild.tar.gz",
    extract_to="/content/data/itw",
    verify_path="/content/data/itw/release_in_the_wild",
    name="itw",
)

# MUSAN (only needed for Section 1A, but cheap to extract now)
musan_arch = "/content/drive/MyDrive/Jakob/Datasets/musan.tar.gz"
if os.path.exists(musan_arch):
    extract_one(
        drive_archive=musan_arch,
        extract_to="/content/data",
        verify_path="/content/data/musan",
        name="musan",
    )
else:
    print("[musan] archive missing on Drive; Section 1A will fall back to RawBoost-only")

# RIR_NOISES
rir_arch = "/content/drive/MyDrive/Jakob/Datasets/RIRS_NOISES.tar.gz"
if os.path.exists(rir_arch):
    extract_one(
        drive_archive=rir_arch,
        extract_to="/content/data",
        verify_path="/content/data/RIRS_NOISES/simulated_rirs",
        name="rir_noises",
    )
else:
    print("[rir] archive missing on Drive; Section 1A will fall back to RawBoost-only")

print()
print("Datasets ready:")
for p in ["/content/data/asvspoof/LA",
          "/content/data/itw/release_in_the_wild",
          "/content/data/musan",
          "/content/data/RIRS_NOISES/simulated_rirs"]:
    print(f"  {'OK ' if os.path.isdir(p) else 'NO '} {p}")
"""))

CELLS.append(code(r"""
# 0.5 — Run-launching utilities.
# Each launch_run() spawns a python subprocess for run_pgvib.py with the
# given CLI args, tees stdout+stderr to a Drive log file, and tracks the PID.
# Run a watcher cell after launches to monitor progress without re-blocking
# the runtime.
import os, subprocess, time, json, datetime
from pathlib import Path

CAMPAIGN_ROOT = Path("/content/drive/MyDrive/Jakob/Experiments/may19")
LOGS_ROOT = CAMPAIGN_ROOT / "_logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)
AGG_ROOT = CAMPAIGN_ROOT / "_aggregate"
AGG_ROOT.mkdir(parents=True, exist_ok=True)

RUNS: dict[str, dict] = {}  # name -> {proc, log_path, output_dir, started_at}


def launch_run(name: str, extra_args: list[str]) -> dict:
    output_dir = CAMPAIGN_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_ROOT / f"{name}.log"

    # Don't re-launch if completed (history.json has all 20 epochs).
    hist_path = output_dir / "history.json"
    if hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text())
            done = len(hist) if isinstance(hist, list) else hist.get("num_epochs", 0)
        except Exception:
            done = 0
        if done >= 20:
            print(f"[skip] {name} already has {done} epochs")
            return {"name": name, "status": "skipped"}

    cmd = [
        "python", "/content/may19/run_pgvib.py",
        "--output", str(output_dir),
    ] + extra_args
    print(f"[launch] {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    f = open(log_path, "a")
    f.write(f"\n=== launch @ {datetime.datetime.now().isoformat()} ===\n")
    f.flush()
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    RUNS[name] = {
        "proc": proc, "log_path": str(log_path),
        "output_dir": str(output_dir),
        "started_at": time.time(),
    }
    return {"name": name, "status": "launched", "pid": proc.pid}


def status_all() -> None:
    if not RUNS:
        print("No runs launched yet."); return
    for name, info in RUNS.items():
        rc = info["proc"].poll()
        elapsed = (time.time() - info["started_at"]) / 60.0
        state = "RUNNING" if rc is None else f"EXIT {rc}"
        print(f"  {name:<40s} {state:<12s} {elapsed:>6.1f} min")


def tail_logs(n_lines: int = 5) -> None:
    # Print the last n_lines of each log file.
    for name, info in RUNS.items():
        print(f"--- {name} ---")
        os.system(f"tail -n {n_lines} '{info['log_path']}'")


print("Launchers loaded.  Use launch_run(name, args), status_all(), tail_logs().")
"""))


# ============================================================================
# Section 1 — Quick wins for Jade
# ============================================================================
CELLS.append(md_block(r"""
## 1. Quick wins — paper Future Work (b) + (c)

Both runs share the verified pg-VIB headline base config: `PerGroupVIBModel`
with `vib_disabled_groups=(1,2)` on WavLM-Base-Plus, batch_size=64,
Phase 1=3ep + Phase 2=17ep, vib_beta=0.01, seed=42.  Paper baseline =
11.67% ITW EER (Table V `per_group_vib` row).

**1A** patches Future Work (c) by chaining MUSAN+RIR onto RawBoost.
**1B** patches Future Work (b) by switching from G=3 (registry default for
WavLM-Base-Plus) to G=2, which the paper itself nominates as "the smallest
experimental increment that would distinguish architectural failure from
training-dynamics failure" (main.tex lines 462-472).
"""))

CELLS.append(code(r"""
# 1A — MUSAN + RIR on pg-VIB g0-only headline (seed 42)
launch_run(
    name="1a_pgvib_g0only_musan_rir_s42",
    extra_args=[
        "--seed", "42",
        "--backbone", "wavlm-base-plus",
        "--model", "PerGroupVIBModel",
        "--vib_disabled_groups", "1,2",
        "--vib_beta", "0.01",
        "--phase1_epochs", "3", "--phase2_epochs", "17",
        "--batch_size", "64", "--effective_batch_size", "384",
        "--musan_path", "/content/data/musan",
        "--musan_prob", "0.5",
        "--rir_path", "/content/data/RIRS_NOISES/simulated_rirs",
        "--rir_prob", "0.5",
    ],
)
status_all()
"""))

CELLS.append(code(r"""
# 1B — G=2 number-of-groups sweep + pg-VIB g0-only (seed 42)
# vib_disabled_groups uses the LAST group index for G=2 (so g0 still has VIB,
# g1 is the second/coarsest group with VIB disabled).
launch_run(
    name="1b_pgvib_g0only_G2_s42",
    extra_args=[
        "--seed", "42",
        "--backbone", "wavlm-base-plus",
        "--model", "PerGroupVIBModel",
        "--num_groups_override", "2",
        "--vib_disabled_groups", "1",   # G=2 means indices 0,1; disable group 1
        "--vib_beta", "0.01",
        "--phase1_epochs", "3", "--phase2_epochs", "17",
        "--batch_size", "64", "--effective_batch_size", "384",
    ],
)
status_all()
"""))

CELLS.append(code(r"""
# Watcher cell — re-run anytime to peek at progress.  Interrupt to release.
# This cell is intentionally NOT a blocking wait: the runs continue in the
# background even if you interrupt this poll.
import time
while True:
    print(f"\n=== {time.strftime('%H:%M:%S')} ===")
    status_all()
    tail_logs(3)
    # Exit the watcher when all section-1 launches have finished.
    section1 = [n for n in RUNS if n.startswith('1')]
    done = all(RUNS[n]['proc'].poll() is not None for n in section1)
    if section1 and done:
        print("\nAll Section 1 runs finished.")
        break
    time.sleep(60)
"""))

CELLS.append(code(r"""
# 1.aggregate — Section 1 deliverable for Jade.
# Emits table_v_rows.md / .tex / .csv covering Section 1 only.
!python /content/may19/aggregate_results.py \
    --runs_dir "{CAMPAIGN_ROOT}" \
    --out_dir  "{AGG_ROOT}/section1" \
    --section 1 --check
print()
print("=== Table V rows (Section 1) ===")
print(open(f"{AGG_ROOT}/section1/table_v_rows.md").read())
print()
print("=== Limitations rewrite (Section 1) ===")
print(open(f"{AGG_ROOT}/section1/limitations_rewrite.md").read())
""".replace("{CAMPAIGN_ROOT}", "/content/drive/MyDrive/Jakob/Experiments/may19")
   .replace("{AGG_ROOT}", "/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate")))


# ============================================================================
# Section 2 — Essential strengthening (multi-seed rigor)
# ============================================================================
CELLS.append(md_block(r"""
## 2. Essential strengthening — multi-seed rigor

The pg-VIB g0-only baseline at 11.67% is single-seed (s42).  This section adds
seeds 123 and 456 so the headline carries a 3-seed mean ± std for Abri.
Section 2B re-runs whichever of 1A or 1B improved on the baseline at the
remaining two seeds; Section 2C (optional, compute permitting) adds multi-seed
to the β-sensitivity probe.
"""))

CELLS.append(code(r"""
# 2A — pg-VIB g0-only multi-seed (seeds 123, 456; seed 42 already exists)
for seed in [123, 456]:
    launch_run(
        name=f"2a_pgvib_g0only_s{seed}",
        extra_args=[
            "--seed", str(seed),
            "--backbone", "wavlm-base-plus",
            "--model", "PerGroupVIBModel",
            "--vib_disabled_groups", "1,2",
            "--vib_beta", "0.01",
            "--phase1_epochs", "3", "--phase2_epochs", "17",
            "--batch_size", "64", "--effective_batch_size", "384",
        ],
    )
status_all()
"""))

CELLS.append(code(r"""
# 2B — Multi-seed for Section-1 winner.
# Inspect the section-1 aggregate; pick whichever variant has the lower mean EER
# and replicate it at seeds 123, 456.  If both improved the baseline, prefer
# the one with the smaller seed-42 EER.
import json
sec1_idx = AGG_ROOT / "section1" / "run_index.csv"
import csv
winner_extra_args = None
winner_name_prefix = None
if sec1_idx.exists():
    rows = list(csv.DictReader(open(sec1_idx)))
    s1_rows = [r for r in rows if r['seed'] == '42' and float(r['itw_eer']) < 11.67]
    if s1_rows:
        best = min(s1_rows, key=lambda r: float(r['itw_eer']))
        print(f"Section 1 winner: {best['name']}  (ITW EER = {best['itw_eer']})")
        if "musan_rir" in best['name']:
            winner_name_prefix = "2b_winner_musan_rir"
            winner_extra_args = [
                "--musan_path", "/content/data/musan", "--musan_prob", "0.5",
                "--rir_path", "/content/data/RIRS_NOISES/simulated_rirs",
                "--rir_prob", "0.5",
            ]
        elif "G2" in best['name']:
            winner_name_prefix = "2b_winner_G2"
            winner_extra_args = ["--num_groups_override", "2",
                                 "--vib_disabled_groups", "1"]
    else:
        print("Section 1 had no improvement over baseline.  Skipping 2B.")
else:
    print("Section 1 aggregate not yet available; run the section-1 aggregate cell first.")

if winner_extra_args is not None:
    base_args = [
        "--backbone", "wavlm-base-plus",
        "--model", "PerGroupVIBModel",
        "--vib_disabled_groups", "1,2",
        "--vib_beta", "0.01",
        "--phase1_epochs", "3", "--phase2_epochs", "17",
        "--batch_size", "64", "--effective_batch_size", "384",
    ]
    # winner_extra_args may override --vib_disabled_groups (G=2 case)
    if "--vib_disabled_groups" in winner_extra_args:
        idx = base_args.index("--vib_disabled_groups")
        del base_args[idx:idx + 2]
    for seed in [123, 456]:
        launch_run(
            name=f"{winner_name_prefix}_s{seed}",
            extra_args=["--seed", str(seed)] + base_args + winner_extra_args,
        )
status_all()
"""))

CELLS.append(code(r"""
# 2C — Optional: multi-seed β-sweep validation.
# Only run if Section 2A + 2B are done AND there's compute budget left.
# Sweeps vib_beta in {0.005, 0.01, 0.05} on pg-VIB g0-only at seeds 123, 456
# (the existing single-seed sweep already covered seed 42 at these betas).
RUN_2C = False  # flip to True to enable

if RUN_2C:
    for beta in [0.005, 0.05]:  # 0.01 already covered by 2A
        for seed in [123, 456]:
            launch_run(
                name=f"2c_betasweep_b{beta}_s{seed}",
                extra_args=[
                    "--seed", str(seed),
                    "--backbone", "wavlm-base-plus",
                    "--model", "PerGroupVIBModel",
                    "--vib_disabled_groups", "1,2",
                    "--vib_beta", str(beta),
                    "--phase1_epochs", "3", "--phase2_epochs", "17",
                    "--batch_size", "64", "--effective_batch_size", "384",
                ],
            )
    status_all()
else:
    print("Section 2C skipped (RUN_2C=False).  Flip to True if compute remains.")
"""))

CELLS.append(code(r"""
# 2.aggregate — Section 2 deliverable for Abri.
!python /content/may19/aggregate_results.py \
    --runs_dir "/content/drive/MyDrive/Jakob/Experiments/may19" \
    --out_dir  "/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/section2" \
    --section 2 --check
print()
print("=== Multi-seed summary (Section 2) ===")
print(open("/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/section2/multiseed_summary.csv").read())
"""))


# ============================================================================
# Section 3 — Stretch: confidence-aware adversarial weighting
# ============================================================================
CELLS.append(md_block(r"""
## 3. Stretch — confidence-aware adversarial weighting (IB-CAAN extension)

Applies IB-CAAN's confidence-margin weighting (Huang et al., 2025, eq. 7) to
GLADA's per-group discriminator bank.  Per-sample CE is multiplied by
`|softmax(D_k(g_k))[:,1] - 0.5|` before being averaged across the batch,
so the GRL gradient is dominated by samples on which the discriminator is
confident.

Trainer modification is opt-in via `--confidence_aware_adv_weight`; default
behavior is unchanged.  Runs the best Section-1/2 config at seed 42 first;
if it improves, replicates at seed 123 for noise-rejection.
"""))

CELLS.append(code(r"""
# 3 — confidence-aware adversarial weighting on the best config so far
# Reads section-2 aggregate to pick the best variant; runs it with confweight on.
import csv
sec2_idx = AGG_ROOT / "section2" / "run_index.csv"
best_args = None
best_name = None
if sec2_idx.exists():
    rows = list(csv.DictReader(open(sec2_idx)))
    rows = [r for r in rows if r['itw_eer'] not in ('', None)]
    if rows:
        best_row = min(rows, key=lambda r: float(r['itw_eer']))
        best_name = best_row['name']
        # Reconstruct CLI args from the run's config.json.
        cfg_path = CAMPAIGN_ROOT / best_row['name'] / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            best_args = []
            # Reuse the same knobs the winner used, but force confweight.
            if cfg.get('num_groups_override', 0) > 0:
                best_args += ["--num_groups_override", str(cfg['num_groups_override'])]
            best_args += ["--vib_beta", str(cfg.get('vib_beta', 0.01))]
            mk = cfg.get('__model_kwargs__', {})
            if mk.get('vib_disabled_groups'):
                vdg = ",".join(map(str, mk['vib_disabled_groups']))
                best_args += ["--vib_disabled_groups", vdg]
            if cfg.get('musan_path'):
                best_args += ["--musan_path", cfg['musan_path'],
                              "--musan_prob", str(cfg.get('musan_prob', 0.5))]
            if cfg.get('rir_path'):
                best_args += ["--rir_path", cfg['rir_path'],
                              "--rir_prob", str(cfg.get('rir_prob', 0.5))]

if best_args is not None:
    print(f"Stretch base config from {best_name}")
    launch_run(
        name="3_confweight_s42",
        extra_args=[
            "--seed", "42",
            "--backbone", "wavlm-base-plus",
            "--model", "PerGroupVIBModel",
            "--phase1_epochs", "3", "--phase2_epochs", "17",
            "--batch_size", "64", "--effective_batch_size", "384",
            "--confidence_aware_adv_weight",
        ] + best_args,
    )
    status_all()
else:
    print("No Section 2 results yet.  Run Section 2 aggregate first.")
"""))


# ============================================================================
# Section 4 — Final aggregation
# ============================================================================
CELLS.append(md_block(r"""
## 4. Final aggregation — paper deltas

Emits the consolidated Table V rows, β-sensitivity CSV, multi-seed summary,
and Limitations subsection rewrite covering every May-19 result.
"""))

CELLS.append(code(r"""
!python /content/may19/aggregate_results.py \
    --runs_dir "/content/drive/MyDrive/Jakob/Experiments/may19" \
    --out_dir  "/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/all" \
    --section all --check

print()
print("=== Table V (all May-19 variants) ===")
print(open("/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/all/table_v_rows.md").read())
print()
print("=== Limitations rewrite ===")
print(open("/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/all/limitations_rewrite.md").read())
print()
print("=== LaTeX (paste into Table V in main.tex) ===")
print(open("/content/drive/MyDrive/Jakob/Experiments/may19/_aggregate/all/table_v_rows.tex").read())
"""))


# ============================================================================
# Notebook assembly
# ============================================================================
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "pgvib_strengthening_a100.ipynb",
            "provenance": [],
            "gpuType": "A100",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    out = Path(__file__).parent / "pgvib_strengthening_a100.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
