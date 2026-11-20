"""Runs the training plan (TRAINING_PLAN.md): trains each experiment, evaluates it, and summarizes the results.
- python sweep.py sanity               # reward check: quitting must score below random driving
- python sweep.py phase1               # hyperparameter sweep: train + evaluate every PHASE1 run
- python sweep.py summary phase1       # results table (finish rate, win rate, time, recklessness)
- python sweep.py phase2-train         # retrain FIXED under each reward/physics change
- python sweep.py phase2-eval          # evaluate the fixed policy, unchanged, under env changes
- python sweep.py summary phase2
- Options: --only a,b (just those runs) | --seed N (repeat with another seed) | --races N | --dry-run | --quick
- Runs that reached their step count and evaluations that have all their races are skipped, so after a stop
  rerun the same command. A stopped training run restarts from scratch (replay isn't saved)."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

RUNS = Path("runs")

# === Phase 1: tune how well the agent learns; each run changes one knob from BASE ===
BASE = {"total_steps": 100_000, "alpha": 0.2, "gamma": 0.99, "sequence_length": 8,
        "difficulty": 0}  # novice opponent: the only level where a decent driver can win
PHASE1 = {
    "baseline": {"total_steps": 300_000},  # longer: its 100k/200k/300k checkpoints give the length comparison
    "alpha_0.05": {"alpha": 0.05},
    "alpha_0.1": {"alpha": 0.1},
    "gamma_0.995": {"gamma": 0.995},
    "gamma_0.998": {"gamma": 0.998},
    "seq_16": {"sequence_length": 16},
}
PHASE1_CHECKPOINTS = {"baseline": [100_000, 200_000, 300_000]}  # other runs: their final (100k) policy
BASELINES = ["random", "full_throttle"]  # reference policies, evaluated alongside phase 1

# === After phase 1: put the winning settings here (e.g. {**BASE, "alpha": 0.05, "total_steps": 200_000}) ===
FIXED = {**BASE}

# === Phase 2a: retrain FIXED under each change (how the agent's decisions adapt) ===
PHASE2_TRAIN = {
    "fixed": [],  # FIXED unchanged: the reference every change is compared against
    "step_0.01": ["step_penalty=0.01"],  # less time pressure
    "step_0.05": ["step_penalty=0.05"],  # more time pressure
    "offroad_0": ["off_road_factor=0.0"],  # off-road meters worth nothing
    "offroad_1": ["off_road_factor=1.0"],  # off-road as good as the road
    "win_45": ["win_bonus=45"],  # winning matters 3x more
    "no_brakes": ["no_brakes=True"],
}

# === Phase 2b: evaluate the fixed policy unchanged under each change (robustness, no retraining) ===
PHASE2_EVAL = {
    "same": {},
    "expert": {"difficulty": 2},  # note: difficulty also changes our kart's speed
    "karts_4": {"env": ["n_karts=4"]},
    "noise_0.2": {"env": ["action_noise=0.2"]},
    "delay_2": {"env": ["action_delay=2"]},
    "kart_gnu": {"env": ["kart=gnu"]},
    "track_zengarden": {"env": ["track=zengarden"]},
}


# === Commands ===
def run(cmd, dry_run):
    print("$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def train_cmd(run_dir, settings, env=(), seed=0, quick=False):
    settings = {**settings, **({"total_steps": 600} if quick else {})}
    cmd = [sys.executable, "train.py", "--run-dir", str(run_dir), "--seed", str(seed)]
    if quick:  # tiny smoke test of the whole pipeline
        cmd += ["--learning-starts", "300", "--buffer-size", "1000", "--checkpoint-every", "300"]
    for key, value in settings.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    for option in env:
        cmd += ["--env", option]
    return cmd, settings["total_steps"]


def eval_cmd(policy, name, out, races, difficulty, env=()):
    cmd = [sys.executable, "evaluate.py", str(policy), "--name", name, "--out", str(out),
           "--races", str(races), "--difficulty", str(difficulty)]
    for option in env:
        cmd += ["--env", option]
    return cmd


def trained(run_dir, total_steps):
    """True if run_dir has a final.pt that reached total_steps (a stopped run also saves final.pt)."""
    final = run_dir / "final.pt"
    if not final.exists():
        return False
    import torch
    return torch.load(final, map_location="cpu", weights_only=False)["step"] >= total_steps


def evaluated(csv_path, races):
    return csv_path.exists() and sum(1 for _ in csv_path.open()) - 1 >= races


def selected(names, only):
    return [n for n in names if not only or n in only]


# === Phases ===
def phase1(args):
    out = RUNS / ("quick" if args.quick else "") / "eval" / "phase1"
    for name in selected(PHASE1, args.only):
        run_name = name + (f"_s{args.seed}" if args.seed else "")
        run_dir = RUNS / ("quick" if args.quick else "") / "phase1" / run_name
        cmd, steps = train_cmd(run_dir, {**BASE, **PHASE1[name]}, seed=args.seed, quick=args.quick)
        if not trained(run_dir, steps):
            run(cmd, args.dry_run)
        checkpoints = [] if args.quick else PHASE1_CHECKPOINTS.get(name, [])
        targets = [(f"{run_name}@{s // 1000}k", run_dir / f"step_{s:06d}.pt") for s in checkpoints]
        targets = targets or [(run_name, run_dir / "final.pt")]
        for label, policy in targets:
            if not evaluated(out / f"{label}.csv", args.races):
                run(eval_cmd(policy, label, out, args.races, BASE["difficulty"]), args.dry_run)
    for baseline in selected(BASELINES, args.only):
        if not evaluated(out / f"{baseline}.csv", args.races):
            run(eval_cmd(baseline, baseline, out, args.races, BASE["difficulty"]), args.dry_run)


def phase2_train(args):
    out = RUNS / ("quick" if args.quick else "") / "eval" / "phase2"
    for name in selected(PHASE2_TRAIN, args.only):
        run_name = name + (f"_s{args.seed}" if args.seed else "")
        run_dir = RUNS / ("quick" if args.quick else "") / "phase2" / run_name
        cmd, steps = train_cmd(run_dir, FIXED, PHASE2_TRAIN[name], args.seed, args.quick)
        if not trained(run_dir, steps):
            run(cmd, args.dry_run)
        if not evaluated(out / f"{run_name}.csv", args.races):  # evaluated in the setting it was trained in
            run(eval_cmd(run_dir / "final.pt", run_name, out, args.races, FIXED["difficulty"],
                         PHASE2_TRAIN[name]), args.dry_run)


def phase2_eval(args):
    root = RUNS / ("quick" if args.quick else "")
    policy, out = root / "phase2" / "fixed" / "final.pt", root / "eval" / "phase2"
    if not policy.exists() and not args.dry_run:
        sys.exit(f"{policy} not found: run `python sweep.py phase2-train --only fixed` first")
    for name in selected(PHASE2_EVAL, args.only):
        change = PHASE2_EVAL[name]
        label = f"fixed_on_{name}"
        if not evaluated(out / f"{label}.csv", args.races):
            run(eval_cmd(policy, label, out, args.races, change.get("difficulty", FIXED["difficulty"]),
                         change.get("env", [])), args.dry_run)


def summary(args):
    eval_dir = RUNS / ("quick" if args.quick else "") / "eval" / args.phase
    metrics = ["finished", "won", "stuck", "finish_time", "mean_speed", "offroad_frac", "steer_jerk",
               "mean_yaw_rate", "crashes"]
    rows = []
    for path in sorted(eval_dir.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        races = list(csv.DictReader(path.open()))
        row = {"run": path.stem, "races": len(races)}
        for key in metrics:
            values = [float(r[key]) for r in races if r.get(key) not in ("", None, "None")]
            row[key] = round(float(np.mean(values)), 3) if values else ""
        rows.append(row)
    if not rows:
        sys.exit(f"No results in {eval_dir}")
    rows.sort(key=lambda r: (-r["finished"], r["finish_time"] if r["finish_time"] != "" else 1e9))
    with (eval_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'run':28s} {'races':>5s} {'finish':>7s} {'win':>6s} {'stuck':>6s} {'time s':>7s} {'speed':>6s} "
          f"{'offroad':>8s} {'jerk':>6s} {'crash':>6s}")
    for r in rows:
        print(f"{r['run']:28s} {r['races']:5d} {r['finished']:7.0%} {r['won']:6.0%} {r['stuck']:6.0%} "
              f"{r['finish_time'] if r['finish_time'] != '' else '-':>7} {r['mean_speed']:6.1f} "
              f"{r['offroad_frac']:8.0%} {r['steer_jerk']:6.3f} {r['crashes']:6.1f}")
    print(f"\nSaved {eval_dir / 'summary.csv'} (sorted by finish rate, then finish time)")


def sanity(args):
    """Score simple behaviors under the current reward: quitting must score below random driving."""
    from environment import make_env
    from train import parse_env_options
    env, rng = make_env("state", FIXED["difficulty"], **parse_env_options(args.env)), np.random.default_rng(0)
    A = lambda accel=0.0, steer=0.0, brake=0.0: np.array([accel, steer, brake, 0, 0], np.float32)
    behaviors = {
        "stand still": (lambda o: A(), 1),
        "reverse off the start": (lambda o: A(brake=1.0), 1),
        "random actions": (lambda o: rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32), 5),
        "path follower": (lambda o: A(1.0, np.clip(2 * np.arctan2(o[9], o[10]), -1, 1)), 1),
    }
    scores = {}
    for name, (act, n) in behaviors.items():
        returns = []
        for seed in range(n):
            obs, _ = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(act(obs))
                total, done = total + reward, terminated or truncated
            returns.append(total)
        scores[name] = float(np.mean(returns))
        print(f"{name:22s}: mean return {scores[name]:8.2f}")
    env.close()
    ok = max(scores["stand still"], scores["reverse off the start"]) < scores["random actions"] < scores["path follower"]
    print("OK: quitting < random driving < a lap" if ok else
          "WARNING: the reward makes quitting look as good as driving; the agent may learn to park")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sanity", "phase1", "phase2-train", "phase2-eval", "summary"])
    parser.add_argument("phase", nargs="?", choices=["phase1", "phase2"], help="for summary")
    parser.add_argument("--only", type=lambda s: set(s.split(",")), default=None, help="comma-separated run names")
    parser.add_argument("--seed", type=int, default=0, help="training seed; nonzero adds _s<seed> to run names")
    parser.add_argument("--races", type=int, default=None, help="races per evaluation (default 50, --quick 1)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="for sanity")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--quick", action="store_true", help="tiny runs under runs/quick/ to test the pipeline")
    args = parser.parse_args()
    args.races = args.races or (1 if args.quick else 50)
    if args.command == "summary" and not args.phase:
        parser.error("summary needs a phase: python sweep.py summary phase1")
    {"sanity": sanity, "phase1": phase1, "phase2-train": phase2_train,
     "phase2-eval": phase2_eval, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
