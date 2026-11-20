"""Runs the training plan (TRAINING_PLAN.md): warm start, sweep, fine-tune, evaluate and summarize.
- python sweep.py sanity               # reward check: quitting must score below random driving
- python sweep.py bc                   # record the path follower and pretrain the pixel policy (warm start)
- python sweep.py phase1               # gamma x seeds, each run started from the BC policy
- python sweep.py summary phase1       # results table (finish rate, win rate, time, recklessness)
- python sweep.py phase2-train         # fine-tune FIXED_POLICY under each reward/physics change
- python sweep.py phase2-eval          # evaluate FIXED_POLICY, unchanged, under env changes
- python sweep.py summary phase2
- Options: --only a,b (just those runs) | --seeds 0,1 | --fixed-policy PATH | --races N | --dry-run | --quick
- Runs that reached their step count and evaluations that have all their races are skipped, so after a stop
  rerun the same command. A stopped training run restarts from scratch (replay isn't saved)."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

RUNS = Path("runs")

# === Warm start: behavior cloning from the path follower (bc_pretrain.py) ===
BC = {"episodes": 20, "steps": 3000, "noise": 0.3,  # teacher races, pretraining steps, steering noise
      "dagger_rounds": 3, "dagger_episodes": 5, "dagger_steps": 1000}  # student drives, teacher relabels

# === Phase 1: every run starts from the BC policy and changes gamma; each is repeated for every seed ===
BASE = {"total_steps": 150_000, "gamma": 0.99,
        "difficulty": 0,  # novice opponent: the only level where a decent driver can win
        "critic_warmup": 2000,  # critic-only updates first, so untrained critics can't wreck the BC actor
        "buffer_size": 30_000}  # room for the ~15k demonstration transitions plus fresh experience
PHASE1 = {
    "gamma_0.98": {"gamma": 0.98},
    "gamma_0.99": {},
    "gamma_0.995": {"gamma": 0.995},
}
SEEDS = [0, 1]
PHASE1_CHECKPOINTS = [50_000, 100_000, 150_000]  # the training-length comparison, from each run's checkpoints
BASELINES = ["random", "full_throttle"]  # reference policies, evaluated alongside phase 1

# === After phase 1: the chosen checkpoint and the settings it was trained with ===
FIXED_POLICY = None  # e.g. "runs/phase1/gamma_0.99_s0/step_100000.pt"
FIXED = {**BASE}  # e.g. {**BASE, "gamma": 0.995}

# === Phase 2a: fine-tune FIXED_POLICY under each change (how its decisions adapt) ===
FINETUNE = {"total_steps": 50_000, "learning_starts": 1000, "critic_warmup": 0}  # critics are already trained
PHASE2_TRAIN = {
    "fixed": [],  # fine-tuned with no change: the reference, with the same extra training as every change
    "step_0.01": ["step_penalty=0.01"],  # less time pressure
    "step_0.05": ["step_penalty=0.05"],  # more time pressure
    "offroad_0": ["off_road_factor=0.0"],  # off-road meters worth nothing
    "offroad_1": ["off_road_factor=1.0"],  # off-road as good as the road
    "win_45": ["win_bonus=45"],  # winning matters 3x more
    "no_brakes": ["no_brakes=True"],
}

# === Phase 2b: evaluate FIXED_POLICY unchanged under each change (robustness, no training) ===
PHASE2_EVAL = {
    "same": {},
    "expert": {"difficulty": 2},  # note: difficulty also changes our kart's speed
    "karts_4": {"env": ["n_karts=4"]},
    "noise_0.2": {"env": ["action_noise=0.2"]},
    "delay_2": {"env": ["action_delay=2"]},
    "kart_gnu": {"env": ["kart=gnu"]},
    "track_zengarden": {"env": ["track=zengarden"]},
}

QUICK = {"total_steps": 600, "learning_starts": 300, "buffer_size": 1000, "checkpoint_every": 300,
         "critic_warmup": 100}  # --quick: tiny runs that only test the pipeline
QUICK_BC = {"episodes": 2, "steps": 100, "dagger_rounds": 1, "dagger_episodes": 1, "dagger_steps": 50}


# === Commands ===
def run(cmd, dry_run):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def flags(settings):
    """{'total_steps': 600, 'init_from': path} -> ['--total-steps', 600, '--init-from', path]."""
    return [item for key, value in settings.items() for item in (f"--{key.replace('_', '-')}", value)]


def train_cmd(run_dir, settings, env=(), seed=0):
    cmd = [sys.executable, "train.py", "--run-dir", run_dir, "--seed", seed, *flags(settings)]
    for option in env:
        cmd += ["--env", option]
    return cmd


def eval_cmd(policy, name, out, races, difficulty, env=()):
    cmd = [sys.executable, "evaluate.py", policy, "--name", name, "--out", out, "--races", races,
           "--difficulty", difficulty]
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


def root_dir(args):
    return RUNS / "quick" if args.quick else RUNS


def require(path, hint, args):
    if not Path(path).exists() and not args.dry_run:
        sys.exit(f"{path} not found: {hint}")


# === Steps ===
def bc(args):
    root = root_dir(args)
    out, settings = root / "bc", {**BC, **(QUICK_BC if args.quick else {})}
    if not (out / "policy.pt").exists():
        run([sys.executable, "bc_pretrain.py", "--out", out, "--difficulty", BASE["difficulty"],
             "--check-races", 0, *flags(settings)], args.dry_run)
    eval_out = root / "eval" / "phase1"  # the BC policy appears in phase 1's summary as a reference
    if not evaluated(eval_out / "bc.csv", args.races):
        run(eval_cmd(out / "policy.pt", "bc", eval_out, args.races, BASE["difficulty"]), args.dry_run)


def phase1(args):
    root = root_dir(args)
    bc_dir, out = root / "bc", root / "eval" / "phase1"
    require(bc_dir / "policy.pt", "run `python sweep.py bc` first", args)
    for name in selected(PHASE1, args.only):
        for seed in args.seeds or SEEDS:
            run_name, run_dir = f"{name}_s{seed}", root / "phase1" / f"{name}_s{seed}"
            settings = {**BASE, **PHASE1[name], "init_from": bc_dir / "policy.pt", "demos": bc_dir / "demos.npz",
                        **(QUICK if args.quick else {})}
            if not trained(run_dir, settings["total_steps"]):
                run(train_cmd(run_dir, settings, seed=seed), args.dry_run)
            steps = [] if args.quick else PHASE1_CHECKPOINTS
            targets = [(f"{run_name}@{s // 1000}k", run_dir / f"step_{s:06d}.pt") for s in steps]
            for label, policy in targets or [(run_name, run_dir / "final.pt")]:
                if not evaluated(out / f"{label}.csv", args.races):
                    run(eval_cmd(policy, label, out, args.races, BASE["difficulty"]), args.dry_run)
    for baseline in selected(BASELINES, args.only):
        if not evaluated(out / f"{baseline}.csv", args.races):
            run(eval_cmd(baseline, baseline, out, args.races, BASE["difficulty"]), args.dry_run)


def fixed_policy(args):
    policy = args.fixed_policy or FIXED_POLICY
    if policy is None:
        if args.dry_run:
            return "<FIXED_POLICY>"
        sys.exit("Set FIXED_POLICY in sweep.py (or pass --fixed-policy) to the checkpoint chosen in phase 1")
    require(policy, "check the FIXED_POLICY path", args)
    return policy


def phase2_train(args):
    root, policy = root_dir(args), fixed_policy(args)
    out = root / "eval" / "phase2"
    for name in selected(PHASE2_TRAIN, args.only):
        for seed in args.seeds or [0]:
            run_name = name + (f"_s{seed}" if seed else "")
            run_dir = root / "phase2" / run_name
            settings = {**FIXED, **FINETUNE, "init_from": policy, **(QUICK if args.quick else {}),
                        "critic_warmup": 0}
            if not trained(run_dir, settings["total_steps"]):
                run(train_cmd(run_dir, settings, PHASE2_TRAIN[name], seed), args.dry_run)
            if not evaluated(out / f"{run_name}.csv", args.races):  # evaluated in the setting it was tuned in
                run(eval_cmd(run_dir / "final.pt", run_name, out, args.races, FIXED["difficulty"],
                             PHASE2_TRAIN[name]), args.dry_run)


def phase2_eval(args):
    root, policy = root_dir(args), fixed_policy(args)
    out = root / "eval" / "phase2"
    for name in selected(PHASE2_EVAL, args.only):
        change, label = PHASE2_EVAL[name], f"fixed_on_{name}"
        if not evaluated(out / f"{label}.csv", args.races):
            run(eval_cmd(policy, label, out, args.races, change.get("difficulty", FIXED["difficulty"]),
                         change.get("env", [])), args.dry_run)


def summary(args):
    eval_dir = root_dir(args) / "eval" / args.phase
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
    parser.add_argument("command", choices=["sanity", "bc", "phase1", "phase2-train", "phase2-eval", "summary"])
    parser.add_argument("phase", nargs="?", choices=["phase1", "phase2"], help="for summary")
    parser.add_argument("--only", type=lambda s: set(s.split(",")), default=None, help="comma-separated run names")
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=None,
                        help="training seeds (default: phase 1 uses SEEDS, phase 2 uses 0)")
    parser.add_argument("--fixed-policy", default=None, help="overrides FIXED_POLICY for phase 2")
    parser.add_argument("--races", type=int, default=None, help="races per evaluation (default 20, --quick 1)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="for sanity")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--quick", action="store_true", help="tiny runs under runs/quick/ to test the pipeline")
    args = parser.parse_args()
    args.races = args.races or (1 if args.quick else 20)
    if args.command == "summary" and not args.phase:
        parser.error("summary needs a phase: python sweep.py summary phase1")
    {"sanity": sanity, "bc": bc, "phase1": phase1, "phase2-train": phase2_train,
     "phase2-eval": phase2_eval, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
