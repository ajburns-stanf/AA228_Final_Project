# === Evaluate a Trained Policy ===
"""Runs a saved SAC state policy and three baselines for several races each, prints and saves statistics,
and records an MP4 of the SAC policy's best run.
- Usage: python evaluate.py [policy_path] [n_runs]   (defaults: runs/sac_state/best_policy.pt, 10)
- Outputs in runs/eval/ (git-ignored): runs.csv (one row per race), summary.csv, best_run.mp4
- Rendering is on so frames can be recorded; the policies still drive from the 18-float state.
- Every policy races the same seeds (0 .. n_runs - 1), so the comparison is fair."""

# === Imports ===
import csv
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

import environment  # imported first: sets up the STK DLLs on Windows
from claude_gen.SAC_State import Policy, to_env

# === Definitions ===
POLICY_PATH = Path(__file__).parent / "runs" / "sac_state" / "best_policy.pt"
N_RUNS = 10  # races per policy
OUT_DIR = Path(__file__).parent / "runs" / "eval"
VIDEO_SIZE = (
    640,
    480,
)  # (width, height) of the recorded frames; must be multiples of 16
STEP_TIME = 0.1  # seconds of game time per step
FPS = round(1 / STEP_TIME)  # one frame per step, so the video plays in real time


# === Policies: each maps the 18-float state to an env action [acceleration, steer] ===
def load_sac(path, n_obs, action_space):
    """The trained SAC policy, driving with its mean action (no exploration)."""
    policy = Policy(n_obs, action_space.shape[0])
    policy.load_state_dict(torch.load(path))
    policy.eval()

    def act(state):
        with torch.no_grad():
            _, _, mean_action = policy(torch.as_tensor(state).unsqueeze(0))
        return to_env(mean_action[0].numpy(), action_space)

    return act


def follow_path(state):
    """Hand-written baseline: steer toward the 2nd upcoming centerline point (state[9], state[10])."""
    steer = np.clip(2.0 * np.arctan2(state[9], state[10]), -1, 1)
    return np.array([1.0, steer], dtype=np.float32)


def full_throttle(state):
    return np.array([1.0, 0.0], dtype=np.float32)


# === Running Races ===
def run(env, act, seed, video_path=None):
    """One race with policy act. Returns its statistics; writes its frames to video_path if given."""
    frame, _ = env.reset(seed=seed)
    writer = imageio.get_writer(video_path, fps=FPS) if video_path else None
    total, offroad, done = 0.0, 0, False
    while not done:
        if writer:
            writer.append_data(frame)
        frame, r, terminated, truncated, info = env.step(act(env._state()))
        total += r
        offroad += not env.world.players[0].kart.is_on_road
        done = terminated or truncated
    if writer:
        writer.append_data(frame)
        writer.close()

    return {
        "return": round(total, 2),
        "distance_m": round(info["distance"], 1),
        "finished": int(terminated),
        "lap_time_s": round(env.steps * STEP_TIME, 1)
        if terminated
        else "",  # only for finished races
        "offroad_pct": round(100 * offroad / env.steps, 1),
        "position": env.world.players[0].kart.position,
        "steps": env.steps,
    }


def summarize(rows):
    """Mean and standard deviation per policy."""
    summary = []
    for name in dict.fromkeys(
        r["policy"] for r in rows
    ):  # policies in the order they ran
        runs = [r for r in rows if r["policy"] == name]
        laps = [r["lap_time_s"] for r in runs if r["finished"]]
        stat = lambda key: (
            np.mean([r[key] for r in runs]),
            np.std([r[key] for r in runs]),
        )
        summary.append(
            {
                "policy": name,
                "runs": len(runs),
                "finish_pct": round(100 * np.mean([r["finished"] for r in runs]), 1),
                "return_mean": round(stat("return")[0], 2),
                "return_std": round(stat("return")[1], 2),
                "distance_mean": round(stat("distance_m")[0], 1),
                "distance_std": round(stat("distance_m")[1], 1),
                "lap_time_mean": round(np.mean(laps), 1) if laps else "",
                "lap_time_std": round(np.std(laps), 1) if laps else "",
                "offroad_pct_mean": round(stat("offroad_pct")[0], 1),
            }
        )
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# === Main ===
def main(policy_path=POLICY_PATH, n_runs=N_RUNS):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    environment.FRAME_SIZE = (
        VIDEO_SIZE  # record larger frames than the pixel policy trains on
    )
    env = environment.make_env("pixels")
    env.reset(seed=0)
    n_obs = len(env._state())
    rng = np.random.default_rng(0)
    policies = {
        "SAC (state)": load_sac(policy_path, n_obs, env.action_space),
        "follow path": follow_path,
        "full throttle": full_throttle,
        "random": lambda state: rng.uniform(
            env.action_space.low, env.action_space.high
        ).astype(np.float32),
    }

    # -- Race every policy on the same seeds; keep the video of SAC's best race --
    rows, best_return = [], -np.inf
    run_video, best_video = OUT_DIR / "current_run.mp4", OUT_DIR / "best_run.mp4"
    for name, act in policies.items():
        for seed in range(n_runs):
            record = name == "SAC (state)"
            stats = run(env, act, seed, run_video if record else None)
            rows.append({"policy": name, "seed": seed, **stats})
            if record and stats["return"] > best_return:
                best_return = stats["return"]
                os.replace(run_video, best_video)
            print(
                f"{name:13s} seed {seed:2d} | return {stats['return']:7.2f} | distance {stats['distance_m']:5.0f} m | "
                f"{'finished in ' + str(stats['lap_time_s']) + ' s' if stats['finished'] else 'cut off'} | "
                f"off-road {stats['offroad_pct']:4.1f}%"
            )
    env.close()
    if run_video.exists():
        run_video.unlink()

    # -- Save and print the results --
    summary = summarize(rows)
    write_csv(OUT_DIR / "runs.csv", rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    print(
        f"\n{'policy':13s} | finished | return (mean +/- std) | distance (m)  | lap time (s) | off-road"
    )
    for s in summary:
        lap = (
            f"{s['lap_time_mean']} +/- {s['lap_time_std']}"
            if s["lap_time_mean"] != ""
            else "-"
        )
        print(
            f"{s['policy']:13s} | {s['finish_pct']:6.1f}%  | {s['return_mean']:7.2f} +/- {s['return_std']:6.2f}  | "
            f"{s['distance_mean']:5.0f} +/- {s['distance_std']:4.0f} | {lap:12s} | {s['offroad_pct_mean']:5.1f}%"
        )
    print(
        f"\nSaved {OUT_DIR / 'runs.csv'}, {OUT_DIR / 'summary.csv'} and {best_video} (SAC return {best_return:.2f})"
    )


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else POLICY_PATH,
        int(sys.argv[2]) if len(sys.argv) > 2 else N_RUNS,
    )
