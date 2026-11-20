"""Races a policy on fixed seeds and writes one CSV row per race. Every run uses this script, so all
results share the same metrics and can be concatenated for the speed vs. recklessness comparison.
- Usage: python evaluate.py <policy> [--races 50] [--difficulty 2] [--out runs/eval]
- <policy> is a checkpoint (runs/<name>/final.pt), or a baseline: "random" or "full_throttle"
- Output: <out>/<name>.csv"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from environment import make_env
from SAC_CNN_Pixels import SACAgent
from train import to_env

CRASH_DROP = 5.0


def load_policy(name, action_dim):
    """Return a function (obs, h) -> (action in [-1, 1]^action_dim, h)."""
    if name == "random":
        return lambda obs, h: (np.random.uniform(-1, 1, action_dim), h)
    if name == "full_throttle":  # straight ahead; brake, drift and nitro off (-1 maps to 0)
        return lambda obs, h: (np.array([1.0, 0.0] + [-1.0] * (action_dim - 2)), h)
    checkpoint = torch.load(name, map_location="cpu", weights_only=False)
    agent = SACAgent(state_dim=checkpoint["config"]["embedding_dim"], action_dim=action_dim)
    for part in ("encoder", "belief", "actor"):
        getattr(agent, part).load_state_dict(checkpoint["agent"][part])
    return lambda obs, h: agent.act(obs, h, deterministic=True)


def race(env, policy, seed):
    """Run one race and return its metrics."""
    obs, _ = env.reset(seed=seed)
    h, steer_prev, speed_prev = None, 0.0, 0.0
    speeds, off_road, jerks, yaws, crashes = [], [], [], [], 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, h = policy(obs, h)
        obs, _, terminated, truncated, info = env.step(to_env(action, env.action_space))
        kart = env.world.players[0].kart
        speeds.append(kart.speed)
        off_road.append(not kart.is_on_road)
        jerks.append(abs(action[1] - steer_prev))
        yaws.append(abs(kart.angular_velocity[1]))
        crashes += speed_prev - kart.speed > CRASH_DROP
        steer_prev, speed_prev = action[1], kart.speed
    return {
        "seed": seed,
        "finished": int(terminated),
        "won": int(terminated and kart.position == 1),
        "finish_time": kart.finish_time if terminated else "",
        "steps": len(speeds),
        "distance": info["distance"],
        "mean_speed": np.mean(speeds),
        "offroad_frac": np.mean(off_road),
        "steer_jerk": np.mean(jerks),
        "mean_yaw_rate": np.mean(yaws),
        "crashes": crashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy")
    parser.add_argument("--races", type=int, default=50)
    parser.add_argument("--difficulty", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args()

    name = Path(args.policy).parent.name if args.policy.endswith(".pt") else args.policy
    env = make_env("pixels", args.difficulty)
    policy = load_policy(args.policy, env.action_space.shape[0])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for seed in range(args.races):  # same seeds for every policy for fair comparison
            rows.append({"run": name, "difficulty": args.difficulty, **race(env, policy, seed)})
            print(rows[-1], flush=True)
    finally:
        env.close()
        if rows:
            with (args.out / f"{name}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    for key in ("finished", "won", "mean_speed", "offroad_frac", "steer_jerk", "crashes"):
        print(f"{key:>14}: {np.mean([r[key] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
