"""Races a policy on fixed seeds and writes one CSV row per race. Every run uses this script, so all
results share the same metrics and can be concatenated for the speed vs. recklessness comparison.
- Usage: python evaluate.py <policy> [--races 50] [--difficulty N] [--out runs/eval] [--name NAME] [--env KEY=VALUE]
- --difficulty defaults to the difficulty a checkpoint was trained at (baselines: expert)
- <policy> is a checkpoint (runs/<name>/final.pt), or a baseline: "random" or "full_throttle"
- --env changes env settings for every race (e.g. --env track=zengarden), see environment.DEFAULT_OPTIONS
- Output: <out>/<name>.csv (name defaults to the checkpoint's folder or the baseline's name)
- brake/drift/nitro_frac: share of steps the policy pressed each (with no_brakes, brake presses do nothing)"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from environment import DIFFICULTY, PRESS_THRESHOLD, make_env
from SAC_CNN_Pixels import SACAgent
from train import parse_env_options, to_env, training_difficulty

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
    speeds, off_road, jerks, yaws, presses, crashes = [], [], [], [], [], 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, h = policy(obs, h)
        env_action = to_env(action, env.action_space)
        presses.append(env_action[2:5] > PRESS_THRESHOLD)  # brake, drift, nitro pressed this step?
        obs, _, terminated, truncated, info = env.step(env_action)
        kart = env.world.players[0].kart
        speeds.append(kart.speed)
        off_road.append(not kart.is_on_road)
        jerks.append(abs(action[1] - steer_prev))
        yaws.append(abs(kart.angular_velocity[1]))
        crashes += speed_prev - kart.speed > CRASH_DROP
        steer_prev, speed_prev = action[1], kart.speed
    finished = info["finished"]  # terminated also covers getting stuck
    brake, drift, nitro = np.mean(presses, axis=0)
    return {
        "seed": seed,
        "finished": int(finished),
        "won": int(finished and kart.position == 1),
        "stuck": int(info["stuck"]),
        "finish_time": kart.finish_time if finished else "",
        "steps": len(speeds),
        "distance": info["distance"],
        "mean_speed": np.mean(speeds),
        "offroad_frac": np.mean(off_road),
        "steer_jerk": np.mean(jerks),
        "mean_yaw_rate": np.mean(yaws),
        "crashes": crashes,
        "brake_frac": brake,
        "drift_frac": drift,
        "nitro_frac": nitro,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy")
    parser.add_argument("--races", type=int, default=50)
    parser.add_argument("--difficulty", type=int, choices=(0, 1, 2), default=None,
                        help="opponent AI (default: the checkpoint's training difficulty; baselines: 2)")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    parser.add_argument("--name", default=None, help="output CSV name (default: checkpoint folder or baseline)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="env setting for every race, repeatable, e.g. --env track=zengarden")
    args = parser.parse_args()

    name = args.name or (Path(args.policy).parent.name if args.policy.endswith(".pt") else args.policy)
    if args.difficulty is None:  # the opponent it trained against (difficulty also changes our kart's speed)
        args.difficulty = training_difficulty(args.policy, DIFFICULTY) if args.policy.endswith(".pt") else DIFFICULTY
    env = make_env("pixels", args.difficulty, **parse_env_options(args.env))
    policy = load_policy(args.policy, env.action_space.shape[0])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for seed in range(args.races):  # same seeds for every policy for fair comparison
            rows.append({"run": name, "difficulty": args.difficulty, "env": " ".join(args.env),
                         **race(env, policy, seed)})
            print(rows[-1], flush=True)
    finally:
        env.close()
        if rows:
            with (args.out / f"{name}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    for key in ("finished", "won", "stuck", "mean_speed", "offroad_frac", "steer_jerk", "crashes", "brake_frac",
                "drift_frac", "nitro_frac"):
        print(f"{key:>14}: {np.mean([r[key] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
