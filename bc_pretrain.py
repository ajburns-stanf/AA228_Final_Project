"""Behavior-cloning warm start: record the path follower driving, then train the pixel policy to copy it.
- Usage: python bc_pretrain.py [--episodes 20] [--steps 3000] [--dagger-rounds 3] [--difficulty 0] [--out runs/bc]
- Teacher: the path follower steers toward the 2nd upcoming centerline point using the true state, at full
  throttle, brake/drift/nitro off. Student: the pixel policy (encoder + GRU + actor) sees only camera frames.
- While recording, random steering noise is added to the action that is executed (not to the label), so the
  student also sees how the teacher recovers from being off the ideal line.
- DAgger rounds: the student drives, the teacher labels every frame the student reaches, and the student
  retrains on all the data, so it learns to recover from its own mistakes.
- Outputs in <out>/: demos.npz (train.py --demos) and policy.pt (train.py --init-from, evaluate.py,
  render_agent.py)."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from environment import make_env  # imported first: sets up the STK DLLs on Windows
from SAC_CNN_Pixels import SACAgent, random_shift
from train import to_env

LABEL_LIMIT = 0.97  # targets of exactly +-1 would need an infinitely large pre-tanh output


def teacher(state):
    """Path follower's action in the policy's [-1, 1] units (-1 = not pressed for brake/drift/nitro)."""
    steer = np.clip(2.0 * np.arctan2(state[9], state[10]), -1, 1)
    return np.array([1.0, steer, -1.0, -1.0, -1.0], np.float32)


def record(env, episodes, noise, rng, student=None, first_seed=0):
    """Record races labeled by the teacher; each frame is stored once (a transition points at two frames).
    The teacher drives (with steering noise), or the student drives if given (DAgger)."""
    frames, data = [], {k: [] for k in ("obs_ix", "next_ix", "actions", "labels", "rewards",
                                        "terminated", "truncated", "episode")}
    who = "student" if student else "teacher"
    for episode in range(episodes):
        obs, _ = env.reset(seed=first_seed + episode)
        frames.append(obs)
        hidden, done = None, False
        while not done:
            label = teacher(env._state())  # what the teacher would do here, whoever is driving
            if student:
                action, hidden = student.act(obs, hidden, deterministic=True)
            else:
                action = label.copy()
                action[1] = np.clip(action[1] + rng.normal(0, noise), -1, 1)  # wobble off the ideal line
            obs, reward, terminated, truncated, info = env.step(to_env(action, env.action_space))
            frames.append(obs)
            for key, value in (("obs_ix", len(frames) - 2), ("next_ix", len(frames) - 1), ("actions", action),
                               ("labels", label), ("rewards", reward), ("terminated", terminated),
                               ("truncated", truncated), ("episode", episode)):
                data[key].append(value)
            done = terminated or truncated
        print(f"  {who} race {episode + 1}/{episodes}: {'finished' if info['finished'] else 'cut off'} "
              f"after {env.steps} steps, {info['distance']:.0f} m", flush=True)
    return {"frames": np.asarray(frames), **{k: np.asarray(v) for k, v in data.items()}}


def merge(a, b):
    """Combine two recordings, shifting b's frame indices and episode numbers past a's."""
    shift = {"obs_ix": len(a["frames"]), "next_ix": len(a["frames"]), "episode": a["episode"].max() + 1}
    return {k: np.concatenate([a[k], b[k] + shift.get(k, 0)]) for k in a}


def pretrain(agent, demos, steps, batch_size, length, lr, rng):
    """Train encoder + GRU + actor so the actor's mean action matches the teacher's label (MSE)."""
    episode = demos["episode"]
    starts = np.array([i for i in range(len(episode) - length + 1) if episode[i] == episode[i + length - 1]])
    params = list(agent.encoder.parameters()) + list(agent.belief.parameters()) + list(agent.actor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    for step in range(1, steps + 1):
        ix = rng.choice(starts, batch_size)[:, None] + np.arange(length)  # (B, T) transition indices
        obs = torch.as_tensor(demos["frames"][demos["obs_ix"][ix]], device=agent.device)
        labels = torch.as_tensor(demos["labels"][ix], device=agent.device).clamp(-LABEL_LIMIT, LABEL_LIMIT)
        if agent.augment:
            obs = random_shift(obs)  # same augmentation SAC uses
        beliefs, _ = agent.belief(agent.encoder(obs))
        actions, _ = agent.actor(beliefs, deterministic=True)
        loss = F.mse_loss(actions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == steps:
            print(f"  pretrain step {step}/{steps}: loss {loss.item():.4f}", flush=True)


def check(env, agent, races):
    """Drive the student (mean action) for a few races and report how it does."""
    for seed in range(races):
        obs, _ = env.reset(seed=1000 + seed)  # seeds the teacher never raced
        hidden, done = None, False
        while not done:
            action, hidden = agent.act(obs, hidden, deterministic=True)
            obs, _, terminated, truncated, info = env.step(to_env(action, env.action_space))
            done = terminated or truncated
        print(f"  student race {seed + 1}/{races}: {'finished' if info['finished'] else 'cut off'} after "
              f"{env.steps} steps, {info['distance']:.0f} m", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20, help="teacher races to record")
    parser.add_argument("--steps", type=int, default=3000, help="pretraining gradient steps")
    parser.add_argument("--noise", type=float, default=0.3, help="std of steering noise while recording")
    parser.add_argument("--dagger-rounds", type=int, default=3, help="rounds of student driving + relabeling")
    parser.add_argument("--dagger-episodes", type=int, default=5, help="student races per DAgger round")
    parser.add_argument("--dagger-steps", type=int, default=1000, help="extra pretraining steps per round")
    parser.add_argument("--difficulty", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--check-races", type=int, default=2, help="student test races at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/bc"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    env = make_env("pixels", args.difficulty)
    agent = SACAgent(state_dim=args.embedding_dim, action_dim=env.action_space.shape[0], lr=args.lr)
    try:
        print(f"Recording {args.episodes} teacher races...", flush=True)
        demos = record(env, args.episodes, args.noise, rng)
        print("Pretraining the pixel policy...", flush=True)
        pretrain(agent, demos, args.steps, args.batch_size, args.sequence_length, args.lr, rng)

        # -- DAgger: the student drives, the teacher labels what it sees, retrain on everything --
        for round_ in range(1, args.dagger_rounds + 1):
            print(f"DAgger round {round_}/{args.dagger_rounds}: student drives {args.dagger_episodes} races...",
                  flush=True)
            demos = merge(demos, record(env, args.dagger_episodes, args.noise, rng, student=agent,
                                        first_seed=100 * round_))
            pretrain(agent, demos, args.dagger_steps, args.batch_size, args.sequence_length, args.lr, rng)

        np.savez(args.out / "demos.npz", **demos)
        print(f"Saved {len(demos['actions'])} transitions to {args.out / 'demos.npz'}", flush=True)
        config = {"embedding_dim": args.embedding_dim, "lr": args.lr, "alpha": 0.2, "gamma": 0.99,
                  "difficulty": args.difficulty,
                  "bc_episodes": args.episodes, "bc_steps": args.steps, "bc_noise": args.noise,
                  "dagger_rounds": args.dagger_rounds, "dagger_episodes": args.dagger_episodes}
        torch.save({"step": 0, "config": config,
                    "agent": {name: getattr(agent, name).state_dict() for name in ("encoder", "belief", "actor")}},
                   args.out / "policy.pt")
        print(f"Saved {args.out / 'policy.pt'}", flush=True)

        if args.check_races:
            print("Checking the student:", flush=True)
            check(env, agent, args.check_races)
    finally:
        env.close()


if __name__ == "__main__":
    main()
