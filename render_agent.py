# === Render a Trained Pixel Agent to MP4 ===
"""Runs a trained pixel SAC agent (a train.py final.pt) for one race and saves it as a 1920x1080 MP4.
- Usage: python render_agent.py [--checkpoint runs/full_training_01/final.pt] [--out video.mp4] [--seed 0] [--stochastic]
- Two passes, because pystk2 renders at one resolution per process and the agent must see the 128x96
  frames it was trained on:
  1. A worker process lets the agent drive at the training resolution and records its actions.
  2. This process replays those actions in the same seeded race, rendered at 1920x1080, and writes the MP4.
  The kart's position is compared step by step, so a mismatched replay is reported."""

# === Imports ===
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

import environment  # imported first: sets up the STK DLLs on Windows
from SAC_CNN_Pixels import SACAgent
from train import to_env

# === Definitions ===
CHECKPOINT = Path("runs/full_training_01/final.pt")
VIDEO_SIZE = (1920, 1080)  # (width, height) of the video
STEP_TIME = 0.1  # seconds of game time per env step
FPS = round(1 / STEP_TIME)  # one frame per step, so the video plays in real time
MAX_DRIFT = 1.0  # meters the replay may differ from the agent's own drive before warning


def load_agent(path, action_dim):
    """Rebuild the SACAgent from a train.py checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    agent = SACAgent(state_dim=config["embedding_dim"], action_dim=action_dim, lr=config["lr"],
                     alpha=config["alpha"], gamma=config["gamma"])
    for name, weights in checkpoint["agent"].items():
        getattr(agent, name).load_state_dict(weights)
    return agent, checkpoint["step"]


def kart_location(env):
    return np.array(env.world.players[0].kart.location, dtype=np.float64)  # copy: pystk2 returns a live view


# === Pass 1: the agent drives (worker process) ===
def drive(checkpoint, seed, deterministic, results):
    """Run one race at the training resolution; send back the actions taken and the kart's positions."""
    env = environment.make_env("pixels")
    agent, trained_steps = load_agent(
        checkpoint, env.action_space.shape[0]
    )  # 6 actions: accel, steer, brake, drift, nitro, fire
    obs, _ = env.reset(seed=seed)
    hidden, actions, locations = None, [], [kart_location(env)]
    total, done = 0.0, False
    while not done:
        action, hidden = agent.act(obs, hidden, deterministic=deterministic)  # GRU memory carries over
        env_action = to_env(action, env.action_space)
        obs, reward, terminated, truncated, info = env.step(env_action)
        actions.append(env_action)
        locations.append(kart_location(env))
        total += reward
        done = terminated or truncated
    results.put({"actions": actions, "locations": locations, "return": total, "distance": info["distance"],
                 "finished": bool(terminated), "steps": env.steps, "trained_steps": trained_steps})
    env.close()


# === Pass 2: replay and record at 1920x1080 (this process) ===
def record(run, seed, out):
    """Replay the recorded actions in the same seeded race and write every frame to out."""
    # -- With Windows display scaling (e.g. 150%), an unaware process gets a smaller game window. --
    # Opt this process out so it renders at real pixels (the driving process stays as in training).
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass
    environment.FRAME_SIZE = VIDEO_SIZE  # render this process's frames at video resolution
    env = environment.make_env("pixels")
    frame, _ = env.reset(seed=seed)

    def to_video_size(frame):
        if frame.shape[1::-1] != VIDEO_SIZE:  # (width, height) differs: the window was limited by the screen
            return np.asarray(Image.fromarray(frame).resize(VIDEO_SIZE, Image.LANCZOS))
        return frame

    if frame.shape[1::-1] != VIDEO_SIZE:
        print(f"Note: STK rendered {frame.shape[1]}x{frame.shape[0]}; frames are resized to {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, quality=6, macro_block_size=1)  # 1080 isn't a multiple of 16
    writer.append_data(to_video_size(frame))
    drift = np.linalg.norm(kart_location(env) - run["locations"][0])
    for action, expected in zip(run["actions"], run["locations"][1:]):
        frame, *_ = env.step(action)
        writer.append_data(to_video_size(frame))
        drift = max(drift, np.linalg.norm(kart_location(env) - expected))
    writer.close()
    env.close()
    return drift


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=None, help="default: <checkpoint folder>/agent_seed<seed>_1080p.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean action")
    args = parser.parse_args()
    out = args.out or args.checkpoint.parent / f"agent_seed{args.seed}_1080p.mp4"

    # -- Pass 1 in its own process, at the training resolution --
    context = mp.get_context("spawn")
    results = context.Queue()
    worker = context.Process(target=drive, args=(str(args.checkpoint), args.seed, not args.stochastic, results))
    worker.start()
    run = results.get()  # read before join, so the worker can exit
    worker.join()
    lap = f"finished in {run['steps'] * STEP_TIME:.1f} s" if run["finished"] else "cut off"
    print(f"Agent ({run['trained_steps']} training steps): return {run['return']:.2f} | "
          f"distance {run['distance']:.0f} m | {lap} | {run['steps']} steps")

    # -- Pass 2 here, at 1920x1080 --
    drift = record(run, args.seed, out)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB, {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}, "
          f"{run['steps'] + 1} frames at {FPS} fps)")
    if drift > MAX_DRIFT:
        print(f"WARNING: the replay drifted up to {drift:.1f} m from the agent's own drive, "
              "so the video may not show exactly what the agent did")
    else:
        print(f"Replay matched the agent's drive (largest position difference {drift:.3f} m)")


if __name__ == "__main__":
    main()
