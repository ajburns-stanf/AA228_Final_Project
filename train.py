"""Collect pixel transitions with SACAgent.act, then train from replay windows."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from SAC_CNN_Pixels import SACAgent
from replay_buffer import ReplayBuffer


def to_env(action, action_space):
    """Normalized [-1, 1] -> acceleration [0, 1], steering [-1, 1]."""
    return (action_space.low + (action + 1) * 0.5 *
            (action_space.high - action_space.low)).astype(np.float32)


def train(args, env_factory=None):
    if env_factory is None:
        from environment import make_env
        env_factory = make_env
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    buffer = ReplayBuffer(args.buffer_size, args.sequence_length, args.seed)
    agent = SACAgent(state_dim=args.embedding_dim, lr=args.lr, alpha=args.alpha,
                     gamma=args.gamma)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = env_factory('pixels')
    step = 0
    try:
        with (args.run_dir / 'episodes.csv').open('w', newline='') as episode_file, \
             (args.run_dir / 'updates.csv').open('w', newline='') as update_file:
            episodes = csv.writer(episode_file)
            updates = csv.writer(update_file)
            episodes.writerow(['step', 'episode', 'return', 'length', 'distance', 'terminated', 'truncated'])
            updates.writerow(['step', 'critic_loss', 'actor_loss', 'q1_mean', 'q2_mean'])
            # 1. Reset starts a race and returns its first raw RGB screenshot.
            obs, _ = env.reset(seed=args.seed)
            hidden = None
            episode = episode_length = 0
            episode_return = 0.0
            for step in range(1, args.total_steps + 1):
                # Even during replay warmup, collect with the stochastic actor.
                # 2. Choose controls and carry GRU memory into the next step.
                action, hidden = agent.act(obs, hidden)
                # 3. Apply controls; the environment returns the next screenshot.
                next_obs, reward, terminated, truncated, info = env.step(to_env(action, env.action_space))
                # 4. Store normalized actions, rewards and both screenshots.
                buffer.add(obs, action, reward, next_obs, terminated, truncated)
                obs = next_obs
                episode_return += reward
                episode_length += 1

                # 5. After warmup, sample sequences and update the networks.
                if len(buffer) >= max(args.learning_starts, args.batch_size):
                    for _ in range(args.updates_per_step):
                        batch = buffer.sample(args.batch_size)
                        if batch is None:
                            break
                        metrics = agent.update(batch)
                        updates.writerow([step, metrics['loss/critic'], metrics['loss/actor'],
                                          metrics['train/q1_mean'], metrics['train/q2_mean']])
                    update_file.flush()

                if terminated or truncated:
                    episode += 1
                    episodes.writerow([step, episode, episode_return, episode_length,
                                       info.get('distance', ''), terminated, truncated])
                    episode_file.flush()
                    print(f'step={step} episode={episode} return={episode_return:.3f} length={episode_length}', flush=True)
                    # 6. Start a new race with no memory from the previous one.
                    obs, _ = env.reset()
                    hidden = None
                    episode_return, episode_length = 0.0, 0
    finally:
        try:
            # Save weights and optimizers even when interrupted; replay isn't saved.
            config = {**vars(args), 'run_dir': str(args.run_dir)}
            # SACAgent is a plain class, so save its five modules individually.
            models = {name: getattr(agent, name).state_dict() for name in
                      ('encoder', 'belief', 'actor', 'critic1', 'critic2')}
            torch.save({'step': step, 'config': config, 'agent': models,
                        'actor_optimizer': agent.actor_optimizer.state_dict(),
                        'critic_optimizer': agent.critic_optimizer.state_dict()},
                       args.run_dir / 'final.pt')
        finally:
            env.close()
    return agent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-steps', type=int, default=100_000)
    parser.add_argument('--learning-starts', type=int, default=1_000)
    parser.add_argument('--buffer-size', type=int, default=10_000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--updates-per-step', type=int, default=1)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/sac_pixels'))
    args = parser.parse_args(argv)
    for name in ('total_steps', 'learning_starts', 'buffer_size', 'batch_size',
                 'sequence_length', 'updates_per_step', 'embedding_dim'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.buffer_size < max(args.learning_starts, args.batch_size, args.sequence_length):
        parser.error('--buffer-size must cover learning-starts, batch-size and sequence-length')
    if args.lr <= 0 or args.alpha < 0 or not 0 <= args.gamma <= 1:
        parser.error('Require lr > 0, alpha >= 0, and gamma in [0, 1]')
    return args


if __name__ == '__main__':
    train(parse_args())
