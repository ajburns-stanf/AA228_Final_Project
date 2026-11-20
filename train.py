"""Collect pixel transitions with SACAgent.act, then train from replay windows."""
import argparse
import ast
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


def parse_env_options(pairs):
    """['step_penalty=0.05', 'track=zengarden'] -> {'step_penalty': 0.05, 'track': 'zengarden'}."""
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep:
            raise ValueError(f'--env expects key=value, got {pair!r}')
        try:
            options[key] = ast.literal_eval(value)  # numbers, True/False
        except (ValueError, SyntaxError):
            options[key] = value  # plain strings, e.g. a track name
    return options


AGENT_MODULES = ('encoder', 'belief', 'actor', 'critic1', 'critic2', 'critic1_target', 'critic2_target')


def save_checkpoint(path, agent, args, step):
    """Save weights, optimizers, alpha, step and settings (replay isn't saved)."""
    config = {**vars(args), 'run_dir': str(args.run_dir)}
    # SACAgent is a plain class, so save its modules individually.
    models = {name: getattr(agent, name).state_dict() for name in AGENT_MODULES}
    torch.save({'step': step, 'config': config, 'agent': models, 'log_alpha': agent.log_alpha.item(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict()}, path)


def load_weights(agent, path):
    """Start from a checkpoint's networks. A behavior-cloning policy has no critics; those start fresh."""
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    for name, weights in checkpoint['agent'].items():
        getattr(agent, name).load_state_dict(weights)
    for live, target in (('critic1', 'critic1_target'), ('critic2', 'critic2_target')):
        if target not in checkpoint['agent']:  # older checkpoints: targets start equal to the critics
            getattr(agent, target).load_state_dict(getattr(agent, live).state_dict())
    if 'log_alpha' in checkpoint and agent.auto_alpha:
        agent.log_alpha.data.fill_(checkpoint['log_alpha'])
    return sorted(checkpoint['agent'])


def preload_demos(buffer, path):
    """Put recorded demonstrations (bc_pretrain.py's demos.npz) in the replay buffer before training."""
    demos = np.load(path)
    frames, obs_ix, next_ix = demos['frames'], demos['obs_ix'], demos['next_ix']
    for i in range(len(obs_ix)):
        buffer.add(frames[obs_ix[i]], demos['actions'][i], demos['rewards'][i], frames[next_ix[i]],
                   bool(demos['terminated'][i]), bool(demos['truncated'][i]))
    return len(obs_ix)


def train(args, env_factory=None):
    if env_factory is None:
        from environment import make_env
        env_factory = make_env
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    buffer = ReplayBuffer(args.buffer_size, args.sequence_length, args.seed)
    env = env_factory('pixels', args.difficulty, **parse_env_options(args.env))
    # Size the agent from the env: accel, steer, brake, drift, nitro.
    agent = SACAgent(state_dim=args.embedding_dim, action_dim=env.action_space.shape[0],
                     lr=args.lr, alpha=args.alpha, gamma=args.gamma, auto_alpha=not args.fixed_alpha,
                     tau=args.tau, augment=not args.no_augment)
    # Optionally start from a checkpoint (e.g. the behavior-cloning policy) and with demonstrations in replay.
    if args.init_from:
        print(f'Starting from {args.init_from}: {", ".join(load_weights(agent, args.init_from))}', flush=True)
    if args.demos:
        print(f'Preloaded {preload_demos(buffer, args.demos)} demonstration transitions', flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    step = n_updates = 0
    try:
        with (args.run_dir / 'episodes.csv').open('w', newline='') as episode_file, \
             (args.run_dir / 'updates.csv').open('w', newline='') as update_file:
            episodes = csv.writer(episode_file)
            updates = csv.writer(update_file)
            episodes.writerow(['step', 'episode', 'return', 'length', 'distance', 'terminated', 'truncated',
                               'finished', 'stuck'])
            updates.writerow(['step', 'critic_loss', 'actor_loss', 'q1_mean', 'q2_mean', 'alpha', 'log_prob'])
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
                        # The first --critic-warmup updates train only the critics, so an untrained
                        # critic can't pull a pretrained actor off course.
                        metrics = agent.update(batch, update_actor=n_updates >= args.critic_warmup)
                        n_updates += 1
                        updates.writerow([step, metrics['loss/critic'], metrics['loss/actor'],
                                          metrics['train/q1_mean'], metrics['train/q2_mean'],
                                          metrics['train/alpha'], metrics['train/entropy_log_prob']])
                    update_file.flush()

                # Save a checkpoint every --checkpoint-every steps, so one long run covers several lengths.
                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    save_checkpoint(args.run_dir / f'step_{step:06d}.pt', agent, args, step)

                if terminated or truncated:
                    episode += 1
                    episodes.writerow([step, episode, episode_return, episode_length,
                                       info.get('distance', ''), terminated, truncated,
                                       info.get('finished', ''), info.get('stuck', '')])
                    episode_file.flush()
                    print(f'step={step} episode={episode} return={episode_return:.3f} length={episode_length}', flush=True)
                    # 6. Start a new race with no memory from the previous one.
                    obs, _ = env.reset()
                    hidden = None
                    episode_return, episode_length = 0.0, 0
    finally:
        try:
            # Save weights and optimizers even when interrupted; replay isn't saved.
            save_checkpoint(args.run_dir / 'final.pt', agent, args, step)
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
    parser.add_argument('--alpha', type=float, default=0.2, help='starting alpha (learned unless --fixed-alpha)')
    parser.add_argument('--fixed-alpha', action='store_true', help='keep alpha constant instead of tuning it')
    parser.add_argument('--tau', type=float, default=0.005, help='target critic update rate')
    parser.add_argument('--no-augment', action='store_true', help='turn off random-shift image augmentation')
    parser.add_argument('--init-from', type=Path, default=None,
                        help='start from this checkpoint (e.g. runs/bc/policy.pt or a phase 1 policy)')
    parser.add_argument('--demos', type=Path, default=None, help='preload these demonstrations into replay')
    parser.add_argument('--critic-warmup', type=int, default=0,
                        help='first N updates train only the critics (use with --init-from)')
    parser.add_argument('--difficulty', type=int, choices=(0, 1, 2), default=2,
                        help='opponent AI: 0 novice, 1 intermediate, 2 expert')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/sac_pixels'))
    parser.add_argument('--checkpoint-every', type=int, default=25_000,
                        help='save step_XXXXXX.pt every N steps (0 = only final.pt)')
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                        help='env setting, repeatable, e.g. --env step_penalty=0.05 (see environment.DEFAULT_OPTIONS)')
    args = parser.parse_args(argv)
    for name in ('total_steps', 'learning_starts', 'buffer_size', 'batch_size',
                 'sequence_length', 'updates_per_step', 'embedding_dim'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.buffer_size < max(args.learning_starts, args.batch_size, args.sequence_length):
        parser.error('--buffer-size must cover learning-starts, batch-size and sequence-length')
    if args.lr <= 0 or args.alpha < 0 or not 0 <= args.gamma <= 1:
        parser.error('Require lr > 0, alpha >= 0, and gamma in [0, 1]')
    if args.checkpoint_every < 0 or args.critic_warmup < 0 or not 0 < args.tau <= 1:
        parser.error('Require --checkpoint-every >= 0, --critic-warmup >= 0 and 0 < --tau <= 1')
    for path in (args.init_from, args.demos):
        if path is not None and not path.exists():
            parser.error(f'{path} not found')
    try:
        parse_env_options(args.env)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == '__main__':
    train(parse_args())
