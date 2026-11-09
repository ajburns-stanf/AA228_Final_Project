# === Soft Actor Critic based on State ===
"""Basic SAC following Haarnoja et al. 2018 (SAC_Paper.pdf, Algorithm 1), trained on the 18-float state.
Simplified from the paper: one Q-network instead of two.
- Networks: policy pi_phi(a|s), soft Q-function Q_theta(s, a), soft value V_psi(s) and its target copy V_psi_bar(s)
- Temperature: folded into the reward, alpha = 1 / REWARD_SCALE (Section 3.2, Table 2)
- "Eq. N" = SAC paper; "DM N" = Algorithms for Decision Making (Kochenderfer, Wheeler, Wray), equation N
- Textbook links: SAC is an actor-critic method (DM ch. 13) whose critic is trained like Q-learning with
  function approximation and experience replay (DM 17.6-17.7). The closest textbook method is the
  deterministic policy gradient (DM 13.3), with two changes from the paper: a stochastic policy rewarded
  for randomness (entropy, -log pi) and the reparameterization trick (Eq. 11). Neither is in the textbook.
- Outputs in runs/sac_state/ (git-ignored): log.csv (one row per run), best_policy.pt, final_policy.pt"""

# === Imports ===
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from environment import make_env  # imported first: sets up the STK DLLs on Windows

# === Definitions (paper Appendix D, Table 1) ===
LR = 3e-4  # Adam learning rate for all networks (the textbook's alpha in DM 17.21, not SAC's temperature)
GAMMA = 0.99  # discount
BUFFER_SIZE = 1_000_000  # replay buffer size
HIDDEN = 256  # 2 hidden layers of 256 ReLU units in every network
BATCH_SIZE = 256  # samples per minibatch
TAU = 0.005  # target smoothing coefficient
REWARD_SCALE = 20.0  # Table 2 uses 5-20; higher = less random policy

# -- Practical additions (not in Algorithm 1) --
START_STEPS = 5_000  # random actions to fill the buffer before learning starts
TOTAL_STEPS = 20_000  # environment steps to train for
EVAL_EVERY = 10  # runs between evaluation runs with the mean action
RUN_DIR = (
    Path(__file__).parent / "runs" / "sac_state"
)  # where the log and policies are saved
LOG_STD_MIN, LOG_STD_MAX = (
    -20,
    2,
)  # keeps the policy's standard deviation in a sane range


def mlp(n_in, n_out):
    """2 hidden layers of 256 ReLU units (Table 1)."""
    return nn.Sequential(
        nn.Linear(n_in, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, n_out),
    )


# === Networks ===
class Policy(nn.Module):
    """pi_phi(a|s): a Gaussian over u, squashed by tanh into actions a in (-1, 1) (Appendix C)."""

    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = mlp(n_obs, 2 * n_act)  # outputs the mean and log std of u

    def forward(self, s):
        mean, log_std = self.net(s).chunk(2, dim=-1)
        std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

        # -- Reparameterized sample: a = f_phi(eps; s) = tanh(mean + std * eps), eps ~ N(0, I)  (Eq. 11) --
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)

        # -- log pi(a|s) = log mu(u|s) - sum_i log(1 - tanh^2(u_i))  (Eq. 21) --
        log_prob = torch.distributions.Normal(mean, std).log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, log_prob, torch.tanh(mean)  # sample, its log-probability, mean action


class QNet(nn.Module):
    """Q_theta(s, a): soft Q-value of taking action a in state s."""

    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = mlp(n_obs + n_act, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class VNet(nn.Module):
    """V_psi(s): soft state value, V(s) = E_a~pi[Q(s, a) - log pi(a|s)]  (Eq. 3).
    The textbook's U(s) (DM 13.4), plus the entropy bonus -log pi."""

    def __init__(self, n_obs):
        super().__init__()
        self.net = mlp(n_obs, 1)

    def forward(self, s):
        return self.net(s).squeeze(-1)


# === Replay Buffer ===
class ReplayBuffer:
    """D: past transitions (s, a, r, s', done), sampled uniformly at random.
    Experience replay (DM 17.7): each update averages over a random batch of m_grad tuples (DM 17.22)."""

    def __init__(self, n_obs, n_act, size=BUFFER_SIZE):
        self.s = np.zeros((size, n_obs), np.float32)
        self.a = np.zeros((size, n_act), np.float32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, n_obs), np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.count, self.i = size, 0, 0

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.done[self.i] = s2, done
        self.i = (self.i + 1) % self.size  # overwrite the oldest once full
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        ix = np.random.randint(0, self.count, batch_size)
        return [
            torch.as_tensor(x[ix]) for x in (self.s, self.a, self.r, self.s2, self.done)
        ]


# === SAC ===
class SAC:
    def __init__(self, n_obs, n_act):
        self.policy = Policy(n_obs, n_act)
        self.q = QNet(n_obs, n_act)
        self.v = VNet(n_obs)
        self.v_target = VNet(n_obs)
        self.v_target.load_state_dict(
            self.v.state_dict()
        )  # psi_bar starts equal to psi

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=LR)

    def act(self, s, deterministic=False):
        """Sample a_t ~ pi(a_t|s_t), or take the mean action (for evaluation)."""
        with torch.no_grad():
            a, _, mean_action = self.policy(torch.as_tensor(s).unsqueeze(0))
        return (mean_action if deterministic else a)[0].numpy()

    def update(self, batch):
        """One gradient step of Algorithm 1: value, Q, policy, then the target value network."""
        s, a, r, s2, done = batch

        # -- Value: J_V = E_s[1/2 (V(s) - E_a~pi[Q(s, a) - log pi(a|s)])^2], a from the current policy  (Eq. 5) --
        # Textbook: the critic loss l(phi) = 1/2 E_s[(U_phi(s) - U^pi(s))^2]  (DM 13.4), with U^pi(s)
        # estimated as Q(s, a) - log pi(a|s) instead of from rollouts
        with torch.no_grad():
            a_new, log_prob, _ = self.policy(s)
            v_goal = self.q(s, a_new) - log_prob
        v_loss = 0.5 * F.mse_loss(self.v(s), v_goal)
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # -- Q: J_Q = E[1/2 (Q(s, a) - Q_hat(s, a))^2], Q_hat = r + gamma * V_target(s')  (Eqs. 7-8) --
        # Textbook: Q-learning with function approximation, l(theta) = 1/2 E[(Q*(s, a) - Q_theta(s, a))^2]
        # (DM 17.16), with Q*(s, a) ~ r + gamma max_a' Q(s', a')  (DM 17.20; Bellman equation DM 17.9).
        # SAC replaces max_a' Q(s', a') with the soft value V(s'). Same loss shape as DM 13.22.
        # (1 - done): no future value after the race is finished
        with torch.no_grad():
            q_hat = r + GAMMA * (1 - done) * self.v_target(s2)
        q_loss = 0.5 * F.mse_loss(self.q(s, a), q_hat)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # -- Policy: J_pi = E[log pi(f(eps; s)|s) - Q(s, f(eps; s))], reparameterized  (Eq. 12) --
        # Textbook: the deterministic policy gradient maximizes U(theta) = E_s[Q_phi(s, pi_theta(s))]
        # (DM 13.24), with the gradient flowing through the action into Q (DM 13.26). SAC does the same
        # through a = f(eps; s), and adds the entropy term log pi to keep the policy exploring.
        a_new, log_prob, _ = self.policy(s)
        policy_loss = (log_prob - self.q(s, a_new)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # -- Target value: psi_bar <- tau * psi + (1 - tau) * psi_bar  (Algorithm 1; not in the textbook) --
        # A slowly moving copy of V for the Q target, so that target doesn't shift every update
        with torch.no_grad():
            for p, p_target in zip(self.v.parameters(), self.v_target.parameters()):
                p_target.mul_(1 - TAU).add_(TAU * p)


# === Training ===
def to_env(a, action_space):
    """tanh range (-1, 1) -> env range (acceleration 0..1, steer -1..1)."""
    return action_space.low + (a + 1) / 2 * (action_space.high - action_space.low)


def evaluate(env, agent):
    """One run with the policy's mean action (no exploration)."""
    s, _ = env.reset()
    total, done = 0.0, False
    while not done:
        s, r, terminated, truncated, info = env.step(
            to_env(agent.act(s, deterministic=True), env.action_space)
        )
        total += r
        done = terminated or truncated
    return total, info["distance"], terminated


def train(total_steps=TOTAL_STEPS, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env("state")
    n_obs, n_act = env.observation_space.shape[0], env.action_space.shape[0]
    agent, buffer = SAC(n_obs, n_act), ReplayBuffer(n_obs, n_act)

    # -- Outputs: one CSV row per run (the learning curve), plus the best and final policies --
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(RUN_DIR / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "episode", "kind", "return", "distance", "finished"])
    best_eval_return = -np.inf

    s, _ = env.reset(seed=seed)
    episode, episode_return = 0, 0.0
    for step in range(1, total_steps + 1):
        # -- Environment step: a_t ~ pi(a_t|s_t), s_t+1 ~ p(s_t+1|s_t, a_t), add to D --
        if step <= START_STEPS:
            a = np.random.uniform(-1, 1, n_act).astype(np.float32)
        else:
            a = agent.act(s)
        s2, r, terminated, truncated, info = env.step(to_env(a, env.action_space))
        buffer.add(
            s, a, REWARD_SCALE * r, s2, float(terminated)
        )  # truncated runs still have a future
        s, episode_return = s2, episode_return + r

        # -- Gradient step --
        if step > START_STEPS:
            agent.update(buffer.sample(BATCH_SIZE))

        # -- End of a run: log it, sometimes evaluate, start a new race --
        if terminated or truncated:
            episode += 1
            log.writerow(
                [
                    step,
                    episode,
                    "train",
                    round(episode_return, 3),
                    round(info["distance"], 1),
                    int(terminated),
                ]
            )
            print(
                f"step {step:7d} | episode {episode:4d} | return {episode_return:7.2f} | "
                f"distance {info['distance']:5.0f} m | {'finished' if terminated else 'cut off'}"
            )
            if episode % EVAL_EVERY == 0:
                eval_return, eval_distance, eval_finished = evaluate(env, agent)
                log.writerow(
                    [
                        step,
                        episode,
                        "eval",
                        round(eval_return, 3),
                        round(eval_distance, 1),
                        int(eval_finished),
                    ]
                )
                new_best = eval_return > best_eval_return
                if new_best:  # keep the best policy, not just the last one
                    best_eval_return = eval_return
                    torch.save(agent.policy.state_dict(), RUN_DIR / "best_policy.pt")
                print(
                    f"  eval (mean action): return {eval_return:7.2f} | distance {eval_distance:5.0f} m"
                    f"{' | finished' if eval_finished else ''}{' | new best, saved' if new_best else ''}"
                )
            log_file.flush()  # keep the log on disk even if training is stopped early
            s, _ = env.reset()
            episode_return = 0.0

    torch.save(agent.policy.state_dict(), RUN_DIR / "final_policy.pt")
    log_file.close()
    env.close()
    return agent


if __name__ == "__main__":
    train()
