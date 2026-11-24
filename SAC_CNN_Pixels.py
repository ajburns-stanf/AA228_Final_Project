# === Soft Actor Critic based on Pixels ===

# Stability: slowly updated target critics (polyak averaging) and automatic tuning of alpha.
# Sample efficiency: random-shift image augmentation (DrQ-style) on every training sequence.

# TODO: / things that can be improved, if necessary
# decide how encoder/belief update gets trained, through actor or critic

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

LOG_STD_MIN, LOG_STD_MAX = -20, 2
SHIFT_PAD = 4  # pixels: random-shift augmentation moves frames by up to this much


def random_shift(obs, pad=SHIFT_PAD):
    """Shift every frame of a sequence by the same random few pixels (edges padded by repetition).
    obs: (B, T, H, W, 3) with values 0..255; returns the same shape as float."""
    B, T, H, W, C = obs.shape
    x = obs.permute(0, 1, 4, 2, 3).reshape(B, T * C, H, W).float()
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    dx, dy = torch.randint(0, 2 * pad + 1, (2, B)).tolist()
    x = torch.stack([x[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])
    return x.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)


class MPSCompatibleAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, x):
        if x.device.type == "mps" and (
            x.shape[-2] % self.output_size[0] or x.shape[-1] % self.output_size[1]
        ):
            # CPU pooling supports arbitrary sizes and preserves autograd through copies.
            return super().forward(x.cpu()).to(x.device)
        return super().forward(x)


class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d((6, 8))
        )
        self.fc = nn.Linear(64 * 6 * 8, embedding_dim)

    def forward(self, x):
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:]).permute(0, 3, 1, 2).float() / 255.0
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        embedding = self.fc(x)
        return embedding.reshape(*lead, -1)


class Actor(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=256):
        super().__init__()
        self.mlpblock = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)  # each action parametrized by normal distribution, mean + std
        )

    def forward(self, x, deterministic=False):
        x = self.mlpblock(x)
        means, log_stds = x.chunk(2, dim=-1)
        log_stds = log_stds.clamp(LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(means, torch.exp(log_stds))
        raw_actions = means if deterministic else dist.rsample()
        actions = torch.tanh(raw_actions)
        # Stable form of log(1 - tanh(raw_actions)**2).
        correction = 2 * (0.6931471805599453 - raw_actions - F.softplus(-2 * raw_actions))
        log_prob = dist.log_prob(raw_actions) - correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return actions, log_prob


class Critic(nn.Module):
    def __init__(self, input_dim=130, hidden_dim=256):
        super().__init__()
        self.mlpblock = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, belief, action):
        return self.mlpblock(torch.cat([belief, action], dim=-1))


class SACAgent:
    def __init__(self, state_dim=128, action_dim=2, gamma=0.99, alpha=0.2, lr=3e-4,
                 auto_alpha=True, tau=0.005, augment=True):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.encoder = CNNEncoder(state_dim).to(self.device)
        self.belief = nn.GRU(state_dim, state_dim, batch_first=True).to(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim + action_dim).to(self.device)
        self.critic2 = Critic(state_dim + action_dim).to(self.device)
        # Target critics: slow copies used for the Q target, so the target doesn't move every update
        self.critic1_target = copy.deepcopy(self.critic1).requires_grad_(False)
        self.critic2_target = copy.deepcopy(self.critic2).requires_grad_(False)
        self.gamma = gamma
        self.tau = tau
        self.augment = augment
        # Alpha (entropy weight): learned to keep the policy's entropy near -action_dim when auto_alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(math.log(alpha), device=self.device, requires_grad=auto_alpha)
        critic_params = (
            list(self.encoder.parameters()) +
            list(self.belief.parameters()) +
            list(self.critic1.parameters()) +
            list(self.critic2.parameters())
        )
        self.critic_optimizer = optim.Adam(critic_params, lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        if auto_alpha:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @staticmethod
    def polyak_update(live_model, target_model, tau=0.005):
        with torch.no_grad():
            for live_param, target_param in zip(live_model.parameters(), target_model.parameters()):
                target_param.data.copy_(
                    tau * live_param.data + (1.0 - tau) * target_param.data
                )

    @torch.no_grad()
    def act(self, obs, h=None, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)[None, None]  # (1, 1, H, W, 3)
        b, h = self.belief(self.encoder(obs), h)
        action, _ = self.actor(b[:, 0], deterministic)
        return action[0].cpu().numpy(), h

    def update(self, batch, update_actor=True):
        """One gradient step. update_actor=False trains only the critics (used for a critic warm-up)."""
        obs = torch.as_tensor(batch[0], device=self.device)
        # Convert before transferring: MPS cannot hold NumPy's default float64.
        a, r, terminated = (
            torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[1:]
        )
        if self.augment:
            obs = random_shift(obs)  # same few-pixel shift across each sequence
        a = a.float()
        r = r.float().unsqueeze(-1)  # (B, T, 1)
        terminated = terminated.float().unsqueeze(-1)
        alpha = self.log_alpha.exp().detach()

        beliefs, _ = self.belief(self.encoder(obs))  # (B, T+1, state_dim)
        b, bp = beliefs[:, :-1], beliefs[:, 1:]

        with torch.no_grad():
            ap, ap_log_prob = self.actor(bp)
            q_next = torch.min(self.critic1_target(bp, ap), self.critic2_target(bp, ap)) - alpha * ap_log_prob # entropy term to reward policy exploration
            q_target = r + self.gamma * (1 - terminated) * q_next

        q_pred1 = self.critic1(b, a)
        q_pred2 = self.critic2(b, a)
        q_loss1 = F.mse_loss(q_pred1, q_target)
        q_loss2 = F.mse_loss(q_pred2, q_target)
        total_critic_loss = q_loss1 + q_loss2
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        self.polyak_update(self.critic1, self.critic1_target, self.tau)
        self.polyak_update(self.critic2, self.critic2_target, self.tau)

        metrics = {
            "loss/critic": total_critic_loss.item(),
            "loss/actor": float("nan"),
            "train/q1_mean": q_pred1.mean().item(),
            "train/q2_mean": q_pred2.mean().item(),
            "train/entropy_log_prob": float("nan"),
        }
        if update_actor:
            b = b.detach()
            a_pred, a_log_prob = self.actor(b)
            actor_loss = (alpha * a_log_prob - torch.min(self.critic1(b, a_pred), self.critic2(b, a_pred))).mean() # actor loss with reparameterization trick
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if self.auto_alpha:  # raise alpha when entropy is below target, lower it when above
                alpha_loss = -(self.log_alpha * (a_log_prob.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            metrics["loss/actor"] = actor_loss.item()
            metrics["train/entropy_log_prob"] = a_log_prob.mean().item()
        metrics["train/alpha"] = self.alpha
        return metrics
