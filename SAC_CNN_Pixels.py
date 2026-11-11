# === Soft Actor Critic based on Pixels ===

# TODO: / things that can be improved, if necessary
# tune the entropy parameter dynamically as training progresses (alpha)
# use EMA of the critic/actor weights to keep training stable (if you do this initialize with
# polyak_update(critic_1, target_critic_1, tau=1.0) so that training isnt extremely slow)
# decide how encoder/belief update gets trained, through actor or critic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

LOG_STD_MIN, LOG_STD_MAX = -20, 2


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
            nn.AdaptiveAvgPool2d((6, 8))
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
    def __init__(self, state_dim=128, action_dim=2, gamma=0.99, alpha=0.2, lr=3e-4):
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
        self.alpha = alpha
        self.gamma = gamma
        critic_params = (
            list(self.encoder.parameters()) +
            list(self.belief.parameters()) +
            list(self.critic1.parameters()) +
            list(self.critic2.parameters())
        )
        self.critic_optimizer = optim.Adam(critic_params, lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

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

    def update(self, batch):
        obs, a, r, terminated = (torch.as_tensor(x, device=self.device) for x in batch)
        a = a.float()
        r = r.float().unsqueeze(-1)  # (B, T, 1)
        terminated = terminated.float().unsqueeze(-1)

        beliefs, _ = self.belief(self.encoder(obs))  # (B, T+1, state_dim)
        b, bp = beliefs[:, :-1], beliefs[:, 1:]

        with torch.no_grad():
            ap, ap_log_prob = self.actor(bp)
            q_next = torch.min(self.critic1(bp, ap), self.critic2(bp, ap)) - self.alpha * ap_log_prob # entropy term to reward policy exploration
            q_target = r + self.gamma * (1 - terminated) * q_next

        q_pred1 = self.critic1(b, a)
        q_pred2 = self.critic2(b, a)
        q_loss1 = F.mse_loss(q_pred1, q_target)
        q_loss2 = F.mse_loss(q_pred2, q_target)
        total_critic_loss = q_loss1 + q_loss2
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        b = b.detach()
        a_pred, a_log_prob = self.actor(b)
        actor_loss = (self.alpha * a_log_prob - torch.min(self.critic1(b, a_pred), self.critic2(b, a_pred))).mean() # actor loss with reparameterization trick
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "loss/critic": total_critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "train/q1_mean": q_pred1.mean().item(),
            "train/q2_mean": q_pred2.mean().item(),
            "train/entropy_log_prob": a_log_prob.mean().item()
        }
