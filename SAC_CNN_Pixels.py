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

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CNNEncoder, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        embedding = self.fc(x)
        return embedding
    
class Actor(nn.Module):
    def __init__(self, state_dim=128, action_dim=4, hidden_dim=256):
        super(Actor, self).__init__()
        self.mlpblock = nn.sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        x = self.mlpblock(x)
        means = torch.tensor(x[..., :2])
        log_stds = torch.tensor(x[..., 2:])
        stds = torch.exp(log_stds)
        dist = Normal(means, stds)
        raw_actions = dist.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = dist.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return actions, log_prob

class Critic(nn.Module):
    def __init__(self, input_dim=130, hidden_dim=256):
        super(Actor, self).__init__()
        self.mlpblock = nn.sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.mlpblock(x)
        return x
    
class SACAGent():
    # get observation from the environment, the main training script will supply it and call this function
    # push it through the CNN
    # update belief about the system through GRU with the embedding
    # this feeds into policy, which generates the action (start off with random policy but we can kickstart the policy later with normal behavior cloning)
    # this also is fed into the critics
    # calculate loss, update weights

    def __init__(self, state_dim, action_dim, gamma, alpha, lr):
        self.encoder = CNNEncoder(state_dim)
        self.belief = nn.GRU(state_dim, state_dim)
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim+action_dim)
        self.critic2 = Critic(state_dim+action_dim)
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.belief.to(self.device)
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)

    def polyak_update(live_model, target_model, tau=0.005):
        with torch.no_grad():
            for live_param, target_param in zip(live_model.parameters(), target_model.parameters()):
                target_param.data.copy_(
                    tau * live_param.data + (1.0 - tau) * target_param.data
                )

    def act(self, obs):
        belief = self.belief(self.encoder(obs))
        return self.actor(belief)
    
    def update(self, batch):
        s, a, r, sp = batch
        # s is batch by pixel dimensions
        b = self.belief(self.encoder(s))
        with torch.no_grad():
            ap, ap_log_prob = self.act(sp)
            q_target = r + self.gamma*(torch.min(self.critic1(sp, ap), self.critic2(sp, ap)) - self.alpha*ap_log_prob)

        q_pred1 = self.critic1(torch.concatenate([b, a], dim=1))
        q_pred2 = self.critic2(torch.concatenate([b, a], dim=1))
        q_loss1 = F.mse_loss(q_pred1, q_target)
        q_loss2 = F.mse_loss(q_pred2, q_target)
        total_critic_loss = q_loss1 + q_loss2
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        a_pred, a_log_prob = self.act(s) 
        q_act_pred1 = self.critic1(torch.concatenate([b, a_pred], dim=1))
        q_act_pred2 = self.critic2(torch.concatenate([b, a_pred], dim=1))
        actor_loss = (self.alpha * a_log_prob - torch.min(q_act_pred1, q_act_pred2)).mean()
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
