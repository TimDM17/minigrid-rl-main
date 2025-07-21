import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import copy
import sys
import os

from torch_ac.algos.base import BaseAlgo
from torch_ac.utils import DictList

# --- Q-Network (Critic) Definition ---
class QNetwork(nn.Module):
    """
    A separate Q-Network (Critic) for SAC.
    It estimates the Q-value (expected return) for a given observation.
    It is designed to be separate from the actor model
    """
    def __init__(self, acmodel):
        super().__init__()
        # The critic needs to process images, so we copy the convolutional layers
        # from the main actor-critic model
        self.image_conv = copy.deepcopy(acmodel.image_conv)
        
        # The embedding size must match the output of the convolutional layers
        self.embedding_size = acmodel.image_embedding_size
        
        # The head of the network is a linear layer that outputs a single Q-value
        self.q_head = nn.Linear(self.embedding_size, 1)

    def forward(self, obs):
        # Handle different observation types
        if hasattr(obs, 'image'):  # For DictList or objects with .image attribute
            x = obs.image
        elif isinstance(obs, dict): # For standard dictionaries
            x = obs["image"]
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
        
        # Check the shape and handle it appropriately
        if len(x.shape) == 4:  # Batch of images
            if x.shape[1] == 3:  # Already in PyTorch format (B, C, H, W)
                pass  # No change needed
            elif x.shape[3] == 3:  # In format (B, H, W, C)
                x = x.permute(0, 3, 1, 2)  # Convert to PyTorch format
        elif len(x.shape) == 3:  # Single image
            if x.shape[0] == 3:  # Already in PyTorch format (C, H, W)
                x = x.unsqueeze(0)  # Add batch dimension
            elif x.shape[2] == 3:  # In format (H, W, C)
                x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to PyTorch format
        
        embedding = self.image_conv(x)
        embedding = embedding.reshape(embedding.shape[0], -1)
        q_value = self.q_head(embedding)
        return q_value

# --- SAC Algorithm Implementation ---
class SACAlgo(BaseAlgo):
    """
    The Soft Actor-Critic (SAC) algorithm.
    This implementation uses twin critics and automatic temperature tuning
    """
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 adam_eps=1e-8, batch_size=256, tau=0.005, alpha=0.2, target_update_interval=1,
                 replay_size=1000000, automatic_entropy_tuning=True, preprocess_obss=None, reshape_reward=None):
        
        # Call the base class constructor. Some parameters like gae_lambda are not used by SAC
        # but are part of the base class signature
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        # --- SAC specific parameters ---
        self.batch_size = batch_size
        self.tau = tau  # Soft update rate for target networks
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # --- Initialize Replay Buffer ---
        self.replay_buffer = ReplayBuffer(replay_size)
        
        # --- Initialize Networks ---
        # Twin Critics and their targets for stable Q-learning
        self.critic1 = QNetwork(self.acmodel).to(self.device)
        self.critic2 = QNetwork(self.acmodel).to(self.device)
        self.target_critic1 = QNetwork(self.acmodel).to(self.device)
        self.target_critic2 = QNetwork(self.acmodel).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Target Actor
        self.target_acmodel = copy.deepcopy(self.acmodel)
        self.target_acmodel.load_state_dict(self.acmodel.state_dict())
        self.target_acmodel.to(self.device)

        # --- Initialize Optimizers ---
        self.policy_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=lr, eps=adam_eps)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr, eps=adam_eps)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr, eps=adam_eps)
        
        self.optimizer = self.policy_optimizer # Alias for compatibility with train.py
        
        # --- Automatic Temperature (alpha) Tuning ---
        if self.automatic_entropy_tuning:
            # Target entropy is usually set to the negative of the action space dimension
            self.target_entropy = -np.log(1.0 / self.env.action_space.n) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr, eps=adam_eps)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
        
        self.update_counter = 0

    def update_parameters(self, exps):
        """
        Updates the actor and critic networks based on experiences sampled from the replay buffer.
        """
        # Add collected experiences to the replay buffer
        for i in range(len(exps.obs)):
            # For off-policy learning, we need to construct the (s, a, r, s', done) tuple
            # The 'next_obs' is the observation from the next step
            next_obs = exps.obs[i + 1] if i < len(exps.obs) - 1 else exps.obs[i]
            done = exps.mask[i + 1] == 0 if i < len(exps.obs) - 1 else True
            mask = exps.mask[i] if hasattr(exps, 'mask') else 1.0
            self.replay_buffer.add(exps.obs[i], exps.action[i], exps.reward[i], next_obs, done, mask)

        # Don't update until the buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {
                "policy_loss": 0,
                "value_loss": 0,
                "entropy": 0,
                "grad_norm": 0,
                "alpha": self.alpha.item(),
                "value": 0 
            }

        # --- Sample a batch from the replay buffer ---
        obs, actions, rewards, next_obs, dones, masks = self.replay_buffer.sample(self.batch_size, self.device)
        
        # --- Critic Update ---
        with torch.no_grad():
            # Get next action and log probability from the target policy
            next_dist, _, _ = self.target_acmodel(next_obs, None)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions).unsqueeze(1)

            # Get Q-values from target critics for the next state
            target_q1 = self.target_critic1(next_obs)
            target_q2 = self.target_critic2(next_obs)
            
            # Use the minimum of the two target Q-values to prevent overestimation
            min_target_q = torch.min(target_q1, target_q2)
            
            # Calculate the soft Q-target
            soft_q_target = rewards + (1.0 - dones) * self.discount * (min_target_q - self.alpha * next_log_probs)

        # Get current Q-estimates from both critics
        current_q1 = self.critic1(obs)
        current_q2 = self.critic2(obs)
        
        # Calculate the loss for each critic
        critic1_loss = F.mse_loss(current_q1, soft_q_target)
        critic2_loss = F.mse_loss(current_q2, soft_q_target)
        critic_loss = critic1_loss + critic2_loss

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # --- Actor and Alpha Update ---
        # Get action distribution from the current policy
        dist, _, _ = self.acmodel(obs, None)
        
        # Sample actions and compute their log probabilities
        new_actions = dist.sample()
        log_probs = dist.log_prob(new_actions).unsqueeze(1)
        entropy = dist.entropy().mean()

        # Get Q-values for the new actions from the critics
        q1_new_actions = self.critic1(obs)
        q2_new_actions = self.critic2(obs)
        min_q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        # Calculate actor loss
        policy_loss = (self.alpha * log_probs - min_q_new_actions).mean()

        # Optimize the actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Update temperature alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

        # --- Soft Update Target Networks ---
        self.update_counter += 1
        if self.update_counter % self.target_update_interval == 0:
            self._soft_update_target(self.target_acmodel, self.acmodel)
            self._soft_update_target(self.target_critic1, self.critic1)
            self._soft_update_target(self.target_critic2, self.critic2)

        # --- Log metrics ---
        logs = {
            "policy_loss": policy_loss.item(),
            "value_loss": critic_loss.item(),
            "grad_norm": grad_norm,
            "entropy": entropy.item(),
            "alpha": self.alpha.item(),
            "value": current_q1.mean().item() # Log one of the Q-values as a proxy for value
        }

        return logs

    def _soft_update_target(self, target_net, source_net):
        """Helper function for soft target network updates"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

# --- Replay Buffer Definition ---
class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, mask=1.0):
        """Adds an experience to the buffer"""
        # Extract image data correctly from DictList or dict
        if hasattr(obs, 'image'):
            # If it's a DictList, extract the image properly
            obs_image = obs.image
            # If obs.image is a tensor, convert to numpy for storage
            if isinstance(obs_image, torch.Tensor):
                obs_image = obs_image.cpu().numpy()
            obs_dict = {'image': obs_image}
        else:
            # Handle regular dictionaries
            obs_dict = {'image': obs['image']}
        
        # Same for next_obs
        if hasattr(next_obs, 'image'):
            next_obs_image = next_obs.image
            if isinstance(next_obs_image, torch.Tensor):
                next_obs_image = next_obs_image.cpu().numpy()
            next_obs_dict = {'image': next_obs_image}
        else:
            next_obs_dict = {'image': next_obs['image']}
        
        self.buffer.append((obs_dict, action, reward, next_obs_dict, done, mask))

    def sample(self, batch_size, device):
        """Samples a batch of experiences from the buffer"""
        batch_indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in batch_indices]
        
        # Unzip the batch into separate lists
        obs, actions, rewards, next_obs, dones, masks = zip(*batch)

        # Convert to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
        masks = torch.tensor(masks, dtype=torch.float, device=device).unsqueeze(1)
        
        # Process observations in a more robust way
        processed_obs = {}
        processed_next_obs = {}
        
        # Get the keys from the first observation
        keys = obs[0].keys()
        
        for key in keys:
            try:
                # Try to stack tensors or convert to tensors
                if isinstance(obs[0][key], np.ndarray):
                    processed_obs[key] = torch.tensor(np.array([o[key] for o in obs]), device=device)
                    processed_next_obs[key] = torch.tensor(np.array([no[key] for no in next_obs]), device=device)
                elif isinstance(obs[0][key], torch.Tensor):
                    processed_obs[key] = torch.stack([o[key].to(device) for o in obs])
                    processed_next_obs[key] = torch.stack([no[key].to(device) for no in next_obs])
                else:
                    processed_obs[key] = torch.tensor([o[key] for o in obs], device=device)
                    processed_next_obs[key] = torch.tensor([no[key] for no in next_obs], device=device)
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                # Skip this key if it can't be processed
                continue
                
        # Convert to DictList for compatibility with the actor model
        obs = DictList(processed_obs)
        next_obs = DictList(processed_next_obs)
        
        return obs, actions, rewards, next_obs, dones, masks
        
    def __len__(self):
        """Returns the current size of the buffer"""
        return len(self.buffer)