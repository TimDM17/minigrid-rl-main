import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import copy
import sys
import os

from torch_ac.algos.base import BaseAlgo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


class QNetwork(nn.Module):
    """Separate Q-Network for SAC"""
    
    def __init__(self, acmodel):
        super().__init__()
        # Extract the image processing components from the acmodel
        self.image_conv = copy.deepcopy(acmodel.image_conv)
        
        # Determine embedding size (depends on your model)
        self.embedding_size = getattr(acmodel, 'semi_memory_size', 
                                     getattr(acmodel, 'embedding_size', 64))
        
        # Q-value head
        self.q_head = nn.Linear(self.embedding_size, 1)
        
    def forward(self, obs, actions=None):
        # Handle different observation types
        if hasattr(obs, 'image'):  # For DictList or objects with .image attribute
            x = obs.image
        elif isinstance(obs, dict): # For standard dictionaries
            x = obs["image"]
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
        
        # Process image through CNN (similar to ACModel)
        x = x.transpose(1, 3).transpose(2, 3)
        embedding = self.image_conv(x)
        
        # Flatten if needed
        if embedding.dim() > 2:
            embedding = embedding.reshape(embedding.shape[0], -1)
        
        # Get Q-value
        q_value = self.q_head(embedding)
        
        return q_value


class SACAlgo(BaseAlgo):
    """The Soft Actor-Critic algorithm with twin critics."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, batch_size=256, tau=0.005, alpha=0.2, target_update_interval=1,
                 replay_size=10000, automatic_entropy_tuning=True, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        # SAC specific parameters
        self.batch_size = batch_size
        self.tau = tau  # For soft target updates
        self.target_update_interval = target_update_interval
        self.initial_alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)
        
        # Create twin critics (Q1 and Q2)
        try:
            # Try to create true separate networks if possible
            self.critic1 = QNetwork(self.acmodel).to(device)
            self.critic2 = QNetwork(self.acmodel).to(device)
            self.target_critic1 = QNetwork(self.acmodel).to(device)
            self.target_critic2 = QNetwork(self.acmodel).to(device)
            self.separate_critics = True
        except Exception as e:
            print(f"Warning: Could not create separate Q networks: {e}")
            print("Falling back to shared network approach")
            # Fall back to the original approach
            self.separate_critics = False
        
        # Initialize optimizers
        if self.separate_critics:
            self.policy_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=lr, eps=adam_eps)
            self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr, eps=adam_eps)
            self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr, eps=adam_eps)
        else:
            self.policy_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=lr, eps=adam_eps)
            self.value_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=lr, eps=adam_eps)
        
        # Initialize target model for policy
        # Create a properly preprocessed observation space
        obs_space, _ = utils.get_obss_preprocessor(self.env.observation_space)
    
        self.target_acmodel = type(self.acmodel)(
            obs_space,
            self.env.action_space,  
            getattr(self.acmodel, 'use_memory', False), 
            getattr(self.acmodel, 'use_text', False)
        )
        self.target_acmodel.load_state_dict(self.acmodel.state_dict())
        self.target_acmodel.to(self.device)
        
        # Initialize target networks for critics if using separate networks
        if self.separate_critics:
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Setup automatic entropy tuning
        if self.automatic_entropy_tuning:
            # Set target entropy to -|A| (negative action dimension)
            self.target_entropy = -numpy.log(1.0 / self.env.action_space.n) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.initial_alpha).to(device)
        
        self.update_counter = 0

    def update_parameters(self, exps):
        # Add experiences to replay buffer
        for i in range(len(exps.obs)):
            next_idx = min(i + 1, len(exps.obs) - 1)
            next_obs = exps.obs[next_idx]
            done = 1.0 if i == len(exps.obs) - 1 else 0.0
            
            self.replay_buffer.add(
                exps.obs[i], 
                exps.action[i], 
                exps.reward[i], 
                next_obs, 
                done,
                exps.mask[i] if hasattr(exps, 'mask') else 1.0
            )
        
        # Skip update if replay buffer doesn't have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {
                "entropy": 0,
                "value": 0,
                "policy_loss": 0,
                "value_loss": 0,
                "grad_norm": 0,
                "alpha": self.alpha.item() if hasattr(self.alpha, "item") else self.alpha
            }
        
        # Initialize log values
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_alphas = []
        log_grad_norms = []

        # Sample batch from replay buffer
        obs, actions, rewards, next_obs, dones, masks = self.replay_buffer.sample(self.batch_size, self.device)

        # Process observations if needed
        if self.preprocess_obss:
            # Check if observations are already preprocessed into a dictionary
            if isinstance(obs, dict) and "image" in obs:
                # Already in the correct format, no need to preprocess again
                processed_obs = obs
                processed_next_obs = next_obs
            else:
                # Need preprocessing
                processed_obs = self.preprocess_obss(obs, device=self.device)
                processed_next_obs = self.preprocess_obss(next_obs, device=self.device)
        else:
            processed_obs = obs
            processed_next_obs = next_obs
            
        # Current alpha value
        alpha_value = self.alpha.item() if hasattr(self.alpha, "item") else self.alpha
        
        # Update critic (value function)
        if self.separate_critics:
            value_loss = self._update_twin_critics(processed_obs, actions, rewards, processed_next_obs, dones, masks, alpha_value)
        else:
            value_loss = self._update_critic(processed_obs, actions, rewards, processed_next_obs, dones, masks, alpha_value)
        log_value_losses.append(value_loss.item())
        
        # Update actor (policy)
        policy_loss, entropy = self._update_actor(processed_obs, alpha_value)
        log_policy_losses.append(policy_loss.item())
        log_entropies.append(entropy.item())
        
        # Update temperature parameter if using automatic tuning
        if self.automatic_entropy_tuning:
            alpha_loss = self._update_alpha(entropy)
            alpha_value = self.alpha.item()
            log_alphas.append(alpha_value)
        
        # Update target networks with soft update
        self.update_counter += 1
        if self.update_counter % self.target_update_interval == 0:
            self._soft_update_target()
        
        # Calculate gradient norm for the policy
        policy_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
        log_grad_norms.append(policy_grad_norm)
        
        # Log values
        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values) if log_values else 0,
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "alpha": alpha_value
        }
        
        return logs
    
    def _update_critic(self, obs, actions, rewards, next_obs, dones, masks, alpha):
        """Update critic (value function) parameters."""
        with torch.no_grad():
            # Convert dictionary observations to DictList format
            from torch_ac.utils import DictList
            if isinstance(next_obs, dict):
                next_obs = DictList(next_obs)
            
            # Get next state action distribution
            if hasattr(self.acmodel, 'recurrent') and self.acmodel.recurrent:
                next_dist, next_value, _ = self.target_acmodel(next_obs, masks)
            else:
                next_dist, next_value = self.target_acmodel(next_obs, None)
            
            # Sample actions from distribution
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions)
            
            # Compute target Q value (soft Q-learning)
            target_q = rewards + (1.0 - dones) * self.discount * (next_value - alpha * next_log_probs)
        
        # Convert for current observations too
        if isinstance(obs, dict):
            obs = DictList(obs)
        
        # Get current Q estimate
        if hasattr(self.acmodel, 'recurrent') and self.acmodel.recurrent:
            _, current_value, _ = self.acmodel(obs, masks)
        else:
            _, current_value = self.acmodel(obs, None)
        
        # Compute critic loss
        value_loss = F.mse_loss(current_value, target_q.detach())
        
        # Optimize critic
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss
    
    def _update_twin_critics(self, obs, actions, rewards, next_obs, dones, masks, alpha):
        """Update twin critics (Q1 and Q2) parameters."""
        with torch.no_grad():
            # Only convert to DictList for the actor model, not the critics
            from torch_ac.utils import DictList
            
            # Create a copy for the actor but keep original for critics
            if isinstance(next_obs, dict):
                actor_next_obs = DictList(next_obs)
            else:
                actor_next_obs = next_obs
            
            # Use actor_next_obs with the actor model
            if hasattr(self.acmodel, 'recurrent') and self.acmodel.recurrent:
                next_dist, _, _ = self.target_acmodel(actor_next_obs, masks)
            else:
                next_dist, _ = self.target_acmodel(actor_next_obs, None)
                
            # Sample next actions and compute log probs
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions).unsqueeze(1)
            
            # Use original next_obs with critics
            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            
            # Use minimum Q-value for targets (clipped double Q-learning)
            min_target_q = torch.min(target_q1, target_q2)
            
            # Compute soft Q-learning targets
            soft_q_target = rewards + (1.0 - dones) * self.discount * (min_target_q - alpha * next_log_probs)
        
        # Get current Q estimates from both critics
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        
        # Compute MSE loss for both critics
        critic1_loss = F.mse_loss(current_q1, soft_q_target.detach())
        critic2_loss = F.mse_loss(current_q2, soft_q_target.detach())
        critic_loss = critic1_loss + critic2_loss
        
        # Optimize critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        return critic_loss
    
    def _update_actor(self, obs, alpha):
        """Update actor (policy) parameters with the improved SAC policy loss."""
        # Convert dictionary observations to DictList format
        from torch_ac.utils import DictList
        if isinstance(obs, dict):
            obs = DictList(obs)
    
        # Get action distribution
        if hasattr(self.acmodel, 'recurrent') and self.acmodel.recurrent:
            # Correctly create the memory tensor
            memory = torch.ones_like(obs.image[:, 0:1])
            dist, _, _ = self.acmodel(obs, memory)
        else:
            dist, _ = self.acmodel(obs, None)  # Pass None for memory
    
        # Sample actions and compute log probs
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        # Compute entropy
        entropy = dist.entropy().mean()
        
        # Compute Q-values for the sampled actions
        if self.separate_critics:
            # Use minimum of twin Q-values to reduce overestimation
            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions)
            q = torch.min(q1, q2)
        else:
            # Fall back to the value function if separate critics aren't available
            if self.acmodel.recurrent:
                _, q, _ = self.acmodel(obs, torch.ones_like(obs[:, 0:1]))
            else:
                _, q = self.acmodel(obs)
        
        # Compute actor loss - Standard SAC policy objective
        # We want to maximize E[min(Q) - α*log_π]
        policy_loss = (alpha * log_probs - q).mean()
        
        # Optimize actor
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        return policy_loss, entropy
    
    def _update_alpha(self, entropy):
        """Update temperature parameter alpha."""
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        return alpha_loss
    
    def _soft_update_target(self):
        """Soft update target network parameters."""
        # Update target policy network
        for target_param, param in zip(self.target_acmodel.parameters(), self.acmodel.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # Update target critic networks if using separate critics
        if self.separate_critics:
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, next_obs, done, mask=1.0):
        """Add experience to buffer."""
        self.buffer.append((obs, action, reward, next_obs, done, mask))
        
    def sample(self, batch_size, device):
        """Sample a batch of experiences."""
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        
        # Extract batch data
        batch = list(zip(*[self.buffer[i] for i in indices]))
        obs, actions, rewards, next_obs, dones, masks = batch
    
        # Special handling for DictList observations from torch_ac
        from torch_ac.utils import DictList
    
        # Process observations
        if isinstance(obs[0], dict) or isinstance(obs[0], DictList):
            # Convert all observations to regular dictionaries first if they're DictLists
            if isinstance(obs[0], DictList):
                obs = [dict(o) for o in obs]
                next_obs = [dict(no) for no in next_obs]
        
            # Create dictionary of batched tensors
            processed_obs = {}
            processed_next_obs = {}
        
            # Get all keys from the first observation
            keys = obs[0].keys()
        
            for key in keys:
                try:
                    # Get all values for this key and convert to tensor directly
                    if isinstance(obs[0][key], torch.Tensor):
                        # If already tensors, stack them
                        processed_obs[key] = torch.stack([o[key] for o in obs]).to(device)
                        processed_next_obs[key] = torch.stack([no[key] for no in next_obs]).to(device)
                    else:
                        # Otherwise convert to tensor
                        processed_obs[key] = torch.tensor([o[key] for o in obs], device=device)
                        processed_next_obs[key] = torch.tensor([no[key] for no in next_obs], device=device)
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    # Fall back to simple conversion if stacking fails
                    try:
                        processed_obs[key] = torch.tensor([o[key] for o in obs], device=device)
                        processed_next_obs[key] = torch.tensor([no[key] for no in next_obs], device=device)
                    except:
                        # Last resort: skip this key if it can't be processed
                        print(f"Skipping problematic key: {key}")
                        continue
                
            obs = processed_obs
            next_obs = processed_next_obs
        else:
            # Handle tensor observations
            if isinstance(obs[0], torch.Tensor):
                obs = torch.stack(obs).to(device)
                next_obs = torch.stack(next_obs).to(device)
            else:
                obs = torch.tensor(obs, device=device)
                next_obs = torch.tensor(next_obs, device=device)
    
        # Process the remaining data
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
        masks = torch.tensor(masks, dtype=torch.float, device=device).unsqueeze(1)
        
        return obs, actions, rewards, next_obs, dones, masks
    
    def __len__(self):
        return len(self.buffer)