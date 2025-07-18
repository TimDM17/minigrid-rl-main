import torch
import numpy as np
import os
import torch.multiprocessing as mp

from torch_ac.algos.base import BaseAlgo
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList

# Set number of threads for each worker process
os.environ["OMP_NUM_THREADS"] = "1"


class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared states for multiprocessing"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                # Share memory for multiprocessing
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class A3CAlgo(BaseAlgo):
    """The Asynchronous Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None,
                 num_processes=4, update_interval=5, max_frames=1e7):
        """
        Parameters:
        ----------
        update_interval : int
            The number of steps before updating the global network
        num_processes : int
            Number of worker processes to use
        """
        num_frames_per_proc = num_frames_per_proc or 8
        
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, 
                        entropy_coef, value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)
        
        self.envs = envs # ADD THIS LINE

        # A3C specific parameters
        self.num_processes = num_processes
        self.update_interval = update_interval
        
        # Share memory of the global network
        self.acmodel.share_memory()
        
        # Use SharedAdam optimizer
        self.optimizer = SharedAdam(self.acmodel.parameters(), lr=lr, betas=(0.9, 0.99))
        
        # Create shared counters for tracking
        self.global_ep = mp.Value('i', 0)
        self.global_ep_r = mp.Value('d', 0.)
        self.global_frames = mp.Value('i', 0)  
        self.res_queue = mp.Queue()
        
        # Create worker processes
        self.workers = []
        
        # Results and tracking
        self.results = []
        self.training_complete = mp.Value('b', False)
        self.max_frames = max_frames

    def update_parameters(self, exps):
        """
        In A3C, the update is actually handled by worker processes.
        This method mainly coordinates the worker processes and returns logs.
        
        The actual parameter updates happen asynchronously.
        """
        # Initialize workers if first run
        if not self.workers:
            self._initialize_workers()
            # Start worker processes
            for worker in self.workers:
                worker.start()
        
        # Check if training is complete
        if self.training_complete:
            return {"entropy": 0, "value": 0, "policy_loss": 0, "value_loss": 0, "grad_norm": 0}
        
        # Collect results from result queue if available
        logs = {"entropy": 0, "value": 0, "policy_loss": 0, "value_loss": 0, "grad_norm": 0}
        
        # Check if any workers have finished
        active_workers = sum(worker.is_alive() for worker in self.workers)
        if active_workers == 0:
            self.training_complete = True
            
        # Process results from queue
        while not self.res_queue.empty():
            result = self.res_queue.get()
            if result is None:  # Worker has finished
                continue
            
            # Collect all worker results
            if isinstance(result, dict):
                # Merge logs
                for key in logs.keys():
                    if key in result:
                        logs[key] = result[key]
            else:
                # Store reward information
                self.results.append(result)
                
        return logs

    def collect_experiences(self):
        """
        In A3C, experiences are collected by worker processes.
        This is mainly a placeholder to satisfy BaseAlgo's interface.
        """
        # Create dummy experiences to satisfy BaseAlgo's interface
        exps = DictList()
        exps.obs = []
        exps.action = torch.tensor([], device=self.device)
        exps.value = torch.tensor([], device=self.device)
        exps.reward = torch.tensor([], device=self.device)
        exps.advantage = torch.tensor([], device=self.device)
        exps.returnn = torch.tensor([], device=self.device)
        exps.log_prob = torch.tensor([], device=self.device)
        
        if self.acmodel.recurrent:
            exps.memory = torch.tensor([], device=self.device)
            exps.mask = torch.tensor([], device=self.device)
        
        # Get global episode information
        with self.global_ep.get_lock():
            global_ep_value = self.global_ep.value
            
        with self.global_ep_r.get_lock():
            global_ep_r_value = self.global_ep_r.value
        
        # Create return logs
        logs = {
            "num_frames": self.update_interval * self.num_processes,
            "return_per_episode": [global_ep_r_value],
            "reshaped_return_per_episode": [global_ep_r_value],
            "num_frames_per_episode": [0],
            "entropy": 0,
            "value": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "grad_norm": 0
        }
        
        return exps, logs
    
    def _initialize_workers(self):
        """Initialize worker processes"""
        for i in range(self.num_processes):
            # Get environment for this worker
            if i < len(self.envs):
                worker_env = self.envs[i]
            else:
                # Create a new environment if needed
                worker_env = self.envs[0] # Use the first env as a template
            
            # Instead of passing the preprocess_obs function directly,
            # we'll pass the observation space and have the worker recreate ist
            worker = Worker(
                i, 
                self.acmodel, 
                self.optimizer,
                self.global_ep,
                self.global_ep_r,
                self.res_queue,
                worker_env, 
                self.device,
                None, # Don't pass preprocess_obss directly
                worker_env.observation_space, # Pass observation space instead
                self.discount,
                self.update_interval,
                self.entropy_coef,
                self.value_loss_coef,
                self.global_frames,
                self.max_frames,
                self.training_complete,
                self.max_grad_norm
            )
            self.workers.append(worker)

class Worker(mp.Process):
    """Worker process for A3C algorithm"""
    
    def __init__(self, rank, global_net, optimizer, global_ep, global_ep_r, res_queue, 
                env, device, preprocess_obss, observation_space, gamma, update_interval, entropy_coef, value_loss_coef,
                global_frames, max_frames, training_complete, max_grad_norm=0.5):
        super(Worker, self).__init__()
        self.name = f'w{rank:02d}'
        self.rank = rank
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.env = env
        self.device = device
        self.gamma = gamma
        self.update_interval = update_interval
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.observation_space = observation_space

        # The worker will create its own preprocessor in the run() method 
        # to avoid pickling issues
        self.preprocess_obss = None
        

        # Import needed module inside the worker
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import utils

        # Get the preprocessed observation space like in the main script
        obs_space, _ = utils.get_obss_preprocessor(env.observation_space)
        action_space = env.action_space
        use_memory = getattr(global_net, 'use_memory', False)
        use_text = getattr(global_net, 'use_text', False)

         # Create local network
        self.local_net = type(global_net)(
             obs_space, action_space, 
             use_memory, use_text
         ).to(device)
        self.local_net.load_state_dict(global_net.state_dict())

    
        # Additional attributes for tracking
        self.global_frames = global_frames
        self.max_frames = max_frames
        self.training_complete = training_complete

    def run(self):
        """Run worker process"""
        # Import needed modules inside the worker
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import utils

        # Create our own preprocessor
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.observation_space)

        # Check if model uses memory
        use_memory = getattr(self.local_net, 'use_memory', False)
        memory_size = getattr(self.local_net, 'memory_size', 0) if use_memory else 0

        # Create local network
        use_text = getattr(self.global_net, 'use_text', False)
        self.local_net = type(self.global_net)(
            obs_space, self.env.action_space, 
            use_memory, use_text
        ).to(self.device)
        self.local_net.load_state_dict(self.global_net.state_dict())

        total_step = 1
        while True:
            obs, _ = self.env.reset()
            buffer_obs, buffer_actions, buffer_rewards = [], [], []
            buffer_values, buffer_log_probs = [], []
            ep_r = 0.0
            done = False
            
            # Initialize memory for recurrent models
            if use_memory:
                memory = torch.zeros(1, memory_size, device=self.device)

            while not done:
                # Process observation
                preprocessed_obs = self.preprocess_obss([obs], device=self.device)
                
                # Get action from local network
                with torch.no_grad():
                    if use_memory:
                        dist, value, memory = self.local_net(preprocessed_obs, memory)
                    else:
                        # For non-recurrent models, the model might still return 3 values
                        outputs = self.local_net(preprocessed_obs, None)
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                            dist, value, _ = outputs  # Ignore the third return value if memory is returned
                        else:
                            dist, value = outputs
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Take action in environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                
                # Store experience
                buffer_obs.append(obs)
                buffer_actions.append(action)
                buffer_rewards.append(reward)
                buffer_values.append(value)
                buffer_log_probs.append(log_prob)
                
                ep_r += reward
                obs = next_obs
                
                # Update global network periodically
                if total_step % self.update_interval == 0 or done:
                    # Compute returns and advantages
                    if done:
                        R = 0
                    else:
                        preprocessed_next_obs = self.preprocess_obss([next_obs], device=self.device)
                        with torch.no_grad():
                            if use_memory:
                                _, R, memory = self.local_net(preprocessed_next_obs, memory)
                            else:
                                outputs = self.local_net(preprocessed_next_obs, None)
                                if isinstance(outputs, tuple) and len(outputs) == 3:
                                    _, R, _ = outputs
                                else:
                                    _, R = outputs
                            R = R.detach().item()
                    
                    # Calculate returns
                    returns = []
                    for r in buffer_rewards[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                    
                    # Convert to tensors
                    batch_obs = self.preprocess_obss(buffer_obs, device=self.device)
                    batch_actions = torch.cat(buffer_actions).to(self.device)
                    returns = torch.tensor(returns, dtype=torch.float).to(self.device)
                    log_probs = torch.cat(buffer_log_probs).to(self.device)  # ADD THIS LINE
                    
                    # Get current values and log probs
                    if use_memory:
                        # When processing the batch, start with fresh memory
                        batch_memory = torch.zeros(len(buffer_obs), memory_size, device=self.device)
                        dist, values, _ = self.local_net(batch_obs, batch_memory)
                    else:
                        outputs = self.local_net(batch_obs, None)
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                            dist, values, _ = outputs
                        else:
                            dist, values = outputs
                    
                    # Calculate advantages
                    advantages = returns - values.squeeze()
                    
                    # Calculate losses
                    entropy = dist.entropy().mean()
                    policy_loss = -(log_probs * advantages.detach()).mean()
                    value_loss = advantages.pow(2).mean()
                    
                    # Total loss
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    
                    # Update global network
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Ensure gradient norm is clipped
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.max_grad_norm)
                    
                    # Push gradients to global network
                    for local_param, global_param in zip(self.local_net.parameters(), self.global_net.parameters()):
                        if global_param.grad is None:
                            global_param._grad = local_param.grad
                    
                    self.optimizer.step()
                    
                    # Update local network with global network parameters
                    self.local_net.load_state_dict(self.global_net.state_dict())
                    
                    # Log results
                    if done:
                        # Update global counters
                        with self.g_ep.get_lock():
                            self.g_ep.value += 1
                        with self.g_ep_r.get_lock():
                            if self.g_ep_r.value == 0:
                                self.g_ep_r.value = ep_r
                            else:
                                self.g_ep_r.value = self.g_ep_r.value * 0.99 + ep_r * 0.01
                        
                        # Send logs to main process
                        log_data = {
                            "entropy": entropy.item(),
                            "value": values.mean().item(),
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                            "grad_norm": 0.0  # Could calculate this if needed
                        }
                        self.res_queue.put(log_data)
                        self.res_queue.put(self.g_ep_r.value)
                        
                        print(f"{self.name} | Episode: {self.g_ep.value} | Reward: {ep_r:.3f}")
                        break
                    
                    # Reset buffers
                    buffer_obs, buffer_actions, buffer_rewards = [], [], []
                    buffer_values, buffer_log_probs = [], []
                
                total_step += 1
            
            # Update global frame count
            with self.global_frames.get_lock():
                self.global_frames.value += 1
                if self.global_frames.value >= self.max_frames:
                    with self.training_complete.get_lock():
                        self.training_complete.value = True
            
            # Check if we should terminate
            with self.training_complete.get_lock():
                if self.training_complete.value:
                    self.res_queue.put(None)  # Signal completion
                    break