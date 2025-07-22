import torch
import numpy as np
import os
import torch.multiprocessing as mp

from torch_ac.algos.base import BaseAlgo
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList

# Ensure each worker uses only one CPU thread to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"

# Handle multiprocessing start method safely
try:
    # Use force=False to avoid errors if already set
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    # Method already set, ignore the error
    pass

class SharedAdam(torch.optim.Adam):
    """
    Adam optimizer with shared states for multiprocessing.
    
    This version of Adam allows parameters to be shared across processes,
    which is necessary for the asynchronous updates in A3C.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # Share optimizer state across processes
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
    """
    Asynchronous Advantage Actor-Critic (A3C) algorithm implementation.
    
    A3C trains a global network using multiple worker processes that each maintain
    their own local copy of the model and environment. Workers periodically update
    the global model with their computed gradients.
    """

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None,
                 num_processes=4, update_interval=5, max_frames=1e7):
        """
        Initialize the A3C algorithm.
        
        """
        # Set a reasonable default for frames per process
        num_frames_per_proc = num_frames_per_proc or 8
        
        # Call the base class constructor with common parameters
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, 
                        entropy_coef, value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)
        
        # Store the environments for later use by workers
        self.envs = envs 

        # A3C specific parameters
        self.num_processes = num_processes
        self.update_interval = update_interval
        
        # Make the global network's parameters shared between processes
        self.acmodel.share_memory()
        
        # Use SharedAdam optimizer for multiprocessing
        self.optimizer = SharedAdam(self.acmodel.parameters(), lr=lr, betas=(0.9, 0.99))
        
        # Create shared variables for tracking progress
        self.global_ep = mp.Value('i', 0)             # Global episode counter
        self.global_ep_r = mp.Value('d', 0.)          # Global episode reward
        self.global_frames = mp.Value('i', 0)         # Total frames processed
        self.res_queue = mp.Queue()                   # Queue for results from workers
        
        # Container for worker processes
        self.workers = []
        
        # Training status tracking
        self.results = []
        self.training_complete = mp.Value('b', False)  # Flag to signal training completion
        self.max_frames = max_frames                   # Max frames to train for

    def update_parameters(self, exps):
        """
        Update model parameters based on worker processes.
        
        In A3C, updates happen asynchronously through the workers.
        This method starts workers if they don't exist yet and
        collects results from workers through the result queue.
        """
        # Initialize workers if first run
        if not self.workers:
            self._initialize_workers()
            # Start worker processes
            for worker in self.workers:
                worker.start()
        
        # Check if training is complete
        if self.training_complete.value:
            return {"entropy": 0, "value": 0, "policy_loss": 0, "value_loss": 0, "grad_norm": 0}
        
        # Default log values
        logs = {"entropy": 0, "value": 0, "policy_loss": 0, "value_loss": 0, "grad_norm": 0}
        
        # Check if all workers have finished
        active_workers = sum(worker.is_alive() for worker in self.workers)
        if active_workers == 0:
            self.training_complete.value = True
            
        # Process results from queue
        while not self.res_queue.empty():
            result = self.res_queue.get()
            if result is None:  # Worker has finished
                continue
            
            # Collect logs from workers
            if isinstance(result, dict):
                # Update logs with worker results
                for key in logs.keys():
                    if key in result:
                        logs[key] = result[key]
            else:
                # Store reward information
                self.results.append(result)
                
        return logs

    def collect_experiences(self):
        """
        Create a placeholder for collecting experiences.
        
        In A3C, experiences are collected by worker processes,
        so this method mainly returns empty containers to satisfy
        the BaseAlgo interface.
        """
        # Create empty experiences structure
        exps = DictList()
        exps.obs = []
        exps.action = torch.tensor([], device=self.device)
        exps.value = torch.tensor([], device=self.device)
        exps.reward = torch.tensor([], device=self.device)
        exps.advantage = torch.tensor([], device=self.device)
        exps.returnn = torch.tensor([], device=self.device)
        exps.log_prob = torch.tensor([], device=self.device)
        
        # Add memory fields for recurrent models
        if self.acmodel.recurrent:
            exps.memory = torch.tensor([], device=self.device)
            exps.mask = torch.tensor([], device=self.device)
        
        # Get global episode information
        with self.global_ep.get_lock():
            global_ep_value = self.global_ep.value
            
        with self.global_ep_r.get_lock():
            global_ep_r_value = self.global_ep_r.value
        
        # Create logs structure
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
        """
        Initialize worker processes.
        
        Creates worker processes, each with its own environment
        and local copy of the model.
        """
        for i in range(self.num_processes):
            # Get or create environment for this worker
            if i < len(self.envs):
                worker_env = self.envs[i]
            else:
                # Use the first environment as a template if needed
                worker_env = self.envs[0]
            
            # Create worker process
            worker = Worker(
                rank=i, 
                global_net=self.acmodel, 
                optimizer=self.optimizer,
                global_ep=self.global_ep,
                global_ep_r=self.global_ep_r,
                res_queue=self.res_queue,
                env=worker_env, 
                device=self.device,
                preprocess_obss=None,  # Created inside worker to avoid pickling issues
                observation_space=worker_env.observation_space, 
                discount=self.discount,
                update_interval=self.update_interval,
                entropy_coef=self.entropy_coef,
                value_loss_coef=self.value_loss_coef,
                global_frames=self.global_frames,
                max_frames=self.max_frames,
                training_complete=self.training_complete,
                max_grad_norm=self.max_grad_norm
            )
            self.workers.append(worker)


class Worker(mp.Process):
    """
    Worker process for A3C algorithm.
    
    Each worker:
    1. Maintains its own environment and local copy of the model
    2. Collects experiences by interacting with its environment
    3. Computes gradients and updates the global model
    4. Synchronizes its local model with the updated global model
    """
    
    def __init__(self, rank, global_net, optimizer, global_ep, global_ep_r, res_queue, 
                env, device, preprocess_obss, observation_space, discount, update_interval, 
                entropy_coef, value_loss_coef, global_frames, max_frames, training_complete, 
                max_grad_norm=0.5):
        """
        Initialize a worker process.
        """
        super(Worker, self).__init__()
        
        # Worker identification
        self.name = f'w{rank:02d}'
        self.rank = rank
        
        # Shared variables for coordination
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue
        self.global_frames = global_frames
        self.max_frames = max_frames
        self.training_complete = training_complete
        
        # Network and optimizer
        self.global_net = global_net
        self.optimizer = optimizer
        
        # Environment and processing
        self.env = env
        self.device = device
        self.observation_space = observation_space
        self.preprocess_obss = None  # Will be created in run()
        
        # Learning parameters
        self.discount = discount
        self.update_interval = update_interval
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Local network will be created in run() to avoid pickling issues

    def run(self):
        """
        Main worker process loop.
        
        This method:
        1. Sets up the local model
        2. Interacts with the environment
        3. Computes gradients and updates the global model
        4. Reports results back to the main process
        """
        
        # Import needed modules inside the worker to avoid pickling issues
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import utils

        # Create observation preprocessor
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.observation_space)

        # Check if model uses memory and create local copy of the model
        use_memory = getattr(self.global_net, 'use_memory', False)
        memory_size = getattr(self.global_net, 'memory_size', 0) if use_memory else 0
        use_text = getattr(self.global_net, 'use_text', False)
        
        # Create local network with same architecture as global network
        self.local_net = type(self.global_net)(
            obs_space, self.env.action_space, 
            use_memory, use_text
        ).to(self.device)
        
        # Initialize local network with global network's weights
        self.local_net.load_state_dict(self.global_net.state_dict())

        # Main training loop
        total_step = 1
        while True:
            # Start a new episode
            obs, _ = self.env.reset()
            buffer_obs, buffer_actions, buffer_rewards = [], [], []
            buffer_values, buffer_log_probs = [], []
            ep_r = 0.0
            done = False
            
            # Initialize memory for recurrent models
            if use_memory:
                memory = torch.zeros(1, memory_size, device=self.device)

            # Episode loop
            while not done:
                # Process observation
                preprocessed_obs = self.preprocess_obss([obs], device=self.device)
                
                # Get action from local network
                with torch.no_grad():
                    if use_memory:
                        dist, value, memory = self.local_net(preprocessed_obs, memory)
                    else:
                        outputs = self.local_net(preprocessed_obs, None)
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                            dist, value, _ = outputs
                        else:
                            dist, value = outputs
                
                # Sample action from distribution
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
                
                # Update episode reward and current observation
                ep_r += reward
                obs = next_obs
                
                # Update global network if enough steps collected or episode finished
                if total_step % self.update_interval == 0 or done:
                    # Compute returns and advantages
                    if done:
                        R = 0  # Terminal state has value 0
                    else:
                        # Estimate value of next state
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
                    
                    # Calculate n-step returns
                    returns = []
                    for r in buffer_rewards[::-1]:  # Reversed rewards
                        R = r + self.discount * R
                        returns.insert(0, R)
                    
                    # Convert experience to tensors
                    batch_obs = self.preprocess_obss(buffer_obs, device=self.device)
                    batch_actions = torch.cat(buffer_actions).to(self.device)
                    returns = torch.tensor(returns, dtype=torch.float).to(self.device)
                    log_probs = torch.cat(buffer_log_probs).to(self.device)
                    
                    # Get current predictions from local network
                    if use_memory:
                        # Start with fresh memory for batch processing
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
                    
                    # Clip gradients by norm
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.max_grad_norm)
                    
                    # Properly accumulate gradients
                    for local_param, global_param in zip(self.local_net.parameters(), self.global_net.parameters()):
                        if global_param.grad is None:
                            global_param._grad = local_param.grad.clone()
                        else:
                            global_param._grad += local_param.grad
                    
                    # Perform update step
                    self.optimizer.step()
                    
                    # Sync local network with updated global network
                    self.local_net.load_state_dict(self.global_net.state_dict())
                    
                    # Log episode results
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
                            "grad_norm": 0.0  # Could calculate if needed
                        }
                        self.res_queue.put(log_data)
                        self.res_queue.put(self.g_ep_r.value)
                        
                        print(f"{self.name} | Episode: {self.g_ep.value} | Reward: {ep_r:.3f}")
                        break
                    
                    # Reset buffers
                    buffer_obs, buffer_actions, buffer_rewards = [], [], []
                    buffer_values, buffer_log_probs = [], []
                
                # Increment step counter
                total_step += 1
            
            # Update global frame count
            with self.global_frames.get_lock():
                self.global_frames.value += len(buffer_rewards)  # Count actual steps
                # Check if training should end
                if self.global_frames.value >= self.max_frames:
                    with self.training_complete.get_lock():
                        self.training_complete.value = True
            
            # Check if training is complete
            with self.training_complete.get_lock():
                if self.training_complete.value:
                    self.res_queue.put(None)  # Signal completion
                    break