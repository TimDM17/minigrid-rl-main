import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from utils import device
from model import ACModel
import random
import os
import torch

from Added_algorithms.a3c import A3CAlgo
from Added_algorithms.sac import SACAlgo  


# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", default="ppo",
                    help="algorithm to use: a2c | ppo | a3c | sac (REQUIRED)")
parser.add_argument("--env", default="custom",#"MiniGrid-DoorKey-8x8-v0",
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default='ppo_8x8_adaptive_reasoner_dense4',
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--reasoner", type=bool, default=True,
                    help="random seed (default: 1)")
parser.add_argument("--adaptive", type=bool, default=True,
                    help="random seed (default: 1)")
parser.add_argument("--two_doors", type=str, default='two_doors',
                    help="options can be one_door, two_doors")
parser.add_argument("--size", type=int, default=8,
                    help="size of environment (default: 16)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=3e7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO and SAC (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.001), sparse is 0.0001")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

# A3C specific parameters
parser.add_argument("--num-processes", type=int, default=4,
                    help="number of worker processes for A3C (default: 4)")
parser.add_argument("--update-interval", type=int, default=5,
                    help="number of steps before each A3C update (default: 5)")

# SAC specific parameters
parser.add_argument("--tau", type=float, default=0.005,
                    help="target network update rate for SAC (default: 0.005)")
parser.add_argument("--alpha", type=float, default=0.2,
                    help="temperature parameter for SAC (default: 0.2)")
parser.add_argument("--target-update-interval", type=int, default=1,
                    help="frequency of target network updates for SAC (default: 1)")
parser.add_argument("--replay-size", type=int, default=10000,
                    help="replay buffer size for SAC (default: 10000)")
parser.add_argument("--automatic-entropy-tuning", action="store_true", default=True,
                    help="automatically tune entropy coefficient in SAC")

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
      #  args.seed = random.randint(0, 10)
        envs.append(utils.make_env(args.env,'goal',args.two_doors, args.reasoner, args.adaptive, args.size, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "a3c":
        algo = A3CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss, None, 
                                args.num_processes, args.update_interval,
                                max_frames=args.frames)
    elif args.algo == "sac":
        algo = SACAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                       args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                       args.optim_eps, args.batch_size, args.tau, args.alpha, args.target_update_interval,
                       args.replay_size, args.automatic_entropy_tuning, preprocess_obss, None)  
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status and args.algo != "sac":
        algo.optimizer.load_state_dict(status["optimizer_state"])
    # For SAC, we have multiple optimizers
    elif args.algo == "sac":
        if "policy_optimizer_state" in status:
            algo.policy_optimizer.load_state_dict(status["policy_optimizer_state"])
        
        if algo.separate_critics:
            if "critic1_optimizer_state" in status:
                algo.critic1_optimizer.load_state_dict(status["critic1_optimizer_state"])
            if "critic2_optimizer_state" in status:
                algo.critic2_optimizer.load_state_dict(status["critic2_optimizer_state"])
                
            # Load critic network states if available
            if "critic1_state" in status:
                algo.critic1.load_state_dict(status["critic1_state"])
            if "critic2_state" in status:
                algo.critic2.load_state_dict(status["critic2_state"])
            if "target_critic1_state" in status:
                algo.target_critic1.load_state_dict(status["target_critic1_state"])
            if "target_critic2_state" in status:
                algo.target_critic2.load_state_dict(status["target_critic2_state"])
        else:
            if "value_optimizer_state" in status:
                algo.value_optimizer.load_state_dict(status["value_optimizer_state"])
        
        # Load alpha optimizer and state if automatic entropy tuning is enabled
        if args.automatic_entropy_tuning and "alpha_optimizer_state" in status:
            algo.alpha_optimizer.load_state_dict(status["alpha_optimizer_state"])
            if "log_alpha" in status:
                with torch.no_grad():
                    algo.log_alpha.copy_(torch.tensor(status["log_alpha"], device=algo.device))
                    algo.alpha = algo.log_alpha.exp()
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    # For A3C, we handle the training loop differently
    if args.algo == "a3c":
        # Initialize A3C - this starts the worker processes
        dummy_exps, _ = algo.collect_experiences()
        _ = algo.update_parameters(dummy_exps)
        
        # Main process just monitors and logs progress
        last_log_time = time.time()
        while num_frames < args.frames and not algo.training_complete.value:
            # Get the actual global frame count
            with algo.global_frames.get_lock():
                num_frames = algo.global_frames.value
            
            # Collect any available results
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            
            # Update counters
            update += 1
            
            # Print logs periodically
            current_time = time.time()
            if update % args.log_interval == 0 and current_time - last_log_time > 1.0:  # Avoid too frequent logging
                last_log_time = current_time
                fps = logs["num_frames"] / (current_time - start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                
                # Add alpha for SAC
                if args.algo == "sac":
                    header += ["alpha"]
                    data += [logs["alpha"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:usmM {:.2f} {:.2f} {:.2f} {:.2f} | F:usmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | grad {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)
            
            # Save status periodically
            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {
                    "num_frames": num_frames, 
                    "update": update,
                    "model_state": acmodel.state_dict(),  # Get global model state
                    "optimizer_state": algo.optimizer.state_dict()
                }
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
            
            # Save model checkpoints
            if num_frames // 50000 > (status.get("num_frames", 0) // 50000):
                model_path = os.path.join(model_dir, f"model_{num_frames}_frames.pt")
                torch.save({
                    "model_state": acmodel.state_dict(),
                    "optimizer_state": algo.optimizer.state_dict(),
                    "num_frames": num_frames,
                    "update": update
                }, model_path)
                txt_logger.info(f"Model checkpoint saved at {model_path}")
            
            # Sleep a bit to avoid busy waiting
            time.sleep(0.1)
        
        # Clean up worker processes when training is complete
        txt_logger.info("Training complete, shutting down worker processes")
        for _ in range(args.num_processes):
            algo.res_queue.put(None)
        
        for worker in algo.workers:
            worker.join()
        txt_logger.info("All worker processes terminated")

    else:
        while num_frames < args.frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                
                # Add alpha for SAC
                if args.algo == "sac":
                    header += ["alpha"]
                    data += [logs["alpha"]]

                if args.algo == "sac":
                    txt_logger.info(
                        "Update {:4d} | Frames {:8d} | FPS {:6.0f} | Duration {:5.0f}s | "
                        "Return: mean={:5.2f} std={:5.2f} min={:5.2f} max={:5.2f} | "
                        "Frames/ep: {:4.1f} {:4.1f} {:3.0f} {:3.0f} | "
                        "Entropy: {:5.3f} | Value: {:5.3f} | PolicyLoss: {:6.3f} | ValueLoss: {:6.3f} | GradNorm: {:5.3f} | Alpha: {:5.3f}"
                        .format(*data))
                else:
                    txt_logger.info(
                        "Update {:4d} | Frames {:8d} | FPS {:6.0f} | Duration {:5.0f}s | "
                        "Return: mean={:5.2f} std={:5.2f} min={:5.2f} max={:5.2f} | "
                        "Frames/ep: {:4.1f} {:4.1f} {:3.0f} {:3.0f} | "
                        "Entropy: {:5.3f} | Value: {:5.3f} | PolicyLoss: {:6.3f} | ValueLoss: {:6.3f} | GradNorm: {:5.3f}"
                        .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status

            if args.save_interval > 0 and update % args.save_interval == 0:
                if args.algo == "sac":
                    status = {
                        "num_frames": num_frames, 
                        "update": update,
                        "model_state": acmodel.state_dict(),
                        "target_model_state": algo.target_acmodel.state_dict(),
                        "policy_optimizer_state": algo.policy_optimizer.state_dict()
                    }
                    
                    # Handle different critic architectures
                    if hasattr(algo, 'separate_critics') and algo.separate_critics:
                        # Twin critics case
                        status["critic1_state"] = algo.critic1.state_dict()
                        status["critic2_state"] = algo.critic2.state_dict()
                        status["target_critic1_state"] = algo.target_critic1.state_dict()
                        status["target_critic2_state"] = algo.target_critic2.state_dict()
                        status["critic1_optimizer_state"] = algo.critic1_optimizer.state_dict()
                        status["critic2_optimizer_state"] = algo.critic2_optimizer.state_dict()
                    else:
                        # Single shared critic case
                        status["value_optimizer_state"] = algo.value_optimizer.state_dict()
                    
                    # Save alpha parameters for automatic entropy tuning
                    if args.automatic_entropy_tuning:
                        status["alpha_optimizer_state"] = algo.alpha_optimizer.state_dict()
                        status["log_alpha"] = algo.log_alpha.detach().cpu().numpy()
                else:
                    status = {
                        "num_frames": num_frames, 
                        "update": update,
                        "model_state": acmodel.state_dict(), 
                        "optimizer_state": algo.optimizer.state_dict()
                    }
                
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")

            # >>>>>> NEW: Save model every 20,000 frames <<<<<<
            if num_frames // 50000 > (status.get("num_frames", 0) // 50000):
                model_path = os.path.join(model_dir, f"model_{num_frames}_frames.pt")
                if args.algo == "sac":
                    checkpoint = {
                        "model_state": acmodel.state_dict(),
                        "target_model_state": algo.target_acmodel.state_dict(),
                        "policy_optimizer_state": algo.policy_optimizer.state_dict(),
                        "num_frames": num_frames,
                        "update": update
                    }
                    # Handle different critic architectures
                    if hasattr(algo, 'separate_critics') and algo.separate_critics:
                        # Twin critics case
                        checkpoint["critic1_state"] = algo.critic1.state_dict()
                        checkpoint["critic2_state"] = algo.critic2.state_dict()
                        checkpoint["target_critic1_state"] = algo.target_critic1.state_dict()
                        checkpoint["target_critic2_state"] = algo.target_critic2.state_dict()
                        checkpoint["critic1_optimizer_state"] = algo.critic1_optimizer.state_dict()
                        checkpoint["critic2_optimizer_state"] = algo.critic2_optimizer.state_dict()
                    else:
                        # Single shared critic case
                        checkpoint["value_optimizer_state"] = algo.value_optimizer.state_dict()
                    
                    # Save alpha parameters for automatic entropy tuning
                    if args.automatic_entropy_tuning:
                        checkpoint["alpha_optimizer_state"] = algo.alpha_optimizer.state_dict()
                        checkpoint["log_alpha"] = algo.log_alpha.detach().cpu().numpy()
                    
                    torch.save(checkpoint, model_path)
                else:
                    torch.save({
                        "model_state": acmodel.state_dict(),
                        "optimizer_state": algo.optimizer.state_dict(),
                        "num_frames": num_frames,
                        "update": update
                    }, model_path)
                txt_logger.info(f"Model checkpoint saved at {model_path}")


