import argparse
import numpy as np
import pickle
import utils
from utils import device


# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--env", default='custom')
parser.add_argument("--model_0", default="ppo_reasoner_8*8_data_collect.")
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--reasoner", type=bool, default=True)
parser.add_argument("--adaptive", type=bool, default=True)
parser.add_argument("--two_doors", type=str, default='two_doors')
parser.add_argument("--size", type=int, default=8)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--save_path", type=str, default="symbolic_dataset2.pkl")
parser.add_argument("--max_seq_len", type=int, default=100)
parser.add_argument("--mode", type=str, choices=["per_step", "per_episode"], default="per_episode",
                    help="Choose whether to save samples at each step or per episode.")
args = parser.parse_args()

# === Set seed ===
utils.seed(args.seed)

# === Load environment ===
env = utils.make_env(args.env, 'goal', args.two_doors, args.reasoner, args.adaptive,
                     args.size, args.seed, render_mode="human")

# === Load pretrained PPO agent ===
model_dir_0 = utils.get_model_dir(args.model_0)
agent_0 = utils.Agent(env.observation_space, env.action_space, model_dir_0,
                      argmax=False, use_memory=False, use_text=False)

# === Start data collection ===
collected_data = []
print(f"🚀 Starting data collection in mode: {args.mode}")

for episode in range(args.episodes):
    obs, _ = env.reset()
    image_seq = []
    action_seq = []
    symbolic_state_seq = []  # <-- Added to store symbolic states per step
    num_steps = 0

    while True:
        image = obs["full_image"] # shape: (256, 256, 3)
       # import matplotlib.pyplot as plt
       # plt.imshow(image.float() / 255.0)
        image_seq.append(image)
       # print(image)
        action = agent_0.get_action(obs)
        #print(action)
        action_seq.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env_symbolic_state = obs["env_symbolic_state"]
        symbolic_state_seq.append(env_symbolic_state.copy())  # Store symbolic state per step

        # === Save per-step sample ===
        if args.mode == "per_step":
            collected_data.append({
                "image_seq": image_seq.copy(),
                "action_seq": action_seq.copy(),
                "env_symbolic_state": symbolic_state_seq.copy()
            })

        agent_0.analyze_feedback(reward, done)
        num_steps += 1

        # === Save per-episode sample ===
        if args.mode == "per_episode" and (done or num_steps >= args.max_seq_len):
            collected_data.append({
                "image_seq": image_seq.copy(),
                "action_seq": action_seq.copy(),
                "env_symbolic_state": symbolic_state_seq.copy()  # <-- store entire symbolic state sequence
            })
            break

        if done or num_steps >= args.max_seq_len:
            break

    print(f"✅ Episode {episode + 1}/{args.episodes} collected.")

# === Save dataset ===
with open(args.save_path, "wb") as f:
    pickle.dump(collected_data, f)

print(f"\n📦 Collected {len(collected_data)} samples.")
print(f"📁 Saved to: {args.save_path}")

