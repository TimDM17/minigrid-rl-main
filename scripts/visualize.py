import argparse
import numpy

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='custom',#"MiniGrid-DoorKey-8x8-v0",#'costum',#"MiniGrid-DoorKey-6x6",
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model_0", default="ppo_8*8_adaptive_reasoner.",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_1", default="ppo_8*8_adaptive_reasoner.",#"MiniGrid-6x6-getkey-costum",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_2", default="ppo_8*8_adaptive_reasoner.",#"MiniGrid-6x6-gotodoor-costum",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_3", default="ppo_8*8_adaptive_reasoner.",#"MiniGrid-6x6-togoal-costum",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_4", default="ppo_8*8_adaptive_reasoner.",#"DoorKey11",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=4,
                    help="random seed (default: 0)")
parser.add_argument("--reasoner", type=bool, default=True,
                    help="random seed (default: 1)")
parser.add_argument("--adaptive", type=bool, default=False,
                    help="random seed (default: 1)")
parser.add_argument("--two_doors", type=str, default='one_door',
                    help="options can be one_door, two_doors, three_doors")
parser.add_argument("--size", type=int, default=6,
                    help="number of processes (default: 16)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default='test',
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=9,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, 'goal',args.two_doors, args.reasoner, args.adaptive,  args.size, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
model_dir_0 = utils.get_model_dir(args.model_0)
agent_0 = utils.Agent(env.observation_space, env.action_space, model_dir_0,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)

model_dir_1 = utils.get_model_dir(args.model_1)
agent_1 = utils.Agent(env.observation_space, env.action_space, model_dir_1,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)

model_dir_2 = utils.get_model_dir(args.model_2)
agent_2 = utils.Agent(env.observation_space, env.action_space, model_dir_2,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)

model_dir_3 = utils.get_model_dir(args.model_3)
agent_3 = utils.Agent(env.observation_space, env.action_space, model_dir_3,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)


model_dir_4 = utils.get_model_dir(args.model_4)
agent_4 = utils.Agent(env.observation_space, env.action_space, model_dir_4,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

for episode in range(args.episodes):
    obs, _ = env.reset()
    num = 0
    hierarchical = True
    while True:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 1, 0))
        if not hierarchical:

            action = agent_0.get_action(obs)
            #import matplotlib.pyplot as plt
            #plt.imshow(obs['full_image'])
            print('actions is: ', action)
            obs, reward, terminated, truncated, _ = env.step(action)
            print('env_symbolic-state is: ', obs['env_symbolic_state'])
            done = terminated | truncated
            print('done is: ', done)
            agent_0.analyze_feedback(reward, done)

        if hierarchical:
            if obs['env_symbolic_state'][2] == 0:
                action = agent_1.get_action(obs)
                obs, reward, terminated, truncated,  _ = env.step(action)
                done = terminated | truncated
                agent_1.analyze_feedback(reward, done)
                print('env_symbolic-state is: ', obs['env_symbolic_state'])
            if obs['env_symbolic_state'][2] == 1 and obs['env_symbolic_state'][4] == 0:
                action = agent_2.get_action(obs)
                obs, reward, terminated, truncated,  _ = env.step(action)
                done = terminated | truncated
                agent_2.analyze_feedback(reward, done)

                #else:
            if obs['env_symbolic_state'][4] == 1:
                action = agent_3.get_action(obs)
                obs, reward, terminated, truncated,  _ = env.step(action)
                done = terminated | truncated
                agent_3.analyze_feedback(reward, done)

            if done:
                break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
