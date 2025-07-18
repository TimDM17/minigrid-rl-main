import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device
from utils.Reasoner.src.predict_kandinsky import predict_reward,predict_step_reward, initialize_reasoner, initialize_percept_reasoner

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='custom',
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model",  default="ppo_8*8_adaptive_reasoner.",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_1", default="ppo_8*8_adaptive_reasoner.",#"MiniGrid-6x6-getkey-costum",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_2", default="ppo_8*8_adaptive_reasoner.",#"MiniGrid-6x6-gotodoor-costum",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model_3", default="ppo_8*8_adaptive_reasoner.",#"MiniGrid-6x6-togoal-costum",
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 0)")
parser.add_argument("--reasoner", type=bool, default=True,
                    help="random seed (default: 1)")
parser.add_argument("--adaptive", type=bool, default=False,
                    help="random seed (default: 1)")
parser.add_argument("--two_doors", type=str, default='two_doors',
                    help="options can be one_door, two_doors")
parser.add_argument("--size", type=int, default=8,
                    help="number of processes (default: 16)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

def plot_comparison(model_names, avg_returns, std_returns ):
    iterations = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000]
    model_names = ["ppo_reasoner", "ppo", "ppo_adaptive_reasoner"]

    import matplotlib.pyplot as plt
    plt.figure()

    for i, model_name in enumerate(model_names):
        plt.plot(iterations, avg_returns[i], label=model_name, marker='o')
        plt.fill_between(iterations,
                         [a - s for a, s in zip(avg_returns[i], std_returns[i])],
                         [a + s for a, s in zip(avg_returns[i], std_returns[i])],
                         alpha=0.2)

    # plt.plot(iterations, min_returns, label="Min", linestyle="--")
    # plt.plot(iterations, max_returns, label="Max", linestyle="--")

    plt.xlabel("Frames")
    plt.ylabel("Return per Episode")
    plt.title("Performance Evaluation at Different Training Frames")
    plt.legend()
    plt.grid(True)
    #plt.savefig("evaluate_returns.pdf", bbox_inches='tight')
    plt.show()


def set_reward_model( ):
    NSFR = initialize_reasoner(args.two_doors)
    reward_model = {'blue_door': 0, 'red_key': 0, 'red_door': 0, 'goal': 0, 'yellow_key': 0}
    V_T_mi, atoms_mi = predict_reward(NSFR,[1,1,1,1,1])
    lst1 = []
    for j, i in enumerate(atoms_mi):
        if i.pred.name == 'plan':
            if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'goal' and \
                    i.all_consts()[2].name == 'rg(A,G)':
                #  print(i, j)
                lst1.append(j)
    max_value, max_index = torch.max(V_T_mi[0][lst1], dim=0)
    #    print('max_value:', max_value, 'max_index:', max_index)
    # print('most possible plan', max_value, atoms_mi[lst1[max_index]])

    '''
    reward_model
    '''
    # Extract relevant actions
    if max_value > 0.9 :
        actions = [i.name for i in atoms_mi[lst1[max_index]].all_consts()[3:-1]]

        print('most possible plan', max_value, atoms_mi[lst1[max_index]])
        # Check if the actions match the blue door condition
        if set(actions) == {"gtbd(A,C)", "gtg(A,G)"}:
            reward_model['blue_door'] = 1
            reward_model['red_key'] = 0
            reward_model['red_door'] = 0
        # Check if the actions match the red key and red door condition
        elif set(actions) == {"grk(A,B)", "gtrd(A,C)", "gtg(A,G)"}:
            reward_model['blue_door'] = 0
            reward_model['red_key'] = 1
            reward_model['red_door'] = 1

def evaluate_hierarchical_model(args):
  #  envs = []

        #  make_env(args.env, 'goal', args.two_doors, args.reasoner, args.seed, render_mode="human")

    env = utils.make_env(args.env, 'goal', args.two_doors, args.reasoner, args.adaptive, args.size,
                         args.seed )

   # env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load multiple agent
    model_dir_1 = utils.get_model_dir(args.model_1)
    agent_1 = utils.Agent(env.observation_space, env.action_space, model_dir_1,
                          argmax=args.argmax, use_memory=args.memory, use_text=args.text)

    model_dir_2 = utils.get_model_dir(args.model_2)
    agent_2 = utils.Agent(env.observation_space, env.action_space, model_dir_2,
                          argmax=args.argmax, use_memory=args.memory, use_text=args.text)

    model_dir_3 = utils.get_model_dir(args.model_3)
    agent_3 = utils.Agent(env.observation_space, env.action_space, model_dir_3,
                          argmax=args.argmax, use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")
    # models_mapping
    model_mapping = {'key': agent_1, 'door': agent_2, 'goal': agent_3}


    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obs, _ = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:

        if obs['env_symbolic_state'][2] == 0:
            agent = model_mapping['key']
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent_1.analyze_feedback(reward, done)
            print('env_symbolic-state is: ', obs['env_symbolic_state'])
        if obs['env_symbolic_state'][2] == 1 and obs['env_symbolic_state'][4] == 0:
            agent = model_mapping['door']
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent_2.analyze_feedback(reward, done)

            # else:
        if obs['env_symbolic_state'][4] == 1:
            agent = model_mapping['goal']
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent_3.analyze_feedback(reward, done)

        log_episode_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)


        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return.item())
            logs["num_frames_per_episode"].append(log_episode_num_frames.item())

        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))


    # Print worst episodes

    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

    # Load environments
def evaluate_multiple_models(args):
    envs = []
    for i in range(args.procs):
      #  make_env(args.env, 'goal', args.two_doors, args.reasoner, args.seed, render_mode="human")

        env = utils.make_env(args.env, 'goal', args.two_doors, args.reasoner,args.adaptive, args.size, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load multiple agent
    model_names = ['model_50048_frames.pt', 'model_100096_frames.pt', 'model_150016_frames.pt', 'model_200064_frames.pt',
                   'model_250112_frames.pt', 'model_300160_frames.pt', 'model_350208_frames.pt', 'model_400256_frames.pt',
                   'model_450304_frames.pt', 'model_500352_frames.pt', 'model_550016_frames.pt', 'model_600192_frames.pt']
    avg_returns = []
    std_returns = []
    min_returns = []
    max_returns = []
    models = ["ppo_8*8_adaptive_reasoner.", "ppo_8*8_adaptive.", "ppo_8*8_adaptive_reasoner_dense2."]
    for model in models:
        avg_return_model = []
        std_return_model = []
        min_return_model = []
        max_return_model = []
        for model_name in model_names:

            model_dir = utils.get_model_dir(model)
            agent = utils.Agent(env.observation_space, env.action_space, model_dir, model_name,
                                argmax=args.argmax, num_envs=args.procs,
                                use_memory=args.memory, use_text=args.text)
            print("Agent loaded\n")

            # Initialize logs

            logs = {"num_frames_per_episode": [], "return_per_episode": []}

            # Run agent

            start_time = time.time()

            obss = env.reset()

            log_done_counter = 0
            log_episode_return = torch.zeros(args.procs, device=device)
            log_episode_num_frames = torch.zeros(args.procs, device=device)

            while log_done_counter < args.episodes:
                actions = agent.get_actions(obss)
                obss, rewards, terminateds, truncateds, _ = env.step(actions)
                dones = tuple(a | b for a, b in zip(terminateds, truncateds))
                agent.analyze_feedbacks(rewards, dones)

                log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
                log_episode_num_frames += torch.ones(args.procs, device=device)

                for i, done in enumerate(dones):
                    if done:
                        log_done_counter += 1
                        logs["return_per_episode"].append(log_episode_return[i].item())
                        logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

                mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
                log_episode_return *= mask
                log_episode_num_frames *= mask

            end_time = time.time()

            # Print logs

            num_frames = sum(logs["num_frames_per_episode"])
            fps = num_frames / (end_time - start_time)
            duration = int(end_time - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
                  .format(num_frames, fps, duration,
                          *return_per_episode.values(),
                          *num_frames_per_episode.values()))
            avg_return_model.append(return_per_episode['mean'])
            std_return_model.append(return_per_episode['std'])
            min_return_model.append(return_per_episode['min'])
            max_return_model.append(return_per_episode['max'])
        avg_returns.append(avg_return_model)
        std_returns.append(std_return_model)
        min_returns.append(min_return_model)
        max_returns.append(max_return_model)
    print("average returns ",avg_returns)
    print(std_returns)

    plot_comparison(models, avg_returns, std_returns)




        # Print worst episodes

       # n = args.worst_episodes_to_show
       # if n > 0:
       #     print("\n{} worst episodes:".format(n))

       #     indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
       #     for i in indexes[:n]:
       #         print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    print(f"Device: {device}\n")

    multipul_model_evaluation = False
    hierarchical_evaluation = True

    # Load environments
    if multipul_model_evaluation:
        evaluate_multiple_models(args)
    if hierarchical_evaluation:

        evaluate_hierarchical_model(args)