from typing import Any, Iterable, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ActType, ObsType


from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall

from .costum_WorldObject import Door, Goal, Key
from minigrid.manual_control import ManualControl
#from minigrid.minigrid_env import MiniGridEnv
from utils.minigrid_env import MiniGridEnv

from utils.Reasoner.src.predict_kandinsky import predict_reward,predict_step_reward, initialize_reasoner, initialize_percept_reasoner
from scripts.train_nn import MultimodalSequenceModel


def make_env(env_key,target='goal', two_doors = False, Reasoner= False, adaptive = False, size = 16, seed=None, render_mode=None):
    if env_key == 'custom':
        env = SimpleEnv(target= target, two_doors = two_doors, Reasoner=Reasoner, adaptive = adaptive, size  = size, render_mode=render_mode)
    else:
        env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env





class SimpleEnv(MiniGridEnv):


        """
        ## Description

        This environment has a key that the agent must pick up in order to unlock a
        goal and then get to the green goal square. This environment is difficult,
        because of the sparse reward, to solve using classical RL algorithms. It is
        useful to experiment with curiosity or curriculum learning.

        ## Mission Space

        "use the key to open the door and then get to the goal"

        ## Action Space

        | Num | Name         | Action                    |
        |-----|--------------|---------------------------|
        | 0   | left         | Turn left                 |
        | 1   | right        | Turn right                |
        | 2   | forward      | Move forward              |
        | 3   | pickup       | Pick up an object         |
        | 4   | drop         | Unused                    |
        | 5   | toggle       | Toggle/activate an object |
        | 6   | done         | Unused                    |

        ## Observation Encoding

        - Each tile is encoded as a 3 dimensional tuple:
            `(OBJECT_IDX, COLOR_IDX, STATE)`
        - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
            [minigrid/minigrid.py](minigrid/minigrid.py)
        - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

        ## Rewards

        A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

        ## Termination

        The episode ends if any one of the following conditions is met:

        1. The agent reaches the goal.
        2. Timeout (see `max_steps`).

        ## Registered Configurations

        - `MiniGrid-DoorKey-5x5-v0`
        - `MiniGrid-DoorKey-6x6-v0`
        - `MiniGrid-DoorKey-8x8-v0`
        - `MiniGrid-DoorKey-16x16-v0`

        """

        def __init__(self, target = None, two_doors = 'two_doors', Reasoner= False, adaptive = False, size=16, max_steps = None, **kwargs):
            if max_steps is None:
                max_steps = 10 * size ** 2
            mission_space = MissionSpace(mission_func=self._gen_mission)
            super().__init__(
                mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
            )
            self.target = target
            self.two_doors = two_doors
            self.reasoner = Reasoner
            self.adaptive = adaptive
            self.door =[]
            self.key = []
            'symbol_state is used to represent the observed state of the agent'
            'blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal'
            self.symbol_state = [0.5, 0.5, 1, 0.5, 1, 0.5, 0.5]
            self.symbol_distance = [45, 45, 45, 45, 45, 45, 45]
            self.change = 0
            self.lock = True
            self.adaptive_step = adaptive
            self.reward_history = 0
            self.enviroment_symbolic_state = [0, 0, 0, 0, 0]


            if self.reasoner:
                self.NSFR = initialize_reasoner(two_doors)
                'enviroment_symbol_state is used to represent the state of the enviroment'
                'blue key position, red key position, blue door position, red door position, goal position'
                #self.enviroment_symbolic_state = [0, 0, 0, 0, 0]
                self.reward_model = {'blue_door':0, 'red_key':0, 'red_door':0, 'goal':0, 'yellow_key':0}
                self.transition_sequence = []
                if self.adaptive_step:
                    self.NSFR_step = initialize_percept_reasoner(two_doors)

                self.symbolic_state_history = torch.zeros(1, 1, 5)
                self.action_history = torch.zeros(1, 1,  dtype=torch.int)
                self.image_history = torch.zeros(1, 1, 40, 40, 3)
                self.model = MultimodalSequenceModel()
                self.model.load_state_dict(torch.load("env_symbolic_model_mps.pt"))

        @staticmethod
        def _gen_mission():
            return "use or not use the key to open the door and then get to the goal"

        #def reset(self, **kwargs):
        #    self.symbolic_state_history = []
        #    self.action_history = []
        #    self.image_history = []
        #    return super().reset(**kwargs)
        def set_three_doors_env(self, width, height):
            import random
            num_walls = 1
            pos_walls = []

            if num_walls > 1:
                while len(pos_walls) < num_walls:
                    num = random.randint(2, width - 2)
                    if all(abs(num - x) >= 2 for x in pos_walls):  # Ensure minimum difference of 2
                        pos_walls.append(num)
                pos_walls = sorted(pos_walls)
                for i in pos_walls:
                    self.grid.vert_wall(i, 0)
                    self.grid.horz_wall(0, i)
            else:
                splitIdx = self._rand_int(3, width /2)  #
                pos_walls.append(splitIdx)
                self.grid.vert_wall(splitIdx, 0, width)
                self.grid.horz_wall(0, splitIdx, height)

            self.place_agent(size=(pos_walls[0], pos_walls[0]))


            'put three doors in different walls'
            'put the door in left wall'
            doorIdx_2 = self._rand_int(1, pos_walls[0] - 1)
            open = bool(random.getrandbits(1))
            door_2 = Door(color="blue", is_locked=False, is_open=open)
            self.door.append(door_2)
            self.put_obj(door_2, doorIdx_2, pos_walls[0])

            'put the door in down or right wall'
            doorIdx_3 = self._rand_int(pos_walls[0], width)
            open = bool(random.getrandbits(1))
            door_3 = Door(color="yellow", is_locked=False, is_open=open)
            self.door.append(door_3)
            down = bool(random.getrandbits(1))
            if down:
                self.put_obj(door_3, doorIdx_3, pos_walls[0])
            else:
                self.put_obj(door_3, pos_walls[0], doorIdx_3)


            'put the door and key in up'
            doorIdx = self._rand_int(1, pos_walls[0] - 1)
            door_1 = Door(color="red", is_locked=True)
            self.door.append(door_1)
            self.put_obj(door_1, pos_walls[0], doorIdx)
            print('door_red_locked:', door_1.is_locked)

            key_1 = Key(color="red")
            self.key.append(key_1)

            self.place_obj(obj=key_1, top=(0, 0), size=(pos_walls[0], pos_walls[0]))
        def set_two_doors_env(self, width, height):
            import random
            num_walls = 1
            pos_walls = []

            if num_walls > 1:
                while len(pos_walls) < num_walls:
                    num = random.randint(2, width - 2)
                    if all(abs(num - x) >= 2 for x in pos_walls):  # Ensure minimum difference of 2
                        pos_walls.append(num)
                pos_walls = sorted(pos_walls)
                for i in pos_walls:
                    self.grid.vert_wall(i, 0)
                    self.grid.horz_wall(0, i)
            else:
                splitIdx = self._rand_int(3, width - 2)  #
                pos_walls.append(splitIdx)
                self.grid.vert_wall(splitIdx, 0, splitIdx + 1)
                self.grid.horz_wall(0, splitIdx, splitIdx + 1)

            self.place_agent(size=(pos_walls[0], pos_walls[0]))

            'put two doors in one wall'
            # doorIdx1, doorIdx2 = random.sample(range(1, width -1), 2)
            # door_1 = Door(color = "red", is_locked=True)
            # door_2 = Door(color="blue", is_open=True)
            # self.door.append(door_1)
            # self.door.append(door_2)
            # self.put_obj(door_1, pos_walls[0], doorIdx1)
            # self.enviroment_symbolic_state[3]=(pos_walls[0], doorIdx)
            # self.put_obj(door_2,  doorIdx2, pos_walls[0])
            'put two doors in different walls'
            'put the door in left wall'
            doorIdx = self._rand_int(1, pos_walls[0] - 1)
            self.lock = not self.lock
            door_3 = Door(color="blue", is_locked=self.lock , is_open=not self.lock )
            #door_3 = Door(color="blue", is_locked=False, is_open=not (lock))
            self.door.append(door_3)
            self.put_obj(door_3, doorIdx, pos_walls[0])
            self.blue_door_pos = [doorIdx, pos_walls[0]]
            print('door_blue_locked:', door_3.is_locked)

            'put the door and key in up'
            doorIdx = self._rand_int(1, pos_walls[0] - 1)
            door_1 = Door(color="red", is_locked=True)
            self.door.append(door_1)
            self.put_obj(door_1, pos_walls[0], doorIdx)
            self.red_door_pos = [pos_walls[0], doorIdx]
            print('door_red_locked:', door_1.is_locked)

           # key_1 = Key(color="red")
            color_random = random.choice([True, False])
            if self.lock:
                key_1 = Key(color="red")
            else:
                key_1 = Key(color="yellow" if color_random else "red")
            self.key.append(key_1)


            self.key_pos  = self.place_obj(obj=key_1, top=(0, 0), size=(pos_walls[0], pos_walls[0]))

        def set_one_door_env(self, width, height):
            splitIdx = self._rand_int(2, width - 2)
            'if only one wall, activate this'
            self.grid.vert_wall(splitIdx, 0, width)

            self.place_agent(size=(splitIdx, height))

            doorIdx = self._rand_int(1, width - 2)

            door_1 = Door(color="red", is_locked=True)
            self.door.append(door_1)
            self.put_obj(door_1, splitIdx, doorIdx)

            key_1 = Key(color="red")
            self.key.append(key_1)
            self.place_obj(obj=key_1, top=(0, 0), size=(splitIdx, height))


        def _gen_grid(self, width, height):
            # Create an empty grid
            self.grid = Grid(width, height)
            self.change = 0
            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            # Place a goal in the bottom-right corner
            if self.target == 'goal':
                self.put_obj(Goal(), width - 2, height - 2)
                self.target_pos = [width - 2, height - 2]
                # Create a vertical splitting wall, generating a set of numbers to split multiple walls

            if self.two_doors == 'two_doors':
                self.set_two_doors_env(width, height)

            if self.two_doors == 'one_door':
                self.set_one_door_env(width, height)
            if self.two_doors == 'three_doors':
                self.set_three_doors_env(width, height)


            self.mission = "use the key to open the door and then get to the goal"

            'symbol_state is used to represent the observed state of the agent'
            'blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal'
            self.symbol_state = [0.5, 0.5, 1, 0.5, 1, 0.5, 0.5]
            self.symbol_distance = [45, 45, 45, 45, 45, 45, 45]
            self.reward_history = 0

            self.symbolic_state_history = torch.zeros(1, 1, 5)
            self.action_history = torch.zeros(1, 1,  dtype=torch.int)
            self.image_history = torch.zeros(1, 1, 40, 40, 3)


            if self.reasoner:

                'enviroment_symbol_state is used to represent the state of the enviroment'
                'blue key position, red key position, blue door position, red door position, goal position'
                self.enviroment_symbolic_state = [0, 0, 0, 0, 0]
                self.reward_model = {'blue_door': 0, 'red_key': 0, 'red_door': 0, 'goal':0, 'yellow_key':0}
                self.transition_sequence = []

        def get_observed_object(self):
            grid, vis_mask = self.gen_obs_grid()
            objects = []
            dis = []
            for i in range(grid.height):
                for j in range(grid.width):
                    if grid.get(i, j) is not None:
                       if grid.get(i, j).type != 'wall':
                            objects.append(grid.get(i, j))
                            dis.append((j-6)**2+(i-3)**2)
            return objects, dis

        def step(self, action):
            if self.reasoner:
                if self.adaptive:
                    return self.step_dual_reasoner(action)
                return self.step_reasoner(action)
            else:
                return self.step_ppo(action)

        'this step is for pure ppo'
        def step_ppo(self, action):

            terminated = False
            truncated = False
            reward = 0

            # Get the position in front of the agent
            fwd_pos = self.front_pos
            crt_pos = fwd_pos - self.dir_vec  # .array([self.agent_pos[0], self.agent_pos[1]])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            crt_cell = self.grid.get(crt_pos[0], crt_pos[1])

            self.step_count += 1
            if crt_cell is not None and crt_cell.type == 'door':

                 pass


            if crt_cell is not None and crt_cell.type == self.target:
                # if  V_T_mi[0][-1] > 0.91:
                self.symbol_state[-1] = 1
                terminated = True
                print('terminated')

                # print('reward: ', reward)
                # self.symbol_state = [0, 0, 1, 0, 1, 0, 0]
                # self.symbol_state[-1] = 1
                reward = self._reward()


            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = tuple(fwd_pos)
                #  if fwd_cell is not None and fwd_cell.type == "goal":
                #      terminated = True
                #      reward = self._reward()
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        self.agent_pos = tuple(fwd_pos)




            # Drop an object
            elif action == self.actions.drop:
                self.carrying = self.carrying
                pass
            # if not fwd_cell and self.carrying:

            #     self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
            #     self.carrying.cur_pos = fwd_pos
            #     self.carrying = None

            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)
            # Done action (not used by default)
            elif action == self.actions.done:
                pass
            else:
                raise ValueError(f"Unknown action: {action}")

            if self.step_count >= self.max_steps:
                truncated = True

            if self.render_mode == "human":
                self.render()

            obs = self.gen_obs()

            'if visulize is True, the mid_state will be added'
            # return obs, reward, terminated, truncated, mid_state, {}
            return obs, reward, terminated, truncated, {}


        'this step is for ppo reasoner'
        def step_reasoner(self, action):


            terminated = False
            truncated = False
            reward = 0

            # Get the position in front of the agent
            fwd_pos = self.front_pos
            crt_pos = fwd_pos - self.dir_vec  # .array([self.agent_pos[0], self.agent_pos[1]])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            crt_cell = self.grid.get(crt_pos[0], crt_pos[1])


            objects, dis = self.get_observed_object()

            #self.set_symbol_state(objects, dis)

            self.step_count += 1
            if self.step_count > 5:
                self.set_reward_model()
            if crt_cell is not None and crt_cell.type == 'door':


                if crt_cell.color == 'red':
                    env_state_idx = 3
                else:
                    env_state_idx = 2
                if self.enviroment_symbolic_state[ env_state_idx] == 0:
                    self.enviroment_symbolic_state[env_state_idx] = 1
                    #V_T_mi, atoms_mi = predict_reward(self.NSFR, self.symbol_state)

                    #reward =0.1*self.reward_model[crt_cell.color+'_'+crt_cell.type]-0.001 * (self.step_count / self.max_steps)*self.reward_model[crt_cell.color+'_'+crt_cell.type]#*100- 0.9 * (self.step_count / self.max_steps)
                    #print('door reward:', reward, crt_cell.color+'_'+crt_cell.type)
                    if env_state_idx == 3:
                        reward = 1 - 0.5 * (self.step_count / self.max_steps)  # - (self.step_count / self.max_steps)
                    elif env_state_idx == 2:
                        reward = 2 - 0.5 * (self.step_count / self.max_steps)
                print('door showed:',  crt_cell.color + '_' + crt_cell.type)

            if crt_cell is not None and crt_cell.type == self.target:

                #if  V_T_mi[0][-1] > 0.91:
                self.symbol_state[-1] = 1
                self.enviroment_symbolic_state[-1] = 1
                terminated = True
                print('terminated')
                #V_T_mi, atoms_mi = predict_reward(self.NSFR, self.symbol_state)
                #reward = 1- 0.9 * (self.step_count / self.max_steps)
                reward = 3 - 0.9*(self.step_count / self.max_steps)
                print('terminate reward:', reward)



            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = tuple(fwd_pos)
                #  if fwd_cell is not None and fwd_cell.type == "goal":
                #      terminated = True
                #      reward = self._reward()
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        self.agent_pos = tuple(fwd_pos)
                        #用environment 和reward model 来决定给不给reward
                        if fwd_cell.color == 'red':
                            env_state_idx = 1
                        else:
                            env_state_idx = 0
                        print(self.enviroment_symbolic_state)
                        if self.enviroment_symbolic_state[env_state_idx] == 0:
                            self.enviroment_symbolic_state[env_state_idx] = 1
                            if env_state_idx == 1:
                                # V_T_mi, atoms_mi = predict_reward(self.NSFR, self.symbol_state)

                                #reward = 0.1 * self.reward_model[fwd_cell.color + '_' + fwd_cell.type]-0.005 * (self.step_count / self.max_steps)*self.reward_model[fwd_cell.color + '_' + fwd_cell.type]
                                reward = 1 - 0.5 * (self.step_count / self.max_steps)
                                print('key reward:',fwd_cell.color + '_' + fwd_cell.type, reward)
                        print('key grabbed:', fwd_cell.color + '_' + fwd_cell.type)


            # Drop an object
            elif action == self.actions.drop:
                self.carrying = self.carrying
                pass

            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)
            # Done action (not used by default)
            elif action == self.actions.done:
                pass
            else:
                raise ValueError(f"Unknown action: {action}")

            if self.step_count >= self.max_steps:
                truncated = True

            if self.render_mode == "human":
                self.render()


            obs = self.gen_obs()


            'if visulize is True, the mid_state will be added'
            #return obs, reward, terminated, truncated, mid_state, {}
            return obs, reward, terminated, truncated, {}

        def step_dual_reasoner(self, action):




            #self.action_history.append(action)
            terminated = False
            truncated = False
            reward = 0



            self.model.eval()
            #predicted_probability = self.model(self.image_history,  self.action_history, self.symbolic_state_history , torch.tensor(action, dtype = torch.int).unsqueeze(0))
            #print('predicted:', predicted_probability.squeeze(0))
            #reward = self.set_adaptive_reward_model_all_plan(predicted_probability.squeeze(0))
           # print('reward is:', reward)
            # Get the position in front of the agent
            fwd_pos = self.front_pos
            crt_pos = fwd_pos - self.dir_vec  # .array([self.agent_pos[0], self.agent_pos[1]])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            crt_cell = self.grid.get(crt_pos[0], crt_pos[1])


            objects, dis = self.get_observed_object()

            self.set_symbol_state(objects, crt_pos)

            self.step_count += 1


            if crt_cell is not None and crt_cell.type == 'door':
                if crt_cell.color == 'red':
                    env_state_idx = 3
                else:
                    env_state_idx = 2
                if self.enviroment_symbolic_state[ env_state_idx] != 1:
                    self.enviroment_symbolic_state[env_state_idx] = 1
                    if env_state_idx == 3:
                        reward = 1- 0.5*(self.step_count / self.max_steps)# - (self.step_count / self.max_steps)
                    elif env_state_idx == 2:
                        reward = 2- 0.5*(self.step_count / self.max_steps)# - 2*(self.step_count / self.max_steps)
                   # reward = 1 + self.set_step_reward_model(objects, dis)#-0.5 * (self.step_count / self.max_steps)

            if crt_cell is not None and crt_cell.type == self.target:
                self.symbol_state[-1] = 1
                self.enviroment_symbolic_state[-1] = 1
                terminated = True
                reward = 3 - 0.9*(self.step_count / self.max_steps)# + self.set_step_reward_model(objects, dis) - 0.9 * (self.step_count / self.max_steps)

            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = tuple(fwd_pos)
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        self.agent_pos = tuple(fwd_pos)
                        #用environment 和reward model 来决定给不给reward
                        if fwd_cell.color == 'red':
                            env_state_idx = 1
                        else:
                            env_state_idx = 0
                        if self.enviroment_symbolic_state[env_state_idx] != 1:
                            self.enviroment_symbolic_state[env_state_idx] = 1
                            if env_state_idx == 1:
                                reward = 1- 0.5*(self.step_count / self.max_steps)#-(self.step_count / self.max_steps) #+ self.set_step_reward_model(objects, dis)#-0.5 * (self.step_count / self.max_steps)
#

            # Drop an object
            elif action == self.actions.drop:
                self.carrying = self.carrying
                pass


            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)
            # Done action (not used by default)
            elif action == self.actions.done:
                pass
            else:
                raise ValueError(f"Unknown action: {action}")

            if self.step_count >= self.max_steps:
                truncated = True

            if self.render_mode == "human":
                self.render()

            #print('curretn reward',cur_reward)

            obs = self.gen_obs()
            # Unsqueeze to match dimensions: (7, 7, 3) --> (1, 1, 7, 7, 3)
            current_image = torch.tensor(obs['full_image'], dtype=torch.float).unsqueeze(0).unsqueeze(0)

            # Append along time axis (dim=1)
            self.image_history = torch.cat([self.image_history, current_image], dim=1)
            'only keep the last 30 '
            if self.image_history.shape[1] > 30:
                self.image_history = self.image_history[:, -30:, :, :, :]

            current_environment_state = torch.tensor(self.enviroment_symbolic_state, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            self.symbolic_state_history = torch.cat([self.symbolic_state_history, current_environment_state], dim=1)
            'only keep the last 30 '
            if self.symbolic_state_history.shape[1] > 30:
                self.symbolic_state_history = self.symbolic_state_history[:, -30:, :]

            current_action = torch.tensor(action, dtype=torch.int).unsqueeze(0).unsqueeze(0)
            self.action_history = torch.cat([self.action_history, current_action], dim=1)
            'only keep the last 30 '
            if self.action_history.shape[1] > 30:
                self.action_history = self.action_history[:, -30:]

            #predicted_probability = self.enviroment_symbolic_state



            predicted_probability  = self.process_symbol_state()

            reward += self.set_adaptive_reward_model_all_plan(predicted_probability)/20-0.01
            #print('reward is:', reward)
            #print('predicted:', predicted_probability)
            return obs, reward, terminated, truncated, {}
        'this step is for ppo reasoner'
        def adaptive_step_reasoner(self, action):


            terminated = False
            truncated = False
            reward = 0

            # Get the position in front of the agent
            fwd_pos = self.front_pos
            crt_pos = fwd_pos - self.dir_vec  # .array([self.agent_pos[0], self.agent_pos[1]])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            crt_cell = self.grid.get(crt_pos[0], crt_pos[1])


            objects, dis = self.get_observed_object()

            self.set_symbol_state(objects, dis)

            self.step_count += 1


            if crt_cell is not None and crt_cell.type == 'door':


                if crt_cell.color == 'red':
                    env_state_idx = 3
                else:
                    env_state_idx = 2
                if self.enviroment_symbolic_state[env_state_idx] != 0:
                    self.enviroment_symbolic_state[env_state_idx] = 1

                    self.transition_sequence.append(crt_cell.color + '_' + crt_cell.type)
                    #self.set_adaptive_reward_model()
                    self.set_adaptive_neural_reward_model(self.symbolic_state_history, self.action_history, self.image_history)
                    #self.set_adaptive_reward_model_all_plan()
                    #reward = self.reward_model[crt_cell.color + '_' + crt_cell.type]
                    reward = max(self.reward_model[crt_cell.color + '_' + crt_cell.type] - 0.05 *(self.step_count / self.max_steps), 0)
                   # reward = 0.6 - 0.5 * (self.step_count / self.max_steps)#*self.reward_model[crt_cell.color+'_'+crt_cell.type]#*100- 0.9 * (self.step_count / self.max_steps)
                    print('door reward:', reward, crt_cell.color+'_'+crt_cell.type)
                    print('door reward for ppo:', reward)
                   # print('door reward reasoner:', reward1)

            if crt_cell is not None and crt_cell.type == self.target:

                #if  V_T_mi[0][-1] > 0.91:
                self.symbol_state[-1] = 1
                self.enviroment_symbolic_state[-1] = 1
                terminated = True
                print('terminated')
                self.transition_sequence.append('goal')
            #    self.transition_sequence.append(crt_cell.type)
                self.set_adaptive_reward_model()
                #self.set_adaptive_reward_model_all_plan()
              #  reward = self.reward_model[crt_cell.type]
                if len(self.transition_sequence) == 2:
                    coeff= 10
                elif len(self.transition_sequence) == 3:
                    coeff= 30
                else:
                    coeff = 1
                reward = coeff*self.reward_model[crt_cell.type]- 0.5 * (self.step_count / self.max_steps)
                print('terminate reward:', reward)
                print('terminated sequence:', self.transition_sequence)



            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = tuple(fwd_pos)
                #  if fwd_cell is not None and fwd_cell.type == "goal":
                #      terminated = True
                #      reward = self._reward()
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        self.agent_pos = tuple(fwd_pos)
                        #用environment 和reward model 来决定给不给reward
                        if fwd_cell.color == 'red':
                            env_state_idx = 1
                        else:
                            env_state_idx = 0
                        print(self.enviroment_symbolic_state)
                        if self.enviroment_symbolic_state[env_state_idx] == 0:
                            self.enviroment_symbolic_state[env_state_idx] = 1
                            self.transition_sequence.append(fwd_cell.color + '_' + fwd_cell.type)
                            self.set_adaptive_reward_model()
                            reward = max(self.reward_model[fwd_cell.color + '_' + fwd_cell.type]-0.05*(self.step_count / self.max_steps), 0)
                            print('key reward:',fwd_cell.color + '_' + fwd_cell.type, reward)



            # Drop an object
            elif action == self.actions.drop:
                self.carrying = self.carrying
                pass

            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)



            # Done action (not used by default)
            elif action == self.actions.done:
                pass
            else:
                raise ValueError(f"Unknown action: {action}")

            if self.step_count >= self.max_steps:
                truncated = True

            if self.render_mode == "human":
                self.render()

            obs = self.gen_obs()

            #print('reward:', reward)
            'if visulize is True, the mid_state will be added'
            #return obs, reward, terminated, truncated, mid_state, {}
            return obs, reward, terminated, truncated, {}
        def set_symbol_state(self, objects, cur_pos):
            '''
            assign probabilities to observed objects
            '''
            'symbol_state is used to represent the observed state of the agent'
            'blue key, red key, blue door closed, blue door open, red door closed, red door open'
            'this is for later use for sing the ovserved onject in the symbolic state'

            self.symbol_distance = [45, 45, 45, 45, 45, 45, 45]
            self.symbol_distance[1] = (cur_pos[0] - self.key_pos[0]) ** 2 + (cur_pos[1] - self.key_pos[1]) ** 2
            self.symbol_distance[0] = (cur_pos[0] - self.key_pos[0]) ** 2 + (cur_pos[1] - self.key_pos[1]) ** 2
            self.symbol_distance[3] = (cur_pos[0] - self.blue_door_pos[0]) ** 2 + (cur_pos[1] - self.blue_door_pos[1]) ** 2
            self.symbol_distance[2] = (cur_pos[0] - self.blue_door_pos[0]) ** 2 + (cur_pos[1] - self.blue_door_pos[1]) ** 2
            self.symbol_distance[5] = (cur_pos[0] - self.red_door_pos[0]) ** 2 + (cur_pos[1] - self.red_door_pos[1]) ** 2
            self.symbol_distance[4] = (cur_pos[0] - self.red_door_pos[0]) ** 2 + (cur_pos[1] - self.red_door_pos[1]) ** 2
            self.symbol_distance[-1] = (cur_pos[0] - self.target_pos[0]) ** 2 + (cur_pos[1] - self.target_pos[1]) ** 2
            '''
            for obj  in objects:
                if obj.type == 'key':
                    if obj.color == 'red':
                        #if dis == 0:
                        #    self.symbol_state[1] = 1
                        self.symbol_state[1] = 1
                        self.symbol_distance[1] = (cur_pos[0]-self.key_pos[0])**2+(cur_pos[1]-self.key_pos[1])**2
                    if obj.color == 'blue':
                        #if dis == 0:
                        #    self.symbol_state[0] self.symbol_state[5]= 1
                        self.symbol_state[0] = 1
                        self.symbol_distance[0] = (cur_pos[0]-self.key_pos[0])**2+(cur_pos[1]-self.key_pos[1])**2
                if obj.type == 'door':
                    if obj.color == 'red':
                        self.symbol_distance[5] = (cur_pos[0]-self.red_door_pos[0])**2+(cur_pos[1]-self.red_door_pos[1])**2
                        self.symbol_distance[4] = (cur_pos[0]-self.red_door_pos[0])**2+(cur_pos[1]-self.red_door_pos[1])**2
                        if obj.is_open:
                            self.symbol_state[5] = 1
                            self.symbol_state[4] = 0

                    if obj.color == 'blue':
                        self.symbol_distance[3] = (cur_pos[0]-self.blue_door_pos[0])**2+(cur_pos[1]-self.blue_door_pos[1])**2
                        self.symbol_distance[2] = (cur_pos[0]-self.blue_door_pos[0])**2+(cur_pos[1]-self.blue_door_pos[1])**2
                        if obj.is_open:
                            self.symbol_state[3] = 1
                            self.symbol_state[2] = 0

                if obj.type == 'goal':
                    self.symbol_state[-1] = 1
                    self.symbol_distance[-1] = (cur_pos[0]-self.target_pos[0])**2+(cur_pos[1]-self.target_pos[1])**2
        '''
        'blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal'

        def process_symbol_state(self):
            processed_symbol = [0.3, 0.3, 1, 0.3, 1, 0.3, 0.3]
            processed_symbol[1] = max(0.5 + 1 / (self.symbol_distance[1] + 2), self.enviroment_symbolic_state[1])
            processed_symbol[3] = max(0.5 + (1 - self.symbol_state[2]) / (self.symbol_distance[3] + 2),
                                      (1 - self.symbol_state[2]) - 0.2, self.enviroment_symbolic_state[2])
            processed_symbol[5] = max(0.5 + self.enviroment_symbolic_state[1] / (self.symbol_distance[5] + 2),
                                      (1 - self.symbol_state[4]) - 0.2, self.enviroment_symbolic_state[3])
            processed_symbol[6] = max(0.5 + (self.enviroment_symbolic_state[2] + self.enviroment_symbolic_state[3]) / (
                        self.symbol_distance[6] + 4),
                                      self.enviroment_symbolic_state[4])
            #with open("output.txt", "a") as f:
            #    f.write("processed_symbol:," + str(processed_symbol) + "\n")
            #    f.write("enviroment_symbolic_state:," + str(self.enviroment_symbolic_state) + "\n")
            #    f.write("symbol_distance:," + str(self.symbol_distance) + "\n")

         #   print('processed_symbol:', processed_symbol)
         #   print('environment_symbol:', self.enviroment_symbolic_state)
         #   print('blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal')
         #   print('symbol_distance:', self.symbol_distance)

            return processed_symbol
        def set_reward_model(self):
            V_T_mi, atoms_mi = predict_reward(self.NSFR, self.symbol_state)
            lst1 = []
            for j, i in enumerate(atoms_mi):
                if i.pred.name == 'plan':
                    if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'rg(A,G)' and \
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
            if max_value > 0.9 and self.change == 0 :
                actions = [i.name for i in atoms_mi[lst1[max_index]].all_consts()[3:-1]]
                self.change= 1
                print('most possible plan', max_value, atoms_mi[lst1[max_index]])
            # Check if the actions match the blue door condition
                if set(actions) == {"gtbd(A,C)", "gtg(A,G)"}:
                    self.reward_model['blue_door'] = 1
                    self.reward_model['red_key'] = 0
                    self.reward_model['red_door'] = 0
                # Check if the actions match the red key and red door condition
                elif set(actions) == {"grk(A,B)", "gtrd(A,C)", "gtg(A,G)"}:
                    self.reward_model['blue_door'] = 0
                    self.reward_model['red_key'] = 1
                    self.reward_model['red_door'] = 1
         #   print('reward model:', self.reward_model)

        #    if self.reward_model['blue_door'] == 1:
          #      input("Paused. Press Enter to continue...")


        def set_step_reward_model(self, obj, dis):
            processed_symbol_state = self.process_symbol_state()
            #V_T_mi, atoms_mi = predict_step_reward(self.NSFR, self.NSFR_step, self.symbol_state,self.symbol_distance)
            V_T_mi, atoms_mi = predict_reward(self.NSFR, processed_symbol_state)
            lst1 = []
            for j, i in enumerate(atoms_mi):
                if i.pred.name == 'plan':
                    if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'rg(A,G)' and \
                            i.all_consts()[2].name == 'rg(A,G)':

                        lst1.append(j)

            cur_reward_value = 0

            # print('lst1:', lst1)
            if len(lst1) > 0:
                # for i in lst1:
                #    print(atoms_mi[i], V_T_mi[0][i])
                print('reward is:', torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0))
                # if torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0)< 0:
                #    for i in lst1:
                #        print(atoms_mi[i], V_T_mi[0][i])
                #    for j, k in enumerate(atoms_mi):
                #        if k.pred.name == 'plan':
                #            if k.all_consts()[0].name == 'initial(A)'  and k.all_consts()[2].name == 'rg(A,G)':
                #                print(atoms_mi[j],V_T_mi[0][j])
                top_five_value = torch.topk(V_T_mi[0][lst1], len(lst1), dim=0)
                cur_reward_value = torch.logsumexp(torch.log(top_five_value.values), dim=0)
            return cur_reward_value



        def set_adaptive_reward_model(self):
            sequence_no_yellowKey = self.transition_sequence.copy()

            if 'yellow_key' in sequence_no_yellowKey:
                sequence_no_yellowKey.remove('yellow_key')
            reward_value = 1
            for i in range(len(sequence_no_yellowKey)):
                reward_value *= self.get_possibility_with_sequence(sequence_no_yellowKey[:i+1])


            self.reward_model[self.transition_sequence[-1]]  = reward_value
            self.reward_model['yellow_key'] = 0
            if 'yellow_key' in self.transition_sequence:
                self.transition_sequence.remove('yellow_key')
        def get_possibility_with_sequence(self, sequence):
            V_T_mi, atoms_mi = predict_reward(self.NSFR, self.enviroment_symbolic_state)
            lst1 = []
            map_transitionsequnce = {'red_key': 'grk(A,B)', 'blue_door': 'gtbd(A,C)', 'red_door': 'gtrd(A,C)',
                                     'goal': 'gtg(A,G)', 'yellow_key': 'gyk(A,B)'}
            if len(sequence) == 1:
                for j, i in enumerate(atoms_mi):
                    if i.pred.name == 'plan':
                        if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[2].name == 'rg(A,G)':
                            if i.all_consts()[:-1][-1].name == map_transitionsequnce[sequence[-1]]:
                                # print(i, V_T_mi[0][j])
                                lst1.append(j)

            if len(sequence) == 2:
                'the actions are gtbd(A,C),grk(A,B),'
                for j, i in enumerate(atoms_mi):
                    if i.pred.name == 'plan':
                        if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[2].name == 'rg(A,G)':
                            if (i.all_consts()[:-1][-1].name == map_transitionsequnce[sequence[-2]] and
                                i.all_consts()[:-1][-2].name == map_transitionsequnce[
                                    sequence[-1]]) :
                                lst1.append(j)
            if len(sequence) == 3:
                'the actions are gtbd(A,C),grk(A,B),'
                for j, i in enumerate(atoms_mi):
                    if i.pred.name == 'plan':
                        if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[2].name == 'rg(A,G)':
                            if (i.all_consts()[:-1][-1].name == map_transitionsequnce[sequence[-3]] and
                                i.all_consts()[:-1][-2].name == map_transitionsequnce[sequence[-2]] and
                                i.all_consts()[:-1][-3].name == map_transitionsequnce[sequence[-1]]) :
                                lst1.append(j)
            cur_reward_value = 0

            # print('lst1:', lst1)
            if len(lst1) > 0:
                print('reward is:', torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0))
                top_five_value = torch.topk(V_T_mi[0][lst1], len(lst1), dim=0)
                cur_reward_value = torch.logsumexp(torch.log(top_five_value.values), dim=0)
            return max(cur_reward_value,0)
        def set_adaptive_reward_model_all_plan(self, probability):
            V_T_mi, atoms_mi = predict_reward(self.NSFR, probability)
            lst1 = []
            map_transitionsequnce = {'red_key': 'grk(A,B)', 'blue_door': 'gtbd(A,C)', 'red_door': 'gtrd(A,C)',
                                     'goal': 'gtg(A,G)'}
            for j, i in enumerate(atoms_mi):
                if i.pred.name == 'plan':
                    if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'rg(A,G)' and \
                            i.all_consts()[2].name == 'rg(A,G)':
                            lst1.append(j)
          #  if torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0) < 0:
          #      for i in lst1:
          #          print(atoms_mi[i], V_T_mi[0][i])
          #      for j, k in enumerate(atoms_mi):
          #          if k.pred.name == 'plan':
          #              if k.all_consts()[0].name == 'initial(A)' and k.all_consts()[1].name == 'rg(A,G)' and \
          #                      k.all_consts()[2].name == 'rg(A,G)':
                            #print('11111')
          #                  print(atoms_mi[j], V_T_mi[0][j])
            top_five_value = torch.topk(V_T_mi[0][lst1], len(lst1), dim=0)

            return torch.logsumexp(torch.log(top_five_value.values),dim=0)






        def adaptive_step_reasoner(self, action):


            terminated = False
            truncated = False
            reward = 0

            # Get the position in front of the agent
            fwd_pos = self.front_pos
            crt_pos = fwd_pos - self.dir_vec  # .array([self.agent_pos[0], self.agent_pos[1]])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            crt_cell = self.grid.get(crt_pos[0], crt_pos[1])


            objects, dis = self.get_observed_object()

            self.set_symbol_state(objects, dis)

            self.step_count += 1


            if crt_cell is not None and crt_cell.type == 'door':


                if crt_cell.color == 'red':
                    env_state_idx = 3
                else:
                    env_state_idx = 2
                if self.enviroment_symbolic_state[env_state_idx] != 0:
                    self.enviroment_symbolic_state[env_state_idx] = 1

                    self.transition_sequence.append(crt_cell.color + '_' + crt_cell.type)
                    #self.set_adaptive_reward_model()
                    self.set_adaptive_neural_reward_model(self.symbolic_state_history, self.action_history, self.image_history)
                    #self.set_adaptive_reward_model_all_plan()
                    #reward = self.reward_model[crt_cell.color + '_' + crt_cell.type]
                    reward = max(self.reward_model[crt_cell.color + '_' + crt_cell.type] - 0.05 *(self.step_count / self.max_steps), 0)
                   # reward = 0.6 - 0.5 * (self.step_count / self.max_steps)#*self.reward_model[crt_cell.color+'_'+crt_cell.type]#*100- 0.9 * (self.step_count / self.max_steps)
                    print('door reward:', reward, crt_cell.color+'_'+crt_cell.type)
                    print('door reward for ppo:', reward)
                   # print('door reward reasoner:', reward1)

            if crt_cell is not None and crt_cell.type == self.target:

                #if  V_T_mi[0][-1] > 0.91:
                self.symbol_state[-1] = 1
                self.enviroment_symbolic_state[-1] = 1
                terminated = True
                print('terminated')
                self.transition_sequence.append('goal')
            #    self.transition_sequence.append(crt_cell.type)
                self.set_adaptive_reward_model()
                #self.set_adaptive_reward_model_all_plan()
              #  reward = self.reward_model[crt_cell.type]
                if len(self.transition_sequence) == 2:
                    coeff= 10
                elif len(self.transition_sequence) == 3:
                    coeff= 30
                else:
                    coeff = 1
                reward = coeff*self.reward_model[crt_cell.type]- 0.5 * (self.step_count / self.max_steps)
                print('terminate reward:', reward)
                print('terminated sequence:', self.transition_sequence)



            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = tuple(fwd_pos)
                #  if fwd_cell is not None and fwd_cell.type == "goal":
                #      terminated = True
                #      reward = self._reward()
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        self.agent_pos = tuple(fwd_pos)
                        #用environment 和reward model 来决定给不给reward
                        if fwd_cell.color == 'red':
                            env_state_idx = 1
                        else:
                            env_state_idx = 0
                        print(self.enviroment_symbolic_state)
                        if self.enviroment_symbolic_state[env_state_idx] == 0:
                            self.enviroment_symbolic_state[env_state_idx] = 1
                            self.transition_sequence.append(fwd_cell.color + '_' + fwd_cell.type)
                            self.set_adaptive_reward_model()
                            reward = max(self.reward_model[fwd_cell.color + '_' + fwd_cell.type]-0.05*(self.step_count / self.max_steps), 0)
                            print('key reward:',fwd_cell.color + '_' + fwd_cell.type, reward)



            # Drop an object
            elif action == self.actions.drop:
                self.carrying = self.carrying
                pass

            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)



            # Done action (not used by default)
            elif action == self.actions.done:
                pass
            else:
                raise ValueError(f"Unknown action: {action}")

            if self.step_count >= self.max_steps:
                truncated = True

            if self.render_mode == "human":
                self.render()

            obs = self.gen_obs()

            #print('reward:', reward)
            'if visulize is True, the mid_state will be added'
            #return obs, reward, terminated, truncated, mid_state, {}
            return obs, reward, terminated, truncated, {}
        def set_symbol_state(self, objects, cur_pos):
            '''
            assign probabilities to observed objects
            '''
            'symbol_state is used to represent the observed state of the agent'
            'blue key, red key, blue door closed, blue door open, red door closed, red door open'
            'this is for later use for sing the ovserved onject in the symbolic state'

            self.symbol_distance = [45, 45, 45, 45, 45, 45, 45]
            self.symbol_distance[1] = (cur_pos[0] - self.key_pos[0]) ** 2 + (cur_pos[1] - self.key_pos[1]) ** 2
            self.symbol_distance[0] = (cur_pos[0] - self.key_pos[0]) ** 2 + (cur_pos[1] - self.key_pos[1]) ** 2
            self.symbol_distance[3] = (cur_pos[0] - self.blue_door_pos[0]) ** 2 + (cur_pos[1] - self.blue_door_pos[1]) ** 2
            self.symbol_distance[2] = (cur_pos[0] - self.blue_door_pos[0]) ** 2 + (cur_pos[1] - self.blue_door_pos[1]) ** 2
            self.symbol_distance[5] = (cur_pos[0] - self.red_door_pos[0]) ** 2 + (cur_pos[1] - self.red_door_pos[1]) ** 2
            self.symbol_distance[4] = (cur_pos[0] - self.red_door_pos[0]) ** 2 + (cur_pos[1] - self.red_door_pos[1]) ** 2
            self.symbol_distance[-1] = (cur_pos[0] - self.target_pos[0]) ** 2 + (cur_pos[1] - self.target_pos[1]) ** 2
            '''
            for obj  in objects:
                if obj.type == 'key':
                    if obj.color == 'red':
                        #if dis == 0:
                        #    self.symbol_state[1] = 1
                        self.symbol_state[1] = 1
                        self.symbol_distance[1] = (cur_pos[0]-self.key_pos[0])**2+(cur_pos[1]-self.key_pos[1])**2
                    if obj.color == 'blue':
                        #if dis == 0:
                        #    self.symbol_state[0] self.symbol_state[5]= 1
                        self.symbol_state[0] = 1
                        self.symbol_distance[0] = (cur_pos[0]-self.key_pos[0])**2+(cur_pos[1]-self.key_pos[1])**2
                if obj.type == 'door':
                    if obj.color == 'red':
                        self.symbol_distance[5] = (cur_pos[0]-self.red_door_pos[0])**2+(cur_pos[1]-self.red_door_pos[1])**2
                        self.symbol_distance[4] = (cur_pos[0]-self.red_door_pos[0])**2+(cur_pos[1]-self.red_door_pos[1])**2
                        if obj.is_open:
                            self.symbol_state[5] = 1
                            self.symbol_state[4] = 0

                    if obj.color == 'blue':
                        self.symbol_distance[3] = (cur_pos[0]-self.blue_door_pos[0])**2+(cur_pos[1]-self.blue_door_pos[1])**2
                        self.symbol_distance[2] = (cur_pos[0]-self.blue_door_pos[0])**2+(cur_pos[1]-self.blue_door_pos[1])**2
                        if obj.is_open:
                            self.symbol_state[3] = 1
                            self.symbol_state[2] = 0

                if obj.type == 'goal':
                    self.symbol_state[-1] = 1
                    self.symbol_distance[-1] = (cur_pos[0]-self.target_pos[0])**2+(cur_pos[1]-self.target_pos[1])**2
        '''
        'blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal'

        def process_symbol_state(self):
            processed_symbol = [0.3, 0.3, 1, 0.3, 1, 0.3, 0.3]
            processed_symbol[1] = max(0.5 + 1 / (self.symbol_distance[1] + 2), self.enviroment_symbolic_state[1])
            processed_symbol[3] = max(0.5 + (1 - self.symbol_state[2]) / (self.symbol_distance[3] + 2),
                                      (1 - self.symbol_state[2]) - 0.2, self.enviroment_symbolic_state[2])
            processed_symbol[5] = max(0.5 + self.enviroment_symbolic_state[1] / (self.symbol_distance[5] + 2),
                                      (1 - self.symbol_state[4]) - 0.2, self.enviroment_symbolic_state[3])
            processed_symbol[6] = max(0.5 + (self.enviroment_symbolic_state[2] + self.enviroment_symbolic_state[3]) / (
                        self.symbol_distance[6] + 4),
                                      self.enviroment_symbolic_state[4])
            #with open("output.txt", "a") as f:
            #    f.write("processed_symbol:," + str(processed_symbol) + "\n")
            #    f.write("enviroment_symbolic_state:," + str(self.enviroment_symbolic_state) + "\n")
            #    f.write("symbol_distance:," + str(self.symbol_distance) + "\n")

         #   print('processed_symbol:', processed_symbol)
         #   print('environment_symbol:', self.enviroment_symbolic_state)
         #   print('blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal')
         #   print('symbol_distance:', self.symbol_distance)

            return processed_symbol
        def set_reward_model(self):
            V_T_mi, atoms_mi = predict_reward(self.NSFR, self.symbol_state)
            lst1 = []
            for j, i in enumerate(atoms_mi):
                if i.pred.name == 'plan':
                    if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'rg(A,G)' and \
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
            if max_value > 0.9 and self.change == 0 :
                actions = [i.name for i in atoms_mi[lst1[max_index]].all_consts()[3:-1]]
                self.change= 1
                print('most possible plan', max_value, atoms_mi[lst1[max_index]])
            # Check if the actions match the blue door condition
                if set(actions) == {"gtbd(A,C)", "gtg(A,G)"}:
                    self.reward_model['blue_door'] = 1
                    self.reward_model['red_key'] = 0
                    self.reward_model['red_door'] = 0
                # Check if the actions match the red key and red door condition
                elif set(actions) == {"grk(A,B)", "gtrd(A,C)", "gtg(A,G)"}:
                    self.reward_model['blue_door'] = 0
                    self.reward_model['red_key'] = 1
                    self.reward_model['red_door'] = 1
         #   print('reward model:', self.reward_model)

        #    if self.reward_model['blue_door'] == 1:
          #      input("Paused. Press Enter to continue...")


        def set_step_reward_model(self, obj, dis):
            processed_symbol_state = self.process_symbol_state()
            #V_T_mi, atoms_mi = predict_step_reward(self.NSFR, self.NSFR_step, self.symbol_state,self.symbol_distance)
            V_T_mi, atoms_mi = predict_reward(self.NSFR, processed_symbol_state)
            lst1 = []
            for j, i in enumerate(atoms_mi):
                if i.pred.name == 'plan':
                    if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'rg(A,G)' and \
                            i.all_consts()[2].name == 'rg(A,G)':

                        lst1.append(j)

            cur_reward_value = 0

            # print('lst1:', lst1)
            if len(lst1) > 0:
                # for i in lst1:
                #    print(atoms_mi[i], V_T_mi[0][i])
                print('reward is:', torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0))
                # if torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0)< 0:
                #    for i in lst1:
                #        print(atoms_mi[i], V_T_mi[0][i])
                #    for j, k in enumerate(atoms_mi):
                #        if k.pred.name == 'plan':
                #            if k.all_consts()[0].name == 'initial(A)'  and k.all_consts()[2].name == 'rg(A,G)':
                #                print(atoms_mi[j],V_T_mi[0][j])
                top_five_value = torch.topk(V_T_mi[0][lst1], len(lst1), dim=0)
                cur_reward_value = torch.logsumexp(torch.log(top_five_value.values), dim=0)
            return cur_reward_value



        def set_adaptive_reward_model(self):
            sequence_no_yellowKey = self.transition_sequence.copy()

            if 'yellow_key' in sequence_no_yellowKey:
                sequence_no_yellowKey.remove('yellow_key')
            reward_value = 1
            for i in range(len(sequence_no_yellowKey)):
                reward_value *= self.get_possibility_with_sequence(sequence_no_yellowKey[:i+1])


            self.reward_model[self.transition_sequence[-1]]  = reward_value
            self.reward_model['yellow_key'] = 0
            if 'yellow_key' in self.transition_sequence:
                self.transition_sequence.remove('yellow_key')
        def get_possibility_with_sequence(self, sequence):
            V_T_mi, atoms_mi = predict_reward(self.NSFR, self.enviroment_symbolic_state)
            lst1 = []
            map_transitionsequnce = {'red_key': 'grk(A,B)', 'blue_door': 'gtbd(A,C)', 'red_door': 'gtrd(A,C)',
                                     'goal': 'gtg(A,G)', 'yellow_key': 'gyk(A,B)'}
            if len(sequence) == 1:
                for j, i in enumerate(atoms_mi):
                    if i.pred.name == 'plan':
                        if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[2].name == 'rg(A,G)':
                            if i.all_consts()[:-1][-1].name == map_transitionsequnce[sequence[-1]]:
                                # print(i, V_T_mi[0][j])
                                lst1.append(j)

            if len(sequence) == 2:
                'the actions are gtbd(A,C),grk(A,B),'
                for j, i in enumerate(atoms_mi):
                    if i.pred.name == 'plan':
                        if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[2].name == 'rg(A,G)':
                            if (i.all_consts()[:-1][-1].name == map_transitionsequnce[sequence[-2]] and
                                i.all_consts()[:-1][-2].name == map_transitionsequnce[
                                    sequence[-1]]) :
                                lst1.append(j)
            if len(sequence) == 3:
                'the actions are gtbd(A,C),grk(A,B),'
                for j, i in enumerate(atoms_mi):
                    if i.pred.name == 'plan':
                        if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[2].name == 'rg(A,G)':
                            if (i.all_consts()[:-1][-1].name == map_transitionsequnce[sequence[-3]] and
                                i.all_consts()[:-1][-2].name == map_transitionsequnce[sequence[-2]] and
                                i.all_consts()[:-1][-3].name == map_transitionsequnce[sequence[-1]]) :
                                lst1.append(j)
            cur_reward_value = 0

            # print('lst1:', lst1)
            if len(lst1) > 0:
                print('reward is:', torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0))
                top_five_value = torch.topk(V_T_mi[0][lst1], len(lst1), dim=0)
                cur_reward_value = torch.logsumexp(torch.log(top_five_value.values), dim=0)
            return max(cur_reward_value,0)
        def set_adaptive_reward_model_all_plan(self, probability):
            V_T_mi, atoms_mi = predict_reward(self.NSFR, probability)
            lst1 = []
            map_transitionsequnce = {'red_key': 'grk(A,B)', 'blue_door': 'gtbd(A,C)', 'red_door': 'gtrd(A,C)',
                                     'goal': 'gtg(A,G)'}
            for j, i in enumerate(atoms_mi):
                if i.pred.name == 'plan':
                    if i.all_consts()[0].name == 'initial(A)' and i.all_consts()[1].name == 'rg(A,G)' and \
                            i.all_consts()[2].name == 'rg(A,G)':
                            lst1.append(j)
          #  if torch.logsumexp(torch.log(V_T_mi[0][lst1]), dim=0) < 0:
          #      for i in lst1:
          #          print(atoms_mi[i], V_T_mi[0][i])
          #      for j, k in enumerate(atoms_mi):
          #          if k.pred.name == 'plan':
          #              if k.all_consts()[0].name == 'initial(A)' and k.all_consts()[1].name == 'rg(A,G)' and \
          #                      k.all_consts()[2].name == 'rg(A,G)':
                            #print('11111')
          #                  print(atoms_mi[j], V_T_mi[0][j])
            top_five_value = torch.topk(V_T_mi[0][lst1], len(lst1), dim=0)

            return torch.logsumexp(torch.log(top_five_value.values),dim=0)





