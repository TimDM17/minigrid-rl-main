from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from gymnasium.spaces import Discrete


class CustomEnv(MiniGridEnv):
    def __init__(self, size=16, num_rooms=4, num_doors=3, num_keys=4):
        # Define the mission space
        mission_space = MissionSpace(lambda: "Navigate and open all doors using keys.")

        super().__init__(
            grid_size=size,
            max_steps=500,
            mission_space=mission_space,
        )
        self.num_rooms = num_rooms
        self.num_doors = num_doors
        self.num_keys = num_keys
        self.size = size

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Define room sizes
        room_size = self.size // 2
        self.rooms = [
            (0, 0, room_size - 1, room_size - 1),  # Room 1
            (0, room_size, room_size - 1, self.size - 1),  # Room 2
            (room_size, 0, self.size - 1, room_size - 1),  # Room 3
            (room_size, room_size, self.size - 1, self.size - 1),  # Room 4
        ]

        # Add room walls
        for i in range(room_size - 1, room_size + 1):
            for j in range(width):
                self.grid.set(j, i, Wall())
                self.grid.set(i, j, Wall())

        # Add doors
        self.doors = []
        door_colors = ["red", "green", "blue"]
        for i in range(self.num_doors):
            x, y = self._get_door_position(i)
            door = Door(door_colors[i], is_locked=True)
            self.grid.set(x, y, door)
            self.doors.append((x, y, door_colors[i]))

        # Add keys
        self.keys = []
        for i in range(self.num_keys):
            room_idx = i % self.num_rooms
            key_color = door_colors[i] if i < len(door_colors) else "yellow"
            x, y = self._place_in_room(room_idx)
            key = Key(key_color)
            self.grid.set(x, y, key)
            self.keys.append((x, y, key_color))

        # Place the agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0

    def _get_door_position(self, door_idx):
        """Get the position for a door based on its index."""
        if door_idx == 0:
            return self.size // 2 - 1, self.size // 4
        elif door_idx == 1:
            return self.size // 4, self.size // 2
        else:
            return self.size // 2, 3 * self.size // 4

    def _place_in_room(self, room_idx):
        """Place an object randomly in a room."""
        x1, y1, x2, y2 = self.rooms[room_idx]
        return self._rand_int(x1 + 1, x2), self._rand_int(y1 + 1, y2)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Handle unlocking logic
        if action == self.actions.pickup:
            cell = self.grid.get(*self.agent_pos)
            if cell and cell.type == "key":
                key_color = cell.color
                if key_color == "yellow":  # Master key
                    for door_x, door_y, _ in self.doors:
                        door = self.grid.get(door_x, door_y)
                        if door and door.type == "door":
                            door.is_locked = False
                else:
                    # Unlock the specific door
                    for door_x, door_y, door_color in self.doors:
                        if door_color == key_color:
                            door = self.grid.get(door_x, door_y)
                            if door and door.type == "door":
                                door.is_locked = False

        return obs, reward, terminated, truncated, info


# Run the environment
if __name__ == "__main__":
    env = CustomEnv(size=16)
    env.reset()
    env.render()