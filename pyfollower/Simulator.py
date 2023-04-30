from typing import List
from follower import Engine, Vector2


class FollowerEngine(Engine):
    def __init__(self, time_step: float=0.1 , neighbors_dist: float = 10.0,
                 agent_radius: float = 0.3, max_speed: float = 1.3):
        super().__init__()
        self.set_agent_defaults(neighbors_dist, 18, 5.0, 5.0, agent_radius,
                                max_speed, Vector2(0.0, 0.0))
        self.set_timestep(time_step)
        self._agent_number = 0
        self.pref_velocity_correction(True)

    @property
    def agent_number(self):
        return self._agent_number

    @property
    def time(self):
        return self.get_global_time()

    def add_agent(self, x: float, y: float, gx: float, gy: float):
        self._agent_number += 1
        return super().add_agent(Vector2(x, y))

    def add_obstacles(self, vertices: List[tuple]):
        # left of the direction is obstacle
        _vertices = [Vector2(x, y) for x, y in vertices]
        return super().add_obstacles(_vertices)

    def process_obstacles(self):
        return super().process_obstacles()

    def set_timestep(self, dt: float):
        return super().set_timestep(dt)

    def set_agent_pref(self, agent_id: int, pv_x: float, pv_y: float):
        super().set_agent_pref(agent_id, Vector2(pv_x, pv_y))

    def get_agent_position(self, agent_id: int):
        return super().get_agent_position(agent_id)

    def get_agent_velocity(self, agent_id: int):
        return super().get_agent_velocity(agent_id)

    def get_agent_positions(self):
        return super().get_agent_positions()

    def get_agent_velocities(self):
        return super().get_agent_velocities()

    def follower_step(self):
        return super().follower_step()

    def pvc(self, correction: bool):
        return super().pref_velocity_correction(correction)
