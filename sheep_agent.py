"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""
from math import atan2

import pygame
import numpy as np
import support


class Sheep_Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, window_pad, target_x, target_y, target_size):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Interaction strength
        # Attraction
        self.s_att = 0.02
        # Repulsion
        self.s_rep = 5
        # Alignment
        self.s_alg = 8

        # Interaction ranges (Zones)
        # Attraction
        self.steepness_att = -0.5
        self.r_att = 250
        # Repulsion
        self.steepness_rep = -0.5
        self.r_rep = 50
        # Alignment
        self.steepness_alg = -0.5
        self.r_alg = 150

        # Noise
        self.noise_sig = 0.1

        self.dt = 0.05

        # Boundary conditions
        # bounce_back: agents bouncing back from walls as particles
        # periodic: agents continue moving in both x and y direction and teleported to other side
        self.boundary = "bounce_back"

        self.id = id
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.orig_color = color
        self.color = self.orig_color
        self.selected_color = support.LIGHT_BLUE
        self.show_stats = True
        self.change_color_with_orientation = False

        # Non-initialisable private attributes
        self.velocity = 1  # agent absolute velocity
        self.v_max = 1  # maximum velocity of agent

        # Interaction
        self.is_moved_with_cursor = 0

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Initial Visualization of agent
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(support.BACKGROUND)
        self.image.set_colorkey(support.BACKGROUND)
        pygame.draw.circle(
            self.image, color, (radius, radius), radius
        )

        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, support.BACKGROUND, (radius, radius),
                         ((1 + np.cos(self.orientation)) * radius, (1 - np.sin(self.orientation)) * radius), 3)
        self.rect = self.image.get_rect()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        self.mask = pygame.mask.from_surface(self.image)

        #############################################
        self.v0 = 0.5
        self.vt = self.v0  # when tick = 0, vt = v0;
        self.v_upper = 2
        self.target_x = target_x
        self.target_y = target_y
        self.target_size = target_size
        self.state = "moving"
        self.f_x = 0.0
        self.f_y = 0.0
        self.v_dot = 0.0
        self.w_dot = 0.0

        ##### sheep interaction parameters#####
        self.rep_distance = 20
        self.att_distance = 50
        self.K_repulsion = 3  #25   #2 #15
        self.K_attraction = 8.0  #0.1 #8  # 0.8 #8
        self.K_shepherd = 15  #5 #2.5 #1.5  #0.6 #18    #1.5 #12
        self.K_Dr = 0.01  # noise_strength
        self.tick_time = 0.01  #0.01  # tick_time
        self.max_turning_angle = np.pi * 1 / 3
        self.f_avoid_x = 0.0
        self.f_avoid_y = 0.0
        self.f_att_x = 0.0
        self.f_att_y = 0.0
        self.num_rep = 0

        #####shepherd relative parameters#########
        self.safe_distance = 200
        self.f_shepherd_force_x = 0.0
        self.f_shepherd_force_y = 0.0

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation += 0.1
            if right_state:
                self.orientation -= 0.1
            self.prove_orientation()
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0

    #################################################
    def Get_repulsion_force(self, agents):
        r_x = 0.0
        r_y = 0.0
        self.f_avoid_x = 0.0
        self.f_avoid_y = 0.0
        self.num_rep = 0
        x_i = self.position[0]
        y_i = self.position[1]
        for agent in agents:
            if agent.id != self.id and agent.state == "moving":
                x_j = agent.position[0]
                y_j = agent.position[1]
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if distance <= self.rep_distance and distance != 0.0:
                    r_x = r_x + (x_i - x_j) / distance
                    r_y = r_y + (y_i - y_j) / distance
                    self.num_rep += 1

        self.f_avoid_x = r_x
        self.f_avoid_y = r_y

    def Get_attraction_force(self, agents):
        # need to improve preference to the front agents
        r_x = 0.0
        r_y = 0.0
        x_i = self.position[0]
        y_i = self.position[1]
        self.f_att_x = 0.0
        self.f_att_y = 0.0
        for agent in agents:
            if agent.id != self.id and agent.state == "moving":
                x_j = agent.position[0]
                y_j = agent.position[1]
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if self.att_distance >= distance >= self.rep_distance and distance != 0.0:
                    r_x = r_x + (x_j - x_i) / distance
                    r_y = r_y + (y_j - y_i) / distance

        self.f_att_x = r_x
        self.f_att_y = r_y

    def update_shepherd_forces(self, shepherd_agents):
        r_x = 0
        r_y = 0
        x_i = self.position[0]
        y_i = self.position[1]
        num_dangerous_shepherd = 0
        for shepherd in shepherd_agents:
            shepherd_x = shepherd.x
            shepherd_y = shepherd.y
            distance = np.sqrt((x_i - shepherd_x) ** 2 + (y_i - shepherd_y) ** 2)
            if distance <= self.safe_distance and distance != 0.0:
                num_dangerous_shepherd = num_dangerous_shepherd + 1
                r_x = r_x + (x_i - shepherd_x) / distance  # unit vector  ?? check distance == 0 ?
                r_y = r_y + (y_i - shepherd_y) / distance  # unit vector
        if num_dangerous_shepherd != 0:
            self.f_shepherd_force_x = r_x / num_dangerous_shepherd
            self.f_shepherd_force_y = r_y / num_dangerous_shepherd
        else:
            self.f_shepherd_force_x = 0.0
            self.f_shepherd_force_y = 0.0
        return

    def update_sheep_state(self, agents):
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius
        distance, angle = support.Get_relative_distance_angle(self.target_x,
                                                              self.target_y,
                                                              x,
                                                              y)
        if distance < self.target_size:
            self.state = "staying"
            self.color = support.LIGHT_BLUE
        else:
            self.state = "moving"
            self.color = support.GREEN

    def update(self, agents, shepherd_agents):  # this is actually sheep_agents;
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param sheep_agents:
        :param agents: a list of all other agents in the environment.
        """
        self.update_sheep_state(agents)

        self.reflect_from_walls(self.boundary)

        self.update_shepherd_forces(shepherd_agents)

        if self.state == "moving":
            self.Get_repulsion_force(agents)
            self.Get_attraction_force(agents)

            # self.f_x = self.f_avoid_x * self.K_repulsion + self.f_att_x * self.K_attraction + self.f_shepherd_force_x * self.K_shepherd
            # self.f_y = self.f_avoid_y * self.K_repulsion + self.f_att_y * self.K_attraction + self.f_shepherd_force_y * self.K_shepherd

            if self.num_rep != 0:
                # pass
                self.f_x = self.f_avoid_x * self.K_repulsion
                self.f_y = self.f_avoid_y * self.K_repulsion
            else:
                self.f_x = self.f_att_x * self.K_attraction + self.f_shepherd_force_x * self.K_shepherd
                self.f_y = self.f_att_y * self.K_attraction + self.f_shepherd_force_y * self.K_shepherd

            # self.f_x = self.f_att_x * self.K_attraction
            # self.f_y = self.f_att_y * self.K_attraction

            self.v_dot = self.f_x * np.cos(self.orientation) + self.f_y * np.sin(self.orientation)
            self.w_dot = -self.f_x * np.sin(self.orientation) + self.f_y * np.cos(self.orientation)

            if self.w_dot > self.max_turning_angle:
                self.w_dot = self.max_turning_angle
            if self.w_dot <= -self.max_turning_angle:
                self.w_dot = -self.max_turning_angle

            # Dr = np.random.normal(0, 1) * np.sqrt(2 * self.K_Dr) / (self.tick_time ** 0.5)
            # self.vt += self.v_dot * self.tick_time
            self.vt = self.v0 + self.v_dot * self.tick_time
            # # limit velocity
            if self.vt >= 0:
                self.vt = np.minimum(self.vt, self.v_upper)
            else:
                self.vt = -np.minimum(np.abs(self.vt), self.v_upper)

            self.position[0] += self.vt * np.cos(self.orientation)
            self.position[1] += self.vt * np.sin(self.orientation)
            self.orientation += self.w_dot * self.tick_time

            self.orientation = support.transform_angle(self.orientation)  # [-pi, pi]

            # self.orientation = support.reflect_angle(self.orientation)   # [0, 2pi]

        # if not self.is_moved_with_cursor and self.state == "moving":  # we freeze agents when we move them
        # # updating agent's state variables according to calculated vel and theta
        # self.orientation += self.dt * self.dtheta
        # self.prove_orientation()  # bounding orientation into 0 and 2pi
        # self.velocity += self.dt * self.dv
        # self.prove_velocity()  # possibly bounding velocity of agent
        #
        # # updating agent's position
        # self.position[0] += self.velocity * np.cos(self.orientation)
        # self.position[1] -= self.velocity * np.sin(self.orientation)

        # boundary conditions if applicable
        # self.reflect_from_walls(self.boundary)

        # updating agent visualization
        self.draw_update()

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        self.color = support.calculate_color(self.orientation, self.velocity)

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # change agent color according to mode
        # if self.change_color_with_orientation:
        #     self.change_color()
        # else:
        #     self.color = self.orig_color

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(support.BACKGROUND)
        self.image.set_colorkey(support.BACKGROUND)
        if self.is_moved_with_cursor:
            pygame.draw.circle(
                self.image, self.selected_color, (self.radius, self.radius), self.radius
            )
        else:
            pygame.draw.circle(
                self.image, self.color, (self.radius, self.radius), self.radius
            )

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, support.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 + np.sin(self.orientation)) * self.radius),
                         3)
        self.mask = pygame.mask.from_surface(self.image)

    def reflect_from_walls(self, boundary_condition):
        """reflecting agent from environment boundaries according to a desired x, y coordinate. If this is over any
        boundaries of the environment, the agents position and orientation will be changed such that the agent is
         reflected from these boundaries."""

        # Boundary conditions according to center of agent (simple)
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius
        self.orientation = support.reflect_angle(self.orientation)  # [0, 2pi]

        if boundary_condition == "bounce_back":
            # Reflection from left wall
            if x < self.boundaries_x[0]:
                self.position[0] = self.boundaries_x[0] - self.radius + 1

                if np.pi / 2 <= self.orientation < np.pi:
                    self.orientation -= np.pi / 2
                elif np.pi <= self.orientation <= 3 * np.pi / 2:
                    self.orientation += np.pi / 2

            # Reflection from right wall
            if x > self.boundaries_x[1]:

                self.position[0] = self.boundaries_x[1] - self.radius - 1

                if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                    self.orientation -= np.pi / 2
                elif 0 <= self.orientation <= np.pi / 2:
                    self.orientation += np.pi / 2

            # Reflection from upper wall
            if y < self.boundaries_y[0]:
                self.position[1] = self.boundaries_y[0] - self.radius + 1

                if np.pi < self.orientation <= np.pi * 3 / 2:
                    self.orientation -= np.pi / 2
                elif np.pi * 3 / 2 < self.orientation <= np.pi * 2:
                    self.orientation += np.pi / 2

            # Reflection from lower wall
            if y > self.boundaries_y[1]:
                self.position[1] = self.boundaries_y[1] - self.radius - 1
                if np.pi / 2 <= self.orientation <= np.pi:
                    self.orientation += np.pi / 2
                elif 0 <= self.orientation < np.pi / 2:
                    self.orientation -= np.pi / 2
            # self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.orientation = support.reflect_angle(self.orientation)

        elif boundary_condition == "periodic":

            if x < self.boundaries_x[0]:
                self.position[0] = self.boundaries_x[1] - self.radius
            elif x > self.boundaries_x[1]:
                self.position[0] = self.boundaries_x[0] + self.radius

            if y < self.boundaries_y[0]:
                self.position[1] = self.boundaries_y[1] - self.radius
            elif y > self.boundaries_y[1]:
                self.position[1] = self.boundaries_y[0] + self.radius

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    def prove_velocity(self):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if np.abs(self.velocity) > self.v_max:
            # stopping agent if too fast during exploration
            self.velocity = self.v_max
