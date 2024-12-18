Target_place_x = 800
Target_place_y = 800
Target_size = 200  # radius

Boundary_x = Target_place_x + Target_size
Boundary_y = Target_place_y + Target_size


TICK = 1
Iterations = 20000

N_sheep = 60
N_shepherd = 2
L3 = 50
Fps = 25
Robot_Loop = True
Show_Animation = False

from loop_function import Loop_Function

loop_function = Loop_Function(N_sheep=N_sheep,  # Number of agents in the environment
                              N_shepherd = N_shepherd,
                              Time=Iterations,  # Simulation timesteps
                              width=Boundary_x,  # Arena width in pixels
                              height=Boundary_y,  # Arena height in pixels
                              target_place_x = Target_place_x,
                              target_place_y = Target_place_y,
                              target_size = Target_size,
                              framerate=Fps,
                              window_pad=30,
                              with_visualization = True,
                              show_animation = Show_Animation,
                              agent_radius= 10,  # 10 Agent radius in pixels
                              L3 = L3,  # repulsion distance
                              robot_loop = Robot_Loop,
                              physical_obstacle_avoidance=False)

# we loop through all the agents of the created simulation
print("Setting parameters for agent", end = ' ')

loop_function.start()