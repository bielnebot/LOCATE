import numpy as np
from utils.locate_operations.locate_skill_score_utils import skill_score, retrieve_experimental_trajectory, retrieve_simulated_trajectory, interpolate_point, plot_skill_score_dashboard


def compute_skill_score(experimental_file,simulation_file):
    real_trajectory = retrieve_experimental_trajectory(experimental_file)
    simulated_trajectory = retrieve_simulated_trajectory(simulation_file)
    # print(simulated_trajectory)
    # print(simulated_trajectory[0, 2],type(simulated_trajectory[0, 2]))
    # Set time reference when particle is created
    real_trajectory[:, 2] = real_trajectory[:, 2] - real_trajectory[0, 2]
    simulated_trajectory[:, 2] = simulated_trajectory[:, 2] - simulated_trajectory[0, 2]

    # Convert timedelta to seconds
    real_trajectory[:, 2] = np.apply_along_axis(lambda x: x[2].total_seconds(), 1, real_trajectory)
    simulated_trajectory[:, 2] = np.apply_along_axis(lambda x: x[2].total_seconds(), 1, simulated_trajectory)

    # Make the simulated trajectory data-points match the experimental trajectory (time-wise)
    max_time_spent = min(max(real_trajectory[:, 2]),max(simulated_trajectory[:, 2]))
    new_trajectory = np.array([interpolate_point(t, simulated_trajectory) for t in real_trajectory[:, 2] if t < max_time_spent])

    list_time, list_d, list_l_obs, ll_skill_score = skill_score(real_trajectory,new_trajectory)

    fig = plot_skill_score_dashboard(ll_skill_score,np.array(list_time),list_l_obs,list_d)

    return fig
