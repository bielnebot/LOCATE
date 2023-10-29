# import datetime
import numpy as np
import xarray
import datetime
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from utils.locate_operations.locate_utils import read_csv_file


def retrieve_experimental_trajectory(experimental_file):
    lon, lat, date_time = experimental_file
    return np.array([lat,lon,date_time]).transpose()


def retrieve_simulated_trajectory(simulation_file):
    fitxer = xarray.open_dataset(simulation_file)

    coordLonMitja = fitxer.lon.mean("traj").values
    coordLatMitja = fitxer.lat.mean("traj").values
    coordTime = fitxer.time[0].values
    coordTime = pd.to_datetime(coordTime).to_pydatetime() # np.datetime64 to datetime.datetime
    return np.array([coordLatMitja, coordLonMitja, coordTime]).transpose()


def pair_of_points_between_time_step(trajectory_matrix, instant):
    """
    Returns the two points before and after a certain instant
    """
    for i in range(trajectory_matrix.shape[0]):
        lat, lon, time_spent = trajectory_matrix[i,:]
        if time_spent > instant:
            return (trajectory_matrix[i-1,0],trajectory_matrix[i-1,1], trajectory_matrix[i-1,2]), (lat, lon, time_spent)


def distance_between_two_points(p1,p2):
    # Euclidean distance
    return np.linalg.norm(p2-p1)


def interpolation_3d(point_of_origin, desired_distance_from_origin, direction_vector):
    """
    point_of_origin is np.array
    desired_distance_from_origin is float
    direction_vector is np.array
    """
    return point_of_origin + direction_vector * desired_distance_from_origin / np.linalg.norm(direction_vector)


def interpolate_point(desired_time_spent, trajectory_matrix):
    p1,p2 = pair_of_points_between_time_step(trajectory_matrix,desired_time_spent)
    lat_1, lon_1, time_spent_1 = p1
    lat_2, lon_2, time_spent_2 = p2
    p1 = np.array([lat_1, lon_1])
    p2 = np.array([lat_2, lon_2])

    desired_time_delta = desired_time_spent - time_spent_1

    # Relative line
    space_dist = distance_between_two_points(p1,p2)
    time_dist = time_spent_2 - time_spent_1

    desired_distance = desired_time_delta * space_dist / time_dist

    direction_vector = p2-p1
    new_point = interpolation_3d(p1, desired_distance, direction_vector)

    return np.array([new_point[0], new_point[1], desired_time_spent])


def haversine_skill_score(p1, p2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def skill_score(observed_trajectory,modeled_trajectory):

    ll_skill_score = []

    list_d = []
    list_dl = []

    for i in range(1, modeled_trajectory.shape[0]):
        point_1 = observed_trajectory[i, [0, 1]]
        previous_point_1 = observed_trajectory[i - 1, [0, 1]]
        point_2 = modeled_trajectory[i, [0, 1]]
        d_i = haversine_skill_score(point_1, point_2)
        dl_i = haversine_skill_score(point_1, previous_point_1)

        list_d.append(d_i)
        list_dl.append(dl_i)

        list_l_obs = np.cumsum(list_dl)

        s = sum(list_d) / sum(list_l_obs)

        n = 1  # a threshold
        ss = 1 - s / n if s <= n else 0

        ll_skill_score.append(ss)

    list_time = list(modeled_trajectory[:,2])[1:] # drop first element
    return list_time, list_d, list_l_obs, ll_skill_score


def plot_skill_score_dashboard(ss_array,time_array,dist_travelled_array,dist_separation_array):
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))

    # Skill score plot
    ax[0].plot(time_array / 3600, ss_array, linewidth=3, c="b")
    ax[0].grid()
    ax[0].set_ylabel("Skill score")

    # Separation distance and travelled distance
    ax[1].plot(time_array / 3600, dist_travelled_array, linewidth=3, c="g", label="Experimental distance travelled")
    ax[1].plot(time_array / 3600, dist_separation_array, linewidth=3, c="r", label="Separation between trajectories")
    ax[1].grid()
    ax[1].set_xlabel("Elapsed time [h]")
    ax[1].set_ylabel("Distance [km]")
    ax[1].legend(loc="upper left")

    y_max = [ax[0].get_ylim(), ax[1].get_ylim()] # used to place vertical lines labels

    # Plot vertical lines
    for sub_plot in range(2):
        for elem in [6,12,24,48,72]:
            if elem < max(time_array/3600):
                ax[sub_plot].axvline(x=elem,linestyle="--",c="k",linewidth=2)
                ax[sub_plot].text(elem,y_max[sub_plot][1],f"{elem} h",bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
    # plt.show()
    return fig
