import pandas as pd
import numpy as np
import datetime
import xarray
import matplotlib.pyplot as plt
import os


def initial_coords_streamlit_read(file_object):
    # input: streamlit file object
    # output: first lat, lon and date coordinates of the file

    df = pd.read_csv(file_object, header=None)

    dataInici, latInici, lonInici = df.iloc[0]
    dayOfYear, hourOfDay = dataInici.split(" ")
    Year, Month, Day = dayOfYear.split("-")
    Hour, Minute, Second = hourOfDay.split(":")

    datetimeInici = datetime.datetime(year=int(Year), month=int(Month), day=int(Day), hour=int(Hour), minute=int(Minute),
                 second=int(Second))
    dateInici = datetime.date(year=int(Year), month=int(Month), day=int(Day))

    return latInici, lonInici, datetimeInici, dateInici


def default_rectangle_from_point(initial_coord,margin):
    lat, lon = initial_coord
    return lat-margin, lat+margin, lon-margin, lon+margin


def transform_single_or_cloud_info_tuple(triplet):
    particle_set_type, amount_of_particles, radius_from_origin = triplet
    if particle_set_type == "... a single particle":
        return False,amount_of_particles, radius_from_origin
    else:
        return True, amount_of_particles, radius_from_origin


def transform_region_time_info_tuple(tuple_var,initial_coord):
    region_type, region_info, simulation_length, use_waves = tuple_var
    if region_type == "Custom region" or "I already have the data":
        return tuple_var
    elif region_type == "Default region":
        return region_type, default_rectangle_from_point(initial_coord, 1), simulation_length, use_waves


def plot_downloaded_data(download_directory): # f"{cfg.data_base_path}/currents/IBI/"
    files_in_folder = os.listdir(download_directory)

    if len(files_in_folder) == 0:
        return False, None, None, None

    file_data = xarray.open_dataset(download_directory+files_in_folder[0])

    fig = plt.figure(figsize=(7,7))

    num_frame = 0
    df = pd.DataFrame(file_data.uo[num_frame], columns=file_data["longitude"], index=file_data["latitude"])
    data_matrix = df.to_numpy()
    plt.imshow(np.flipud(data_matrix))

    plt.yticks(np.linspace(0, len(df.index), 2), [round(df.index[0], 1), round(df.index[-1], 1)])
    plt.xticks(np.linspace(0, len(df.columns), 2), [round(df.columns[0], 1), round(df.columns[-1], 1)])
    return True, fig, file_data.indexes["time"][0].strftime('%Y-%m-%d %H:%M:%S'), file_data.indexes["time"][-1].strftime('%Y-%m-%d %H:%M:%S')
