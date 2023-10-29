import pandas as pd
import numpy as np
import random
import math
import datetime
import os

# def lat_lon_date(tracks_directory):
#     # input: directory where the .csv files are stored (str)
#     # output: list of the first lat, lon and date o each file

#     tracks_directory = tracks_directory if tracks_directory[-1] == "/" else tracks_directory + "/"

#     llistaFitxers = os.listdir(tracks_directory)

#     llistaData = []
#     llistaLat = []
#     llistaLon = []

#     for fitxer in llistaFitxers:
#         ruta = tracks_directory + fitxer
#         df = pd.read_csv(ruta, header=None)

#         dataInici, latInici, lonInici = df.iloc[0]
#         dayOfYear, _ = dataInici.split(" ")
#         Year, Month, Day = dayOfYear.split("-")

#         # if Month != "02":
#         #     continue

#         llistaData.append(np.datetime64(datetime(year=int(Year), month=int(Month), day=int(Day))))
#         llistaLat.append(latInici)
#         llistaLon.append(lonInici)

#     return llistaLat, llistaLon, llistaData

# def initial_coords_streamlit_read(file_object):
#     # input: streamlit file object
#     # output: first lat, lon and date coordinates of the file
#
#     df = pd.read_csv(file_object, header=None)
#
#     dataInici, latInici, lonInici = df.iloc[0]
#     dayOfYear, hourOfDay = dataInici.split(" ")
#     Year, Month, Day = dayOfYear.split("-")
#     Hour, Minute, Second = hourOfDay.split(":")
#
#     datetimeInici = datetime.datetime(year=int(Year), month=int(Month), day=int(Day), hour=int(Hour), minute=int(Minute),
#                  second=int(Second))
#     dateInici = datetime.date(year=int(Year), month=int(Month), day=int(Day))
#
#     return latInici, lonInici, datetimeInici, dateInici


def read_csv_file(file_object):
    """
    file_object can also be the uri
    """
    df = pd.read_csv(file_object, header=None)
    lat_coords = list(df[1])
    lon_coords = list(df[2])
    time_coords = list(df[0])
    time_coords = [datetime.datetime.strptime(data, '%Y-%m-%d %H:%M:%S') for data in time_coords]
    return lon_coords, lat_coords, time_coords


def initialCoords_csv(tracks_directory):
    # input: directory where the .csv file is stored (str)
    # output: first lat, lon and date coordinates of the file

    tracks_directory = tracks_directory if tracks_directory[-1] == "/" else tracks_directory + "/"

    llistaFitxers = os.listdir(tracks_directory)

    ruta = tracks_directory + llistaFitxers[0]
    df = pd.read_csv(ruta, header=None)

    dataInici, latInici, lonInici = df.iloc[0]
    dayOfYear, hourOfDay = dataInici.split(" ")
    Year, Month, Day = dayOfYear.split("-")
    Hour, Minute, Second = hourOfDay.split(":")

    datetimeInici = np.datetime64(
        datetime.datetime(year=int(Year), month=int(Month), day=int(Day), hour=int(Hour), minute=int(Minute),
                 second=int(Second)))

    return latInici, lonInici, datetimeInici


def point_gauss_2d(mu, sigma):
    # input: mu = (muX, muY); sigma = (sigmaX, sigmaY)
    # output: a point in the 2D plane

    muX, muY = mu
    sigmaX, sigmaY = sigma
    x = random.gauss(muX, sigmaX)
    y = random.gauss(muY, sigmaY)
    return (x, y)


def point_cloud_gauss_2d(mu, sigma, amount):
    # input: mu = (muX, muY); sigma = (sigmaX, sigmaY)
    # output: a cloud of points
    return np.array([point_gauss_2d(mu, sigma) for _ in range(amount)])


def distance_to_angle(dist):
    # dist in km
    earth_radius = 6371  # [km]
    angle = dist / earth_radius
    return angle * 180 / math.pi


def point_cloud_from_coordinate(point, radius, amount, date):
    # point = [point_lat,point_lon]
    # radius in km. radius = 1 sigma of distributin (approx 68,3 % of points)
    degreesSigma = distance_to_angle(radius)
    print("degreesSigma val:", degreesSigma)
    point_cloud = point_cloud_gauss_2d(point, [degreesSigma, degreesSigma], amount)

    llistaLat = point_cloud[:, 0]
    llistaLon = point_cloud[:, 1]

    if isinstance(date, list):
        llistaData = [np.datetime64(datetime.datetime(year=date[0], month=date[1], day=date[2])) for _ in range(amount)]
    elif isinstance(date, np.datetime64):
        llistaData = [date for _ in range(amount)]

    return llistaLat, llistaLon, llistaData
