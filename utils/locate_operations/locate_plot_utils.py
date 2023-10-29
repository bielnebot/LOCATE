import pandas as pd
import matplotlib.pyplot as plt

# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature  # pels features
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plotLlistaDeTrajectories(ll_longituds, ll_latituds, ll_labels, longMin, longMax, latMin, latMax):
    fig = plt.figure(figsize=(10, 10))
    mapa = fig.add_subplot(projection=ccrs.PlateCarree())

    mapa.set_extent([longMin, longMax, latMin, latMax], ccrs.PlateCarree())

    mapa.add_feature(cfeature.RIVERS)
    mapa.add_feature(cfeature.LAND)
    mapa.add_feature(cfeature.OCEAN)
    mapa.add_feature(cfeature.COASTLINE)
    mapa.add_feature(cfeature.BORDERS, linestyle=":")

    grid_lines = mapa.gridlines(draw_labels=True)

    for nomLabel, longitud, latitud in zip(ll_labels, ll_longituds, ll_latituds):
        plt.plot(longitud, latitud, linestyle='--', transform=ccrs.PlateCarree(),
                 marker='o', markersize=7, alpha=0.9, label=nomLabel)

    mapa.legend()


def boundingBoxOfTrack(lon, lat, margin):
    # input: lon and lat coordinates of a track.
    # output: the bounding box that fits the trajectory
    lonMax = max(lon)
    lonMin = min(lon)
    latMax = max(lat)
    latMin = min(lat)
    return lonMin - margin, lonMax + margin, latMin - margin, latMax + margin


# def boundingBoxOfMultipleTracks(listLon,listLat,margin):
#     lonMax_list, lonMin_list, latMax_list, latMin_list = [], [], [], []
#     for lonCoords,latCoords in zip(listLon,listLat):
#         boundBox = boundingBoxOfTrack(lonCoords,latCoords,margin)
#         lonMax_list.append(boundBox[1])
#         lonMin_list.append(boundBox[0])
#         latMax_list.append(boundBox[3])
#         latMin_list.append(boundBox[2])
#     return min(lonMin_list), max(lonMax_list), min(latMin_list), max(latMax_list)


