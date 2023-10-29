# Matplotlib
import matplotlib.pyplot as plt

import xarray
import numpy as np
from datetime import datetime
from utils.locate_operations.locate_utils import read_csv_file

# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature # pels features


def plot_traj_mitja_v0(simulation_file_uri, # or just the name
                    experimental_track_file):

    fitxer = xarray.open_dataset(simulation_file_uri)

    nreTrajectories = len(fitxer.traj)
    # print(nreTrajectories,"trajectories in the file")

    coordLonMitja = fitxer.lon.mean("traj")
    coordLatMitja = fitxer.lat.mean("traj")
    coordTime = fitxer.time[0].values

    ll_longituds = fitxer.lon.values
    ll_latituds = fitxer.lat.values
    ll_labels = ["traj_"+str(i) for i in range(200)]

    margin = 0.22
    boundBox = (
                min(fitxer.lon.min("obs"))-margin,
                max(fitxer.lon.max("obs"))+margin,
                min(fitxer.lat.min("obs"))-margin,
                max(fitxer.lat.max("obs"))+margin,
    )


    fig = plt.figure(figsize = (9,9))
    mapa = fig.add_subplot(projection = ccrs.PlateCarree())

    mapa.set_extent([boundBox[0],boundBox[1],boundBox[2],boundBox[3]], ccrs.PlateCarree())
    mapa.add_feature(cfeature.RIVERS)
    mapa.add_feature(cfeature.LAND)
    mapa.add_feature(cfeature.OCEAN)
    mapa.add_feature(cfeature.COASTLINE)
    mapa.add_feature(cfeature.BORDERS, linestyle = ":")

    mapa.gridlines(draw_labels = True,zorder=-1)

    for nomLabel,longitud,latitud in zip(ll_labels,ll_longituds,ll_latituds):
        plt.plot(longitud,
                 latitud,
                 linestyle = '--',
                 c = "r",
                 transform=ccrs.PlateCarree(),
                 alpha = 0.2,
    #              label = nomLabel
                )
    # Trajectòria mitjana
    plt.plot(coordLonMitja,
             coordLatMitja,
             linestyle = '-',
             linewidth = 3,
             c = "k",
             transform=ccrs.PlateCarree(),
             alpha = 1,
             label = "Trajectòria mitjana"
                )
    # Trajectòria experimental
    if experimental_track_file is not None:
        EXP_lonCoords, EXP_latCoords, EXP_timeCoords = experimental_track_file

        EXP_timeCoords = [np.datetime64(data) for data in EXP_timeCoords]
        plt.plot(EXP_lonCoords,
                 EXP_latCoords,
                 linestyle = '-',
                 linewidth = 3,
                 c = "b",
                 transform=ccrs.PlateCarree(),
                 alpha = 1,
                 label = "Trajectòria experimental"
                    )

    mapa.legend()

    if experimental_track_file is not None:
        EXP_timeCoords = EXP_timeCoords - EXP_timeCoords[0]
        EXP_timeCoords = EXP_timeCoords.astype("float64")
        EXP_timeCoords = EXP_timeCoords * 1e-6 / (3600 * 24)  # estava en us
    else:
        EXP_timeCoords = []

    coordTime = coordTime - coordTime[0]
    coordTime = coordTime.astype("float64")
    coordTime = coordTime * 1e-9 / (3600 * 24)  # estava en ns

    fig2 = plt.figure(figsize=(12,12))
    mapa = fig2.add_subplot(projection=ccrs.PlateCarree())

    mapa.set_extent([boundBox[0], boundBox[1], boundBox[2], boundBox[3]], ccrs.PlateCarree())

    mapa.add_feature(cfeature.RIVERS)
    mapa.add_feature(cfeature.LAND)
    mapa.add_feature(cfeature.OCEAN)
    mapa.add_feature(cfeature.COASTLINE)
    mapa.add_feature(cfeature.BORDERS, linestyle=":")

    grid_lines = mapa.gridlines(draw_labels=True, zorder=-1)

    # Pel colorbar
    valorsData = np.concatenate([EXP_timeCoords, coordTime], axis=0)
    min_, max_ = valorsData.min(), valorsData.max()

    # Trajectòria mitjana
    plt.scatter(coordLonMitja,
                coordLatMitja,
                #          linestyle = '-',
                marker="x",
                #          linewidth = 3,
                #          c = "k",
                c=coordTime,
                transform=ccrs.PlateCarree(),
                alpha=1,
                cmap="jet",
                label="Trajectòria mitjana"
                )
    plt.clim(min_, max_)
    if experimental_track_file is not None:
        # Trajectòria experimental
        plt.scatter(EXP_lonCoords,
                    EXP_latCoords,
                    #          linestyle = '-',
                    marker=".",
                    #          linewidth = 3,
                    #          c = "b",
                    c=EXP_timeCoords,
                    transform=ccrs.PlateCarree(),
                    alpha=1,
                    cmap="jet",
                    label="Trajectòria experimental"
                    )
        plt.clim(min_, max_)

    mapa.legend()
    barra = plt.colorbar(pad=0.12)
    barra.set_label('Dies transcorreguts', rotation=90)


    return fig, fig2


def plot_traj_mitja(simulation_file_uri,  # or just the name
                    experimental_track_file):
    if experimental_track_file is not None:
        ################################
        #### Open experimental file ####
        ################################
        # experimental_track_file = read_csv_file(experimental_track_file)

        # For subplot 1
        EXP_lonCoords, EXP_latCoords, EXP_timeCoords = experimental_track_file
        EXP_timeCoords = [np.datetime64(data) for data in EXP_timeCoords]

        # For subplot 2
        EXP_timeCoords = EXP_timeCoords - EXP_timeCoords[0]
        EXP_timeCoords = EXP_timeCoords.astype("float64")
        EXP_timeCoords = EXP_timeCoords * 1e-6 / (3600)  # estava en us
    else:
        EXP_lonCoords = []
        EXP_latCoords = []
        EXP_timeCoords = []

    #####################################
    #### Open simulation output file ####
    #####################################
    fitxer = xarray.open_dataset(simulation_file_uri)

    amount_particles = len(fitxer.traj)

    # Extract data
    coordLonMitja = fitxer.lon.mean("traj")
    coordLatMitja = fitxer.lat.mean("traj")
    coordTime = fitxer.time[0].values

    # Extract domain
    ll_longituds = fitxer.lon.values
    ll_latituds = fitxer.lat.values
    ll_labels = [None for _ in range(amount_particles)]
    ll_labels[0] = "Simulated particles"  # for the legend

    coordTime = coordTime - coordTime[0]
    coordTime = coordTime.astype("float64")
    coordTime = coordTime * 1e-9 / (3600)  # estava en ns

    # Plotting bounding box
    margin = 0.22
    boundBox = (
        min(fitxer.lon.min("obs")) - margin,
        max(fitxer.lon.max("obs")) + margin,
        min(fitxer.lat.min("obs")) - margin,
        max(fitxer.lat.max("obs")) + margin,
    )

    # Create figure
    fig, ax_first_row = plt.subplots(1, 2,
                                     figsize=(13, 8),
                                     subplot_kw={'projection': ccrs.PlateCarree()})

    # Configure cartopy on both subplots
    for ax_range in [0, 1]:
        ax_first_row[ax_range].set_extent([boundBox[0], boundBox[1], boundBox[2], boundBox[3]], ccrs.PlateCarree())
        ax_first_row[ax_range].add_feature(cfeature.RIVERS)
        ax_first_row[ax_range].add_feature(cfeature.LAND)
        ax_first_row[ax_range].add_feature(cfeature.OCEAN)
        ax_first_row[ax_range].add_feature(cfeature.COASTLINE)
        ax_first_row[ax_range].add_feature(cfeature.BORDERS, linestyle=":")
        map_grid = ax_first_row[ax_range].gridlines(draw_labels=True, zorder=-1)
        map_grid.top_labels = False
        map_grid.right_labels = False
    ax_first_row[0].text((boundBox[0] + boundBox[1]) / 2, boundBox[2] * 1.001,
                         f"{amount_particles} particles simulated", horizontalalignment='center',
                         bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))

    # Subplot 1

    # Simulated trajectories
    for nomLabel, longitud, latitud in zip(ll_labels, ll_longituds, ll_latituds):
        ax_first_row[0].plot(longitud,
                             latitud,
                             linestyle='--',
                             c="r",
                             transform=ccrs.PlateCarree(),
                             alpha=0.2,
                             label=nomLabel
                             )
    # Average trajectory
    ax_first_row[0].plot(coordLonMitja,
                         coordLatMitja,
                         linestyle='-',
                         linewidth=3,
                         c="k",
                         transform=ccrs.PlateCarree(),
                         alpha=1,
                         label="Average trajectory"
                         )

    # Trajectòria experimental
    if experimental_track_file is not None:
        ax_first_row[0].plot(EXP_lonCoords,
                             EXP_latCoords,
                             linestyle='-',
                             linewidth=3,
                             c="b",
                             transform=ccrs.PlateCarree(),
                             alpha=1,
                             label="Experimental trajectory"
                             )
    ax_first_row[0].legend()

    # Subplot 2
    fused_lon = list(coordLonMitja.values) + list(EXP_lonCoords)
    fused_lat = list(coordLatMitja.values) + list(EXP_latCoords)
    fused_time = list(coordTime) + list(EXP_timeCoords)

    exp_size = 10
    sim_size = 100

    fused_size = [sim_size for _ in range(len(coordTime))] + [exp_size for _ in range(len(EXP_timeCoords))]

    # Experimental + simulated trajectory
    if experimental_track_file is not None:
        time_plot = ax_first_row[1].scatter(fused_lon,
                                            fused_lat,
                                            #                                             linestyle = '-',
                                            marker=".",
                                            #                                             linewidth = 3,
                                            #                                             c = "b",
                                            c=fused_time,
                                            s=fused_size,
                                            transform=ccrs.PlateCarree(),
                                            alpha=1,
                                            cmap="jet",
                                            )
        ax_first_row[1].scatter([], [], marker=".", c="b", alpha=1, s=sim_size, label="Simulated trajectory")
        ax_first_row[1].scatter([], [], marker=".", c="b", alpha=1, s=exp_size, label="Experimental trajectory")

    # Only simulated trajectory
    else:
        time_plot = ax_first_row[1].scatter(coordLonMitja,
                                            coordLatMitja,
                                            #                                         linestyle = '-',
                                            marker=".",
                                            s=sim_size,
                                            #                                         linewidth = 3,
                                            #                                         c = "k",
                                            c=coordTime,
                                            transform=ccrs.PlateCarree(),
                                            alpha=1,
                                            cmap="jet",
                                            label="Average trajectory"
                                            )
    ax_first_row[1].legend()

    # Colorbar
    cbar_ax = fig.add_axes([0.985,0.24, 0.026, 0.5])
    # new ax with dimensions of the colorbar

    # cbar = fig.colorbar(c, cax=cbar_ax)
    # barra = plt.colorbar(time_plot, ax=ax_first_row[1], location='right', shrink=0.8, pad=0.20)
    barra = plt.colorbar(time_plot, cax=cbar_ax)
    barra.ax.set_title('Elapsed hours')
    #     plt.savefig("holaaaaa.svg")
    # plt.tight_layout()

    return fig






if __name__ == "__main__":
    fig = plot_traj_mitja(simulation_file_uri="HarbourParticles.nc",
                          experimental_track_file="C:/modelsPython/locateBiel/BuoyTracks/COSMOE2_0-2574900-GMT.csv")
    plt.show()