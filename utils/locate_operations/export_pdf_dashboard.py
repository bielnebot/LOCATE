import matplotlib.pyplot as plt

import xarray
import datetime
from utils.locate_operations.locate_utils import read_csv_file

# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature # pels features


import numpy as np
from utils.locate_operations.locate_skill_score_utils import skill_score, retrieve_experimental_trajectory, retrieve_simulated_trajectory, interpolate_point, plot_skill_score_dashboard


def generate_pdf_from_results(simulation_file_uri,  # or just the name
                    experimental_track_file, use_waves, aa, bb):
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
    ll_labels = [None for i in range(amount_particles)]
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
    plt.rcParams.update({'font.family': 'CMU Serif'})
    fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(aa, bb), subplot_kw={'projection': ccrs.PlateCarree()})
    # Fisrt subplot
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0:2, 0]:
        ax.remove()
    axbig_1 = fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree())

    # Second subplot
    gs = axs[0, 1].get_gridspec()
    for ax in axs[0:2, 1]:
        ax.remove()
    axbig_2 = fig.add_subplot(gs[0:2, 1], projection=ccrs.PlateCarree())

    ax_first_row = (axbig_1, axbig_2)

    # Third subplot
    gs = axs[2, 0].get_gridspec()
    for ax in axs[2, :]:
        ax.remove()
    axbig_3 = fig.add_subplot(gs[2, :])

    # Fourth subplot
    gs = axs[3, 0].get_gridspec()
    for ax in axs[3, :]:
        ax.remove()
    axbig_4 = fig.add_subplot(gs[3, :])

    ax_second_row = (axbig_3, axbig_4)

    #     fig, (ax_first_row, ax_second_row) = plt.subplots(2,2,
    #                                     figsize = (19,8),
    #                                     subplot_kw={'projection': ccrs.PlateCarree()})

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
    # ax_first_row[0].text((boundBox[0] + boundBox[1]) / 2, boundBox[2] * 1.001,
    #                      f"{amount_particles} particles simulated", horizontalalignment='center',
    #                      bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
    ax_first_row[0].text((boundBox[0] + boundBox[1]) / 2, boundBox[3] * 0.999,
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
    if use_waves:
        cbar_ax = fig.add_axes([0.90, 0.55, 0.02, 0.3])
    else:
        cbar_ax = fig.add_axes([0.94, 0.55, 0.02, 0.3])
    # new ax with dimensions of the colorbar

    # cbar = fig.colorbar(c, cax=cbar_ax)
    # barra = plt.colorbar(time_plot, ax=ax_first_row[1], location='right', shrink=0.8, pad=0.20)
    barra = plt.colorbar(time_plot, cax=cbar_ax)
    barra.ax.set_title('Elapsed hours')


    # Skill score
    real_trajectory = retrieve_experimental_trajectory(experimental_track_file)
    simulated_trajectory = retrieve_simulated_trajectory(simulation_file_uri)

    # print(simulated_trajectory)
    # print(simulated_trajectory[0, 2],type(simulated_trajectory[0, 2]))
    # Set time reference when particle is created
    real_trajectory[:, 2] = real_trajectory[:, 2] - real_trajectory[0, 2]
    simulated_trajectory[:, 2] = simulated_trajectory[:, 2] - simulated_trajectory[0, 2]

    # Convert timedelta to seconds
    real_trajectory[:, 2] = np.apply_along_axis(lambda x: x[2].total_seconds(), 1, real_trajectory)
    simulated_trajectory[:, 2] = np.apply_along_axis(lambda x: x[2].total_seconds(), 1, simulated_trajectory)

    # Make the simulated trajectory data-points match the experimental trajectory (time-wise)
    max_time_spent = min(max(real_trajectory[:, 2]), max(simulated_trajectory[:, 2]))
    new_trajectory = np.array(
        [interpolate_point(t, simulated_trajectory) for t in real_trajectory[:, 2] if t < max_time_spent])

    list_time, list_d, list_l_obs, ll_skill_score = skill_score(real_trajectory, new_trajectory)

    ss_array,time_array,dist_travelled_array,dist_separation_array = ll_skill_score, np.array(list_time), list_l_obs, list_d
    # fig, ax = plt.subplots(2, 1, figsize=(12, 9))

    # Skill score plot
    ax_second_row[0].plot(time_array / 3600, ss_array, linewidth=3, c="b")
    ax_second_row[0].grid()
    ax_second_row[0].set_ylabel("Skill score", fontsize=12)
    ax_second_row[0].tick_params(axis='both', which='major', labelsize=12)
    ax_second_row[1].plot(time_array / 3600, dist_travelled_array, linewidth=3, c="g", label="Experimental distance travelled")
    ax_second_row[1].plot(time_array / 3600, dist_separation_array, linewidth=3, c="r", label="Separation between trajectories")
    ax_second_row[1].grid()
    ax_second_row[1].set_xlabel("Elapsed time [h]", fontsize=12)
    ax_second_row[1].set_ylabel("Distance [km]", fontsize=12)
    ax_second_row[1].legend(loc="upper left",fontsize=12)
    ax_second_row[1].tick_params(axis='both', which='major', labelsize=12)

    y_max = [ax_second_row[0].get_ylim(), ax_second_row[1].get_ylim()]  # used to place vertical lines labels

    # Plot vertical lines
    for sub_plot in range(2):
        for elem in [6, 12, 24, 48, 72]:
            if elem < max(time_array / 3600):
                ax_second_row[sub_plot].axvline(x=elem, linestyle="--", c="k", linewidth=2)
                ax_second_row[sub_plot].text(elem, y_max[sub_plot][1], f"{elem} h", fontsize=12,
                                  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
                if sub_plot == 0:
                    print(f"ss at t = {elem} = {np.interp(elem, time_array / 3600, ss_array)}")

    # fig.tight_layout()

    plot_data = dict(
        file_name="FINDMESPOT-SLDMB030_BARCELONA_3.csv",
        waves="were" if use_waves else "were not",
    )

    plt.suptitle(plot_data["file_name"],size=30)
    plt.text(0.02, 0.92, f"Waves {plot_data['waves']} used", fontsize=12, transform=plt.gcf().transFigure,bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8))
    # plt.savefig(f"{plot_data['file_name'][:-4]}_waves_{plot_data['waves']}_used.png",dpi=250)

    print("pdf generated!")
    return fig