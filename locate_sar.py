from utils.streamlit_operations.working_mode_layouts import *
from locate_modules.main_simulation import simulation_main
from locate_modules.main_download_data import download_main
from locate_modules.visual_analysis import plot_traj_mitja
from locate_modules.numeric_analysis import compute_skill_score
from utils.streamlit_operations.streamlit_utils import transform_single_or_cloud_info_tuple, \
    transform_region_time_info_tuple
from utils.locate_operations.export_pdf_dashboard import generate_pdf_from_results
import numpy as np

# import pandas as pd

st.set_page_config(page_title="Locate", page_icon="🌎", layout="wide", initial_sidebar_state="expanded")

# Side bar
st.sidebar.title("Locate")
working_mode = st.sidebar.radio("Choose your working mode:", ("Validation", "Operation"))

# Choose a working mode
if working_mode == "Validation":
    st.sidebar.write("""
    ## About

    **Validation** mode is used to compare an experimental trajectory with a simulation.

    Several metrics are measured to determine the accuracy of the prediction.

    This mode is used for development.
    """)
    particle_info, single_or_cloud_info, region_time_info = validation_mode_interface()

elif working_mode == "Operation":
    st.sidebar.write("""
    ## About

    **Operation** mode is used to predict a trajectory based on some initial conditions.

    This is the main working mode.
    """)
    particle_info, single_or_cloud_info, region_time_info = operation_mode_interface()

# st.write(f"{particle_info}")
# st.write(f"{single_or_cloud_info}")
# st.write(f"{region_time_info}")

lat_start, lon_start, datetime_start, file_selected = particle_info
particle_set_type, custom_amount_of_particles, custom_radius_from_origin = transform_single_or_cloud_info_tuple(
    single_or_cloud_info)
region_type, region_info, simulation_length, use_waves = transform_region_time_info_tuple(region_time_info, (lat_start, lon_start))

# 3-step execution
st.write("""
    ___
    ### Simulate
    """)
col_download, col_simulate, col_analyze = st.columns(3)

# 1. Download
with col_download:
    st.write("Click to download your specified region data")
    # download_disable_button = False
    download_is_pressed = st.button("Download", disabled=(region_type == "I already have the data"))

    if download_is_pressed:
        st.write(f"Download **started** at {datetime.datetime.today().strftime('%Y/%m/%d, %H:%M:%S')}")
        download_main(date_start=datetime_start,
                      simulation_length=simulation_length,
                      CMEMS_download_limits=[38.1, 42.75, 0, 4.25])
        st.write(f"Download **terminated** at {datetime.datetime.today().strftime('%Y/%m/%d, %H:%M:%S')}")

# 2. Simulate
with col_simulate:
    st.write("Click to simulate your particle")
    if particle_info[0] is None:
        run_disable_button = True
        st.markdown("Make sure the particle is inside the IBI domain.")
    else:
        run_disable_button = not ((particle_info[0] > 30) and (particle_info[1] > 0))  # inside IBI region
    simulate_is_pressed = st.button("Simulate", disabled=run_disable_button)

    if simulate_is_pressed:
        st.write(f"Simulation **started** at {datetime.datetime.today().strftime('%Y/%m/%d, %H:%M:%S')}")
        simulation_main(cloud_of_particles=particle_set_type,
                        start_coordinates=[lat_start, lon_start],
                        start_datetime=np.datetime64(datetime_start),
                        simulation_days=simulation_length,
                        use_waves=use_waves, # link this to a variable
                        # data_base_path=cfg.data_base_path,
                        radius_from_origin=custom_radius_from_origin,
                        amount_of_particles=custom_amount_of_particles)
        st.write(f"Simulation **terminated** at {datetime.datetime.today().strftime('%Y/%m/%d, %H:%M:%S')}")

# 3. Analyze
with col_analyze:
    st.write("Click to analyze the results.")
    analyze_disable_button = False
    analyze_is_pressed = st.button("Analyze", disabled=analyze_disable_button)

    if analyze_is_pressed:
        fig1 = plot_traj_mitja(simulation_file_uri="file_logging/simulation_output_file/HarbourParticles.nc",
                                     experimental_track_file=file_selected)

        if working_mode == "Validation":
            fig3 = compute_skill_score(simulation_file="file_logging/simulation_output_file/HarbourParticles.nc",
                                       experimental_file=file_selected)

    # Used for development. Won't be in production
    # dashboard_is_pressed = st.button("Genarate dashboard", disabled=analyze_disable_button)
    # if dashboard_is_pressed:
    #     generate_pdf_from_results(simulation_file_uri="file_logging/simulation_output_file/HarbourParticles.nc",
    #                               experimental_track_file=file_selected,
    #                               use_waves=use_waves,
    #                               aa=12,
    #                               bb=13)



if analyze_is_pressed:
    st.pyplot(fig1)
    # col_fig1, col_fig2 = st.columns(2)
    # with col_fig1:
    #     st.pyplot(fig1)
    # with col_fig2:
    #     st.pyplot(fig2)

    if working_mode == "Validation":
        st.write("Skill score results:")
        st.pyplot(fig3)
