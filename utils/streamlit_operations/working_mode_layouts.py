import streamlit as st
import datetime
from utils.streamlit_operations.streamlit_utils import initial_coords_streamlit_read, plot_downloaded_data
from utils.locate_operations import UPC_config as cfg
from utils.locate_operations.locate_utils import read_csv_file


def single_or_cloud_of_particles():
    amount_of_particles, radius_from_origin = None, None
    particle_set_type = st.selectbox("Simulate...",
                                     options=["... a single particle", "... a cloud of particles"],
                                     index=0  # preselected option
                                     )
    if particle_set_type == "... a cloud of particles":
        amount_of_particles = st.number_input("Amount of particles to simulate", min_value=2, max_value=1000)
        radius_from_origin = st.number_input("Radius of the particle cloud [km]", step=50e-3, format="%.3f",
                                             min_value=50e-3, max_value=20.0)
    return particle_set_type, amount_of_particles, radius_from_origin


def simulation_region_and_time(column1, column2, date_start, enable_time_box):
    with column1:
        st.markdown("### Simulation region")
        region_info = None
        region_type = st.selectbox("Choose the region to simulate:",
                                   ["Default region", "Custom region", "I already have the data"])
        if region_type == "Default region":
            st.markdown("""
                    `Default region` simulates a rectangular region around the initial particle coordinates.

                    Make sure you have internet connection as data will be downloaded.
                    """)
        elif region_type == "Custom region":
            st.markdown("""
                    `Custom region` simulates a custom rectangular region. Fill the following boxes to define it.

                    Make sure you have internet connection as data will be downloaded.
                    """)
        elif region_type == "I already have the data":
            st.markdown(
                "`I already have the data` simulates the region you have in `c:/.../locate/your-CMEMS-data-directory/...`")
            # region_info = st.text_input("Directory with CMEMS data:", value="Data_proves_baixar")
            region_info = None
            files_in_directory, fig_data, data_date_starts, data_date_ends = plot_downloaded_data(f"{cfg.data_base_path}/currents/IBI/")
            if files_in_directory:
                st.write(f"The data in `{cfg.data_base_path}` starts on the {data_date_starts} and end on the {data_date_ends}. The region is the following:")
                st.pyplot(fig_data)
            else:
                st.write(f"No files in {cfg.data_base_path}")

    with column2:
        st.markdown("### Simulation time")
        simulation_length = None
        if enable_time_box:
            max_simulation_time = datetime.date.today() - date_start + datetime.timedelta(days=3)
            simulation_length = st.number_input("Number of days to simulate", step=1, min_value=1,
                                                max_value=max_simulation_time.days)
        else:
            st.write("First load a track.")

        st.markdown("### Simulation data")
        select_use_waves = st.selectbox("Choose the data to use:",
                                        options=["Only currents", "Currents and waves"],
                                        index=0  # preselected option
                                        )
        use_waves = True if select_use_waves == "Currents and waves" else False


    if region_type == "Custom region":
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Latitude coordinates**")
            lat_min = st.number_input("Min latitude", step=1e-2, format="%.2f", min_value=-90.0, max_value=90.0)
            lat_max = st.number_input("Max latitude", step=1e-2, format="%.2f", min_value=-90.0, max_value=90.0)
            lat_is_ok = (lat_min < lat_max)
            lat_message = "Latitude **correct**" if lat_is_ok else "Write latitude coordinates"
            st.write(f"{lat_message}")
        with col_b:
            st.markdown("**Longitude coordinates**")
            lon_min = st.number_input("Min longitude", step=1e-2, format="%.2f", min_value=-180.0, max_value=180.0)
            lon_max = st.number_input("Max longitude", step=1e-2, format="%.2f", min_value=-180.0, max_value=180.0)
            lon_is_ok = (lon_min < lon_max)
            lon_message = "Longitude **correct**" if lon_is_ok else "Write longitude coordinates"
            st.write(f"{lon_message}")
        region_info = (lat_min, lat_max, lon_min, lon_max)

    return region_type, region_info, simulation_length, use_waves



def validation_mode_interface():
    """
    Validation mode user interface
    """
    st.markdown("# Validation mode")
    col1, col2 = st.columns(2)

    # Upload a csv file
    with col1:
        file_selected = st.file_uploader("Upload your .csv track", ["csv"])
        lat_start, lon_start, datetime_start, date_start = None, None, None, None
        if file_selected is not None:
            lon_coords, lat_coords, time_coords = read_csv_file(file_selected)
            file_selected = (lon_coords, lat_coords, time_coords)
            lat_start, lon_start, datetime_start = lat_coords[0], lon_coords[0], time_coords[0]
            date_start = datetime.date(year=datetime_start.year, month=datetime_start.month, day=datetime_start.day)
            st.write(f"""
            **Starting latitude**: {lat_start}

            **Starting longitude**: {lon_start}

            **Time start**: {datetime_start}
            
            **Time ends**: {time_coords[-1]}
            """)
        else:
            st.write("No file selected")

    # Choose single or cloud of particles
    with col2:
        particle_set_type, amount_of_particles, radius_from_origin = single_or_cloud_of_particles()

    st.markdown("""
        ___
        ## Simulation conditions
        """)
    col3, col4 = st.columns(2)

    # Choose simulation region and time
    region_type, region_info, simulation_length, use_waves = simulation_region_and_time(col3, col4, date_start,
                                                                             file_selected is not None)

    return (lat_start, lon_start, datetime_start, file_selected), (particle_set_type, amount_of_particles, radius_from_origin), (
    region_type, region_info, simulation_length, use_waves)


def operation_mode_interface():
    """
    Operation mode user interface
    """
    st.markdown("""
    # Operation mode
    ## Particle initial position
    """)
    col1, col2, col3, col4 = st.columns(4)

    # Particle latitude
    with col1:
        st.markdown("### Latitude")
        lat_start = st.number_input("Eg: 41.2",
                                    step=1e-6, format="%.6f", min_value=-90.0, max_value=90.0)
        lat_is_ok = lat_start > 0  # modify to match IBI requirements

    # Particle longitude
    with col2:
        st.markdown("### Longitude")
        lon_start = st.number_input("Eg: 2.3",
                                    step=1e-6, format="%.6f", min_value=-180.0, max_value=180.0)
        lon_is_ok = lon_start > 0  # modify to match IBI requirements

    # Particle date
    with col3:
        st.markdown("### Date")
        date_start = st.date_input("Date of last known position",
                                   min_value=datetime.date(2016, 1, 1),
                                   max_value=datetime.date.today()
                                   )
        date_is_ok = True  # modify to match IBI requirements
    with col4:
        st.markdown("### Time")
        time_start = st.time_input("Time of last known position")

    st.markdown("""
    ___
    ## Simulation conditions
    """)
    col5, col6 = st.columns(2)

    # Choose simulation region and time
    region_type, region_info, simulation_length, use_waves = simulation_region_and_time(col5, col6, date_start, True)

    st.markdown("""
    ___
    """)
    col7, col8 = st.columns(2)

    # Choose single or cloud of particles
    with col7:
        st.markdown("### Cloud of particles")
        particle_set_type, amount_of_particles, radius_from_origin = single_or_cloud_of_particles()

    # Info logging
    with col8:
        st.markdown("""
        Information can be logged here.
        """)

    datetime_start = datetime.datetime(year=date_start.year, month=date_start.month, day=date_start.day,
                          hour=time_start.hour, minute=time_start.minute, second=time_start.second)
    return (lat_start, lon_start, datetime_start, None), (particle_set_type, amount_of_particles, radius_from_origin), (
    region_type, region_info, simulation_length, use_waves)