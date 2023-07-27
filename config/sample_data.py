# -*- coding: utf-8 -*-
"""
Created: 2021/03/30
@author: José M. Alsina (UPC)

Configuration variables  

Notes:
Dates must be in the format yyyy/m/d -> these must NOT be zero padded
"""

sim_name = 'sample_simulation'              # sets the naming convention for the simulation outputs    


# download variables 
domain_name = 'BCN'                         # domain name
domain_N = 41.81                            # domain N lat
domain_S = 40.88                            # domain S lat
domain_E = 3                                # domain E lat
domain_W = 1.38                             # domain W lat

download_regional = True 
download_coastal = True
download_port = True
download_waves = True
download_init = "2023/3/5"
download_end = "2023/3/7"


# simulation variables
rewrite = True                              # creates a new output file (True, requires hydrodynamic data files), or reads existing file (False)
nested_domain = True                        # True for nested data (coastal and harbour), False for regional data only

sim_init = "2023/3/5"                       # simulation start date
sim_init_time = "00:00:00"                  # simulation start time
sim_end = "2023/3/7"                        # simulation end date
sim_end_time = "23:00:00"                   # simulation end time


# If reading data from an external Excel spreadsheet
samples_filename = 'Data_sampler/' + sim_name + '.xlsx'  
rows_to_header = 2                          # row position of headers in excel (0-indexed)
sheets_to_load = ['Llobregat','Besos']      # delimit sheet names with a comma
datetime_col_name = 'Sampling date'         # Specify the name of the datetime column
particle_col_name = 'Particles'             # Specify the name of the column with particle numbers to simulate
particle_frequency = 'hourly'               # 'hourly', 'daily', or None (for one-time release e.g drifter)
coords = [[41.29285, 2.14149], [41.417671, 2.235227]]


depth = 1                                   # Depth of particles (m)
kh = 10                                     # Diffusion - range from 0.1 to 10, float value
sim_dt = 5                                  # Simulation time step
particle_proportion = 1                     # Proportion of simulated particles to be created 



# Variables for plotting 
figure_title = 'Figure title'
plot_title = 'Plot title'
sim_trajectory_label = 'Particle simulation'
terrain_type = 'standard'                   # Define terrain type as 'standard', 'satellite', or None
 
plot_path = 'plots/' + sim_name + '/'       # Path for all plots to be saved
plot_filename = plot_path + sim_name + '.jpg' 
animation_filename = sim_name + '_animation.mp4'  

plot_travel_dist = False                    # Plots distances traveled by particles - argument for parser
plot_animation = True                       # Plots animation of the particles movement
plot_trajectories = True                    # Plots the trajectories of the particles over a map
plot_concentration_heatmap = True           # Plot concentration heatmap
paths_with_arrows = False                   # Draws arrows over the paths in trajectories plot


data_base_path = 'hydrodynamic_data'        # relative to path of files importing this config
output_filename = 'particles/' + sim_name + '.zarr'


port_files_dir = "currents/harbour"
coastal_files_dir = "currents/coastal"
regional_files_dir = "currents/regional"
wave_regional_files_dir = 'waves/regional'
resampled_files_dir = '_resampled'          # suffix added if using resampled files. if not using resampled files have as None

# for resampling
port_file_search_string = 'BCNPRT*-HC.nc'
coastal_file_search_string = 'BCNCST*-HC.nc'
regional_file_search_string = 'MyO-IBI_hm_BCN*_HC01.nc'
wave_regional_file_search_string = 'WAV-IBI_hm*_HC01.nc'

# For distance to shore kernel for beaching
dist_path = 'nodes'
port_dist = 'harbour_nodes.nc'
coastal_dist = 'coastal_nodes.nc'
regional_dist = 'regional_nodes.nc'







