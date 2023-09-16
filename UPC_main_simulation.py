# -*- coding: utf-8 -*-
"""
Created: 2021/03/30
@author: Jose M. Alsina - UPC
"""
from config import sample_data as cfg

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(module)s | %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import argparse
#from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, plotTrajectoriesFile, Variable
from parcels import FieldSet, NestedField, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, plotTrajectoriesFile, Variable
from parcels.field import Field, VectorField
from datetime import timedelta, datetime
import UPC_parcels_objects as lib
from UPC_Parcels_kernels import *
import numpy as np
import xarray as xr
import cartopy.io.img_tiles as cimgt
from pathlib import Path
from operator import attrgetter
import zarr
import warnings

#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')



# Datetime variables and calculations
sim_split_i = cfg.sim_init.split("/")
sim_split_e = cfg.sim_end.split("/")
sim_split_i_t = cfg.sim_init_time.split(":")
sim_split_e_t = cfg.sim_end_time.split(":")

# sim init date and time in a list
YYi = [int(sim_split_i[0]),int(sim_split_i[1]),int(sim_split_i[2])]
YYi_t = [int(sim_split_i_t[0]),int(sim_split_i_t[1]),int(sim_split_i_t[2])]
# sim end date
YYe = [int(sim_split_e[0]),int(sim_split_e[1]),int(sim_split_e[2])]
   
# calculate simulation duration
sim_date_format_str = '%Y/%m/%d %H:%M:%S'

# Get interval between two timstamps as timedelta object
sim_duration = datetime.strptime(cfg.sim_end + ' ' + cfg.sim_end_time, sim_date_format_str) - datetime.strptime(cfg.sim_init + ' ' + cfg.sim_init_time, sim_date_format_str)

sim_duration = sim_duration.total_seconds() / 86400






def main(samples_filename, rewrite, sim_init=[], sim_duration=5, sim_dt=5, particle_proportion=1.0, plot_travel_dist=False, plot_animation=False, animation_filename=None, terrain_type=None, arrows_paths=False, plot_trajectories=False, sheets_to_load=None, coords=None):
    """
    Main function of simulation. It loads the data and perform the desired simulation.
    It can show the resuts in diferents ways depending on the different values of the parameters.
    :param samples_filename: excel (xlxs, xls, odt) defining the samples to be created
    :param rewrite: boolean that specifis if simulation data should be rewrite, therefore re-running complete simulation or just display data
    :param sim_days: number of days to be simulated
    :param sim_dt: differential of time in minutes for the simulation (default is 5 minutes)
    :param particle_proportion: proportion of the total number of particles to be created (useful for running smaller simulations)
    :param plot_travel_dist: draws a plot of the distances traveled for each particle
    :param plot_animation: plots an animation of the movement of the particles over a map
    :param animation_filename: file name for the animation to be saved
    :param terrain_type: type of map
    :param flechas_en_paths: include arrows in trajetories view
    :param plot_trajectories: draws a map with the trajectories of the particles
    :param sheets_to_load: sheets of the excel to be loaded
    :param coords: array containing the sample generation coordinates corresponding to each sheet in the excel file
    """
    if rewrite:

        logger.info(f"Rewrite selected. Loading particle sampler from {samples_filename} and running simulation")

        # loads nested field function if nesting fields        
        if cfg.nested_domain == True:
            fieldset = lib.get_nested_fieldset()
            #lib.get_nested_fieldset_distance(fieldset)
        else:
           fieldset = lib.get_regional_fieldset()
           #lib.get_regional_fieldset_distance(fieldset)
        
        
        waves_dir = Path(cfg.data_base_path) / Path(cfg.wave_regional_files_dir)
        waves_str = cfg.wave_regional_file_search_string
        
        # Add Stokes drift
        lib.set_Stokes(fieldset, waves_dir, waves_str)
        

        # Add difussion
        kh = cfg.kh  # 0.10-10
        fieldset.add_constant_field("Kh_zonal", kh, mesh='spherical')
        fieldset.add_constant_field("Kh_meridional", kh, mesh='spherical')

    
        # Define particle class and variables for nc file
        class PlasticParticle(JITParticle):
            age = Variable('age', dtype=np.float32, initial=0.)
            
            # essential variable to parameterise in a kernel the definition of beaching
            beached = Variable('beached', dtype=np.int32, initial=0.)
            
            # tags particles which move out of the domain limits (exported)
            out_of_bounds = Variable('out_of_bounds', dtype=np.int32, initial=0.)
            
            # calculates the trajectory distance for each particle with a kernel. 
            # requires prev_lon and prev_lat for the calculation
            # distance_traj = Variable('distance_traj', dtype=np.float32, initial=0.)
            # prev_lon = Variable('prev_lon', dtype=np.float32, to_write=True, initial=attrgetter('lon'))  
            # prev_lat = Variable('prev_lat', dtype=np.float32, to_write=True, initial=attrgetter('lat'))                        
            
            # used by the beaching distance kernel, as well as the beached variable
            # distance_shore = Variable('distance_shore', dtype=np.float32, initial=0.)
            
            # used by the beaching proximity kernel, as well as the beached variable
            # proximity = Variable('proximity', dtype=np.float32, initial=0.)  # to calculate time within
            
            # not processed by a kernel, records the original lat and lon through time steps
            # origin_lat = Variable('origin_lat', dtype=np.float32, to_write=True, initial=attrgetter('lat'))
            # origin_lon = Variable('origin_lon', dtype=np.float32, to_write=True, initial=attrgetter('lon'))

    
        
        # Manual configuration of a simulation if not using Excel for input data
        if cfg.samples_filename == None:
            pset = ParticleSet.from_list(fieldset=fieldset,
                                         pclass=PlasticParticle,
                                         lon=[2.14149],
                                         lat=[41.29285],
                                         # time format [datetime(yyyy,mm,dd,h,m)]
                                         # can have more than one date within [] for various releases
                                         time=[datetime(int(YYi[0]), int(YYi[1]), int(YYi[2]), int(YYi_t[0]), int(YYi_t[1]))],
                                         depth=cfg.depth)
        
        # if using Excel for input data
        else: 
    
            # Define initial and final simulation times within a fieldset - this is separate from the duration 
            init_time = datetime(int(YYi[0]),int(YYi[1]),int(YYi[2]))
            final_time = datetime(int(YYe[0]),int(YYe[1]),int(YYe[2]))

    
            # hourly or daily particle release - uses ParticleXLSsampler function 
            if cfg.particle_frequency == 'hourly' or cfg.particle_frequency == 'daily':
                # Particles are loaded from the excel spreadsheet
                sampler = lib.ParticleXLSSampler(cfg.samples_filename, cfg.sheets_to_load, cfg.coords, cfg.rows_to_header, cfg.particle_proportion, init_time, final_time)
                pset = ParticleSet.from_list(fieldset=fieldset,
                                             pclass=PlasticParticle,
                                             lon=sampler.lon,
                                             lat=sampler.lat,
                                             depth=None,
                                             time=sampler.time)
                
            # one time particle releases (e.g for drifters) - uses ParticleXLScreator function - must specify lats and lons in the excel if that is the case
            elif cfg.particle_frequency == None:
                # Lats and lons are loaded from the excel spreadsheet
                creator = lib.ParticleXLSCreator(cfg.samples_filename, cfg.sheets_to_load, cfg.rows_to_header, cfg.particle_proportion, init_time, final_time)
                pset = ParticleSet.from_list(fieldset=fieldset,
                                             pclass=PlasticParticle,
                                             lon=creator.lon,
                                             lat=creator.lat,
                                             depth=None,
                                             time=creator.time)
    
        
        # if using the current velocity kernel for beaching 
        kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh_custom) + pset.Kernel(StokesDrag)  + pset.Kernel(Ageing) + pset.Kernel(beaching_velocity)
        
        # if using the distance to shore kernel for beaching
        # ensure revelant PlasticParticles variables are activated
        # kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh_custom) + pset.Kernel(StokesDrag) + pset.Kernel(Ageing) + pset.Kernel(distance_shore) + pset.Kernel(beaching_distance)  
        
        # if using the proximity kernel for beaching
        # ensure revelant PlasticParticles variables are activated
        # kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh_custom) + pset.Kernel(StokesDrag) + pset.Kernel(Ageing) + pset.Kernel(distance_shore) + pset.Kernel(beaching_proximity)
        
        output_file = pset.ParticleFile(name=cfg.output_filename, outputdt=timedelta(hours=1))
        sim_dt = cfg.sim_dt
              
        # run the simulation using the kernels
        pset.execute(kernel,
                     runtime=timedelta(days=sim_duration),
                     dt=timedelta(minutes=sim_dt),
                     output_file=output_file,
                     recovery={ErrorCode.ErrorOutOfBounds: lib.DeleteParticleFunction},
                     verbose_progress=True
                     )
        
        logger.info(f"Simulation results saved to: {cfg.output_filename}")
       

    # PLOTS
    logger.info(f"Reading simulation data results from: {cfg.output_filename}")    
    
    data = xr.open_dataset(cfg.output_filename)
        
    domain = {'N': cfg.domain_N, 'S': cfg.domain_S, 'E': cfg.domain_E, 'W': cfg.domain_W}
    # Terrain types, options for standard are terrain, terrain-background, toner, watercolor
    assert terrain_type in ['standard', 'satellite', None], "Terrain type must be 'standard', 'satellite' or None"
    if terrain_type=='standard':
        terrain = cimgt.Stamen('terrain-background') # max_zoom=10
        terrain_zoom = 10 
    elif terrain_type=='satellite':
        terrain = cimgt.QuadtreeTiles() # max_zoom=13
        terrain_zoom = 13
    elif terrain_type is None:
        terrain = None
        terrain_zoom = 10
    else:
        terrain = cimgt.OSM() # max_zoom=13
        terrain_zoom = 13
    
    # Shows particle distance travelled per particle
    if plot_travel_dist:
        lib.plot_trajectory_distance(data)

    # Video animation on map
    if plot_animation:
        lib.plot_animation(data, domain=domain, terrain=terrain, filename=animation_filename, show_plot=True)


    # Trayectories on map
    if plot_trajectories:
        lib.plot_trajectories(data, field=None, domain=domain, terrain=terrain, terrain_zoom=terrain_zoom, graph_type='Lines', arrows_paths=arrows_paths)                       # trayectorias sin field
        # flechas_en_paths=flechas_en_paths
    
    # Particle concentration density map
    if cfg.plot_concentration_heatmap == True:
        lib.plot_concentration_heatmap(data, domain=domain, terrain=terrain, sim_steps=(60/cfg.sim_dt)*24)





if __name__ == '__main__':
    
    """
    If/else statements that deal with double negatives and confusion these cause in the argument parser.
    Useful for calling from command prompmt or terminal once all the other values are set in the config file.
    """
    if cfg.rewrite == True:
        rewrite_args_action = 'store_false'
    else:
        rewrite_args_action = 'store_true'
    
    if cfg.plot_travel_dist == True:
        plot_travel_dist_args_action = 'store_false'
    else:
        plot_travel_dist_args_action = 'store_true'
    
    if cfg.plot_animation == True:
        plot_animation_args_action = 'store_false'
    else:
        plot_animation_args_action = 'store_true'
        
    if cfg.plot_trajectories == True:
        plot_trajectories_args_action = 'store_false'
    else:
        plot_trajectories_args_action = 'store_true'
        
    if cfg.paths_with_arrows == True:
        paths_with_arrows_args_action = 'store_false'
    else:
        paths_with_arrows_args_action = 'store_true'
    
    
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Particle simulation")
        parser.add_argument("--samples_filename", help="Particle generation filename", default=cfg.samples_filename)
        parser.add_argument('--rewrite', dest='rewrite', action=rewrite_args_action, help="Creates .nc file and replaces in the directory if there was a previous one. If false, it reads from existing .nc file")
        parser.add_argument('--sim_duration', type=float, dest='sim_duration', default=sim_duration, help="Duration in days to simulate")
        parser.add_argument('--sim_init', type=int, dest='sim_init', default=YYi, help="Initialization of simulation")
        parser.add_argument("--sim_dt", type=int, dest='sim_dt', default=cfg.sim_dt, help="Minutes between simulation steps")
        parser.add_argument("--particle_proportion",type=float, dest='particle_proportion', default=cfg.particle_proportion, help="Total_number_of_particles * particle_proportion = simulated_particles")
        parser.add_argument("--plot_travel_dist", dest='plot_travel_dist', action=plot_travel_dist_args_action, help="Plots distances traveled by particles")
        parser.add_argument("--plot_animation", dest='plot_animation', action=plot_animation_args_action, help="Plots animation of the particles movement")
        parser.add_argument("--animation_filename", dest='animation_filename', help="If provided, and plot_animation is True, the animation will be saved to this file", default=cfg.animation_filename)
        parser.add_argument("--terrain_type", dest='terrain_type', type=str,  help="Can be 'standard', 'satellite' or not defined", default=cfg.terrain_type)
        parser.add_argument("--plot_trajectories", dest='plot_trajectories', action=plot_trajectories_args_action, help='To plot or not the trajectories of the particles over a map')
        parser.add_argument("--paths_with_arrows", dest='arrows_paths', action=paths_with_arrows_args_action, help="Draws arrows over the paths in trajectories plot")
        return parser.parse_args()
    args = parse_args()

    logger.info("Selected parameters values:")
    for arg in vars(args):
        logger.info(f"{arg} = {getattr(args, arg)}")

    main(samples_filename=args.samples_filename,
         rewrite=args.rewrite,
         sim_init=YYi,
         sim_duration=sim_duration,
         sim_dt=cfg.sim_dt,
         particle_proportion=cfg.particle_proportion,
         plot_travel_dist=args.plot_travel_dist,
         plot_animation=args.plot_animation,
         animation_filename=args.animation_filename,
         terrain_type=cfg.terrain_type,
         plot_trajectories=args.plot_trajectories,
         arrows_paths=args.arrows_paths,
         sheets_to_load=cfg.sheets_to_load,
         coords=cfg.coords
         )