# -*- coding: utf-8 -*-
"""
Created: 2021/03/30
@author: Jose M. Alsina - UPC
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(module)s | %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from parcels import ParticleSet, JITParticle, AdvectionRK4,  ErrorCode, Variable
from datetime import timedelta, datetime
from utils.locate_operations.UPC_Parcels_kernels import *
from utils.locate_operations import UPC_config as cfg, UPC_parcels_objects as lib
from utils.locate_operations.locate_utils import point_cloud_from_coordinate
import numpy as np
from pathlib import Path



def simulation_main(cloud_of_particles,
                    start_coordinates, # [lat, lon]
                    start_datetime,
                    simulation_days,
                    use_waves,
                    data_base_path=cfg.data_base_path,  # this parameter could be made compulsory in the future
                    radius_from_origin = 1,  # this parameter is optional as it is only used if cloud_of_particles == True
                    amount_of_particles = 100):
    # Carrega del fieldset
    fieldset = lib.get_ibi_fieldset()

    if use_waves:
        print("Using waves")
        # If waves are present
        waves_dir = Path(data_base_path) / Path(cfg.Wave_IBI_files_dir)
        waves_str = cfg.Wave_IBI_file_search_string
        lib.set_Stokes(fieldset, waves_dir, waves_str)

    use_wind = True
    if use_wind:
        print("Using wind")
        # If waves are present
        waves_dir = Path(data_base_path) / Path(cfg.Wind_files_dir)
        waves_str = cfg.Wind_file_search_string
        lib.set_Leeway(fieldset, waves_dir, waves_str)

    # Add difussion
    kh = 10  # 0.10-10
    fieldset.add_constant_field("Kh_zonal", kh, mesh='spherical')
    fieldset.add_constant_field("Kh_meridional", kh, mesh='spherical')

    # Define particle class
    class PlasticParticle(JITParticle):
        # age = Variable('age', dtype=np.float32, initial=0.)
        # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach
        beached = Variable('beached', dtype=np.int32, initial=0.)
        # unbeachCount = Variable('unbeachCount', dtype=np.int32, initial=0.)

    # Creacio del particle set
    if cloud_of_particles:  # cloud of particles
        lat, lon, times = point_cloud_from_coordinate(start_coordinates,
                                                      radius_from_origin,
                                                      amount_of_particles,
                                                      start_datetime)
    elif not cloud_of_particles:  # single particle
        lat, lon, times = [start_coordinates[0]], [start_coordinates[1]], [start_datetime]

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle, lon=lon, lat=lat, time=times)

    print("El particle set és: ",  pset)

    # Set kernels
    print("\nDefinim el kernel:")
    # kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(BeachTesting_2D) + pset.Kernel(DiffusionUniformKh_custom) + pset.Kernel(StokesDrag)
    kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh_custom)
    if use_waves:
        kernel += pset.Kernel(StokesDrag)
    if use_wind:
        kernel += pset.Kernel(Leeway)

    output_file = pset.ParticleFile(name=cfg.harbour_particles_sim_filename, outputdt=timedelta(hours=1))
    print("\nSimulation starts...")
    simulation_time_step = 5
    pset.execute(kernel,
                 runtime=timedelta(days=simulation_days),
                 dt=timedelta(minutes=simulation_time_step),
                 output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: lib.DeleteParticleFunction},
                 verbose_progress=True
                 )
    print("Simulation terminated")
    output_file.export()
    output_file.close()
    logger.info(f"Simulation results saved to: {cfg.harbour_particles_sim_filename}")


if __name__ == '__main__':
    simulation_main(cloud_of_particles=True, start_coordinates=[42.5031, 3.45183],
                    start_datetime=np.datetime64(datetime(2022, 2, 3)), simulation_days=5,
                    use_waves=True,
                    data_base_path="../Data_proves_baixar", radius_from_origin=1, amount_of_particles=200)