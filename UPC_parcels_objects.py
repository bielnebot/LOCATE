# -*- coding: utf-8 -*-
"""
Created: 2021/03/30
@author: José M. Alsina (UPC) with Deep Solutions contributions
"""
from config import sample_data as cfg

#import UPC_config as cfg

import logging
logger = logging.getLogger(__name__)

from parcels import FieldSet, NestedField, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, plotTrajectoriesFile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from parcels.plotting import create_parcelsfig_axis, cartopy_colorbar, parsetimestr
import cartopy.crs as ccrs
from parcels.field import Field, VectorField
from parcels.grid import GridCode, CurvilinearGrid
from parcels.tools.statuscodes import TimeExtrapolationError
import cmocean
import copy
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.animation import FuncAnimation, FFMpegWriter
from math import radians, cos, sin, asin, sqrt
from matplotlib import cm
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
import os



def get_port_fieldset() -> FieldSet:
    
    # Loads the port files from folder and returns a fieldset
    
    dir = Path(cfg.data_base_path) / Path(cfg.port_files_dir)
    filenames = {'U': sorted(dir.glob(cfg.port_file_search_string)),
                 'V': sorted(dir.glob(cfg.port_file_search_string))}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    return fieldset


def get_coastal_fieldset() -> FieldSet:
    
    # Loads the coastal files from the folder and returns a fieldset
    
    dir = Path(cfg.data_base_path) / Path(cfg.coastal_files_dir)
    filenames = {'U': sorted(dir.glob(cfg.coastal_file_search_string)),
                 'V': sorted(dir.glob(cfg.coastal_file_search_string))}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    return fieldset

def get_regional_fieldset() -> FieldSet:
    
    # Loads the regionall files from the folder and returns a fieldset
    
    dir = Path(cfg.data_base_path) / Path(cfg.regional_files_dir)
    filenames = {'U': sorted(dir.glob(cfg.regional_file_search_string)),
                 'V': sorted(dir.glob(cfg.regional_file_search_string))}
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    return fieldset


def get_nested_fieldset() -> FieldSet:

    # Loads the resampled port-coastal, costal-regional domains and nests them
    
    base_path = Path(cfg.data_base_path)
    
    # Load resampled port files
    # Check to see if using resampled files
    if cfg.resampled_files_dir != None:
        dir_prt = base_path / Path(cfg.port_files_dir + cfg.resampled_files_dir)
    else:
        dir_prt = base_path / Path(cfg.port_files_dir)
        
    assert dir_prt.is_dir(), f"Directory not found: {dir_prt}"
    basepath_prt = '*.nc'
    filenames_prt = sorted(dir_prt.glob(str(basepath_prt)))
    dimensions = {'time': 'time', 'depth': 'depth', 'lon': 'longitude', 'lat': 'latitude'}
    U_prt = Field.from_netcdf(filenames_prt, ('U', 'u'), dimensions, fieldtype='U', allow_time_extrapolation=True)
    V_prt = Field.from_netcdf(filenames_prt, ('V', 'v'), dimensions, fieldtype='V', allow_time_extrapolation=True)

    # Loads resampled coastal files
    # Check to see if using resampled files
    if cfg.resampled_files_dir != None:
        dir_cst = base_path / Path(cfg.coastal_files_dir + cfg.resampled_files_dir)
    else:
        dir_cst = base_path / Path(cfg.coastal_files_dir)
        
    assert dir_cst.is_dir(), f"Directory not found: {dir_cst}"
    basepath_cst = '*.nc'
    filenames_cst = sorted(dir_cst.glob(str(basepath_cst)))
    U_cst = Field.from_netcdf(filenames_cst, ('U', 'u'), dimensions, fieldtype='U', allow_time_extrapolation=True)
    V_cst = Field.from_netcdf(filenames_cst, ('V', 'v'), dimensions, fieldtype='V', allow_time_extrapolation=True)

    # Loads resampled regional files
    
    # Check to see if using resampled files
    if cfg.resampled_files_dir != None:
        dir_reg = base_path / Path(cfg.regional_files_dir + cfg.resampled_files_dir)
    else:
        dir_reg = base_path / Path(cfg.regional_files_dir)
        
    assert dir_reg.is_dir(), f"Directory not found: {dir_reg}"
    basepath_reg = '*.nc'
    filenames_reg = sorted(dir_reg.glob(str(basepath_cst)))
    U_reg = Field.from_netcdf(filenames_reg, ('U', 'u'), dimensions, fieldtype='U', allow_time_extrapolation=True)
    V_reg = Field.from_netcdf(filenames_reg, ('V', 'v'), dimensions, fieldtype='V', allow_time_extrapolation=True)

    # Nest the fieldsets
    fieldset_prt = FieldSet(U_prt, V_prt)
    fieldset_cst = FieldSet(U_cst, V_cst)
    fieldset_reg = FieldSet(U_reg, V_reg)

    Ufield = NestedField('U', [fieldset_prt.U, fieldset_cst.U, fieldset_reg.U])
    Vfield = NestedField('V', [fieldset_prt.V, fieldset_cst.V, fieldset_reg.V])
    fieldset = FieldSet(Ufield, Vfield)

    return fieldset


def set_Stokes(fieldset, data_dir, data_str):
    
    # Adds a field with the Stokes drift
    
    dir = Path(data_dir)
    basepath = data_str
    fnames = sorted(dir.glob(str(basepath)))
    dimensionsU = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    dimensionsV = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    variablesU = ('Uuss', 'VSDX')
    variablesV = ('Vuss', 'VSDY')
    Uuss = Field.from_netcdf(fnames, variablesU, dimensionsU, fieldtype='U', allow_time_extrapolation=True)
    Vuss = Field.from_netcdf(fnames, variablesV, dimensionsV, fieldtype='V', allow_time_extrapolation=True,
                             grid=Uuss.grid, dataFiles=Uuss.dataFiles)
    fieldset.add_field(Uuss)
    fieldset.add_field(Vuss)
    fieldset.Uuss.vmax = 5
    fieldset.Vuss.vmax = 5
    uv_uss = VectorField('UVuss', fieldset.Uuss, fieldset.Vuss)
    fieldset.add_vector_field(uv_uss)



def get_nested_fieldset_distance (fieldset) -> FieldSet:
    
    #Loads the distance fields into a nested fieldset

    base_path_dist = Path(cfg.data_base_path) / Path(cfg.dist_path)
    
    filenames_dist_reg = base_path_dist / Path(cfg.regional_dist)
    filenames_dist_cst = base_path_dist / Path(cfg.coastal_dist)
    filenames_dist_prt = base_path_dist / Path(cfg.port_dist)    
    
    dimensions_dist = {'time': 'time', 'depth': 'depth', 'lat': 'latitude', 'lon': 'longitude'}

    # mesh='spherical' assumes units are in degrees and converts to metres - scaling issues
    # mesh='flat' assumes units are in metres
    # https://nbviewer.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb
    U_dist_prt = Field.from_netcdf(filenames_dist_prt, ('U_dist', 'u'), dimensions_dist, fieldtype='U', mesh='flat', allow_time_extrapolation=True)
    V_dist_prt = Field.from_netcdf(filenames_dist_prt, ('V_dist', 'v'), dimensions_dist, fieldtype='V', mesh='flat', allow_time_extrapolation=True)
    
    U_dist_cst = Field.from_netcdf(filenames_dist_cst, ('U_dist', 'u'), dimensions_dist, fieldtype='U', mesh='flat', allow_time_extrapolation=True)
    V_dist_cst = Field.from_netcdf(filenames_dist_cst, ('V_dist', 'v'), dimensions_dist, fieldtype='V', mesh='flat', allow_time_extrapolation=True)
    
    U_dist_reg = Field.from_netcdf(filenames_dist_reg, ('U_dist', 'u'), dimensions_dist, fieldtype='U', mesh='flat', allow_time_extrapolation=True)
    V_dist_reg = Field.from_netcdf(filenames_dist_reg, ('V_dist', 'v'), dimensions_dist, fieldtype='V', mesh='flat', allow_time_extrapolation=True)
        
    # the fieldsets are nested so there would need to be a conversion of m/s * m/s
    fieldset_dist_prt = FieldSet(U_dist_prt, V_dist_prt)
    fieldset_dist_cst = FieldSet(U_dist_cst, V_dist_cst)
    fieldset_dist_reg = FieldSet(U_dist_reg, V_dist_reg)
        
    # create a distinct U and V field distance field to eliminate confusion inthe kernel
    # the smallest/finest resolution fields have to be listed before the larger/coarser resolution fields
    U_dist = NestedField('U_dist', [fieldset_dist_prt.U, fieldset_dist_cst.U, fieldset_dist_reg.U])
    V_dist = NestedField('V_dist', [fieldset_dist_prt.V, fieldset_dist_cst.V, fieldset_dist_reg.V])
       
    fieldset_dist = fieldset.add_field(U_dist)
    fieldset_dist = fieldset.add_field(V_dist)
    
    return fieldset_dist


def get_regional_fieldset_distance(fieldset) -> FieldSet:
    
    # Loads only the regional distance field when only using the regional domain

    base_path_dist = Path(cfg.data_base_path) / Path(cfg.dist_path)    
    filenames_dist_reg = base_path_dist / Path(cfg.regional_dist)    
    dimensions_dist = {'time': 'time', 'depth': 'depth', 'lat': 'latitude', 'lon': 'longitude'}
    
    U_dist_reg = Field.from_netcdf(filenames_dist_reg, ('U_dist', 'u'), dimensions_dist, fieldtype='U', mesh='flat', allow_time_extrapolation=True)
    V_dist_reg = Field.from_netcdf(filenames_dist_reg, ('V_dist', 'v'), dimensions_dist, fieldtype='V', mesh='flat', allow_time_extrapolation=True)
        
    fieldset_dist_reg = FieldSet(U_dist_reg, V_dist_reg)     
    fieldset_dist = fieldset.add_field('U_dist', [fieldset_dist_reg.U])
    fieldset_dist = fieldset.add_field('V_dist', [fieldset_dist_reg.V])    
  
    return fieldset_dist
    

def DeleteParticleFunction(particle, fieldset, time):
    particle.delete()
    
def OutOfBounds(particle, fieldset, time):
    particle.out_of_bounds = 1
    particle.delete()



def parsedomain(domain, field):
    field.grid.check_zonal_periodic()
    dominio_desplazado = False
    if domain is not None:
        if not isinstance(domain, dict) and len(domain) == 4:  # for backward compatibility with <v2.0.0
            new_domain = {'N': domain[0], 'S': domain[1], 'E': domain[2], 'W': domain[3]}
        else:
            new_domain = domain.copy()
        min_lon, max_lon = field.grid.lon[0], field.grid.lon[-1]
        min_lat, max_lat = field.grid.lat[0], field.grid.lat[-1]
        if new_domain['W'] < min_lon:
            dominio_desplazado = True
            new_domain['W'] = min_lon
        if new_domain['E'] > max_lon:
            dominio_desplazado = True
            new_domain['E'] = max_lon
        if new_domain['S'] < min_lat:
            dominio_desplazado = True
            new_domain['S'] = min_lat
        if new_domain['N'] > max_lat:
            dominio_desplazado = True
            new_domain['N'] = max_lat
        _, _, _, lonW, latS, _ = field.search_indices(new_domain['W'], new_domain['S'], 0, 0, 0, search2D=True)
        _, _, _, lonE, latN, _ = field.search_indices(new_domain['E'], new_domain['N'], 0, 0, 0, search2D=True)
        return latN + 1, latS, lonE + 1, lonW, dominio_desplazado
    else:
        if field.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            return field.grid.lon.shape[0], 0, field.grid.lon.shape[1], 0, dominio_desplazado
        else:
            return len(field.grid.lat), 0, len(field.grid.lon), 0, dominio_desplazado


def plotfield(field, show_time=None, domain=None, depth_level=0, projection=None, land=True,
              vmin=None, vmax=None, savefile=None, arrow_density=1.0, **kwargs):
    
    # Function to plot a Parcels Field   

    """
    :param show_time: Time at which to show the Field
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
    :param depth_level: depth level to be plotted (default 0)
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """

    if type(field) is VectorField:
        spherical = True if field.U.grid.mesh == 'spherical' else False
        field = [field.U, field.V]
        plottype = 'vector'
    elif type(field) is Field:
        spherical = True if field.grid.mesh == 'spherical' else False
        field = [field]
        plottype = 'scalar'
    else:
        raise RuntimeError('field needs to be a Field or VectorField object')

    if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.CurvilinearSGrid]:
        logger.warning('Field.show() does not always correctly determine the domain for curvilinear grids. '
                       'Use plotting with caution and perhaps use domain argument as in the NEMO 3D tutorial')

    plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, land, projection=projection,
                                                   cartopy_features=kwargs.pop('cartopy_features', []))
    if plt is None:
        return None, None, None, None  # creating axes was not possible

    data = {}
    plotlon = {}
    plotlat = {}
    for i, fld in enumerate(field):
        show_time = fld.grid.time[0] if show_time is None else show_time
        if fld.grid.defer_load:
            fld.fieldset.computeTimeChunk(show_time, 1)
        (idx, periods) = fld.time_index(show_time)
        show_time -= periods * (fld.grid.time_full[-1] - fld.grid.time_full[0])
        if show_time > fld.grid.time[-1] or show_time < fld.grid.time[0]:
            raise TimeExtrapolationError(show_time, field=fld, msg='show_time')

        latN, latS, lonE, lonW, dominio_desplazado = parsedomain(domain, fld)
        if isinstance(fld.grid, CurvilinearGrid):
            plotlon[i] = fld.grid.lon[latS:latN, lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN, lonW:lonE]
        else:
            plotlon[i] = fld.grid.lon[lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN]
        if i > 0 and not np.allclose(plotlon[i], plotlon[0]):
            raise RuntimeError('VectorField needs to be on an A-grid for plotting')
        if fld.grid.time.size > 1:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[depth_level, latS:latN,
                          lonW:lonE]
            else:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[latS:latN, lonW:lonE]
        else:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.sim_data)[depth_level, latS:latN, lonW:lonE]
            else:
                data[i] = np.squeeze(fld.sim_data)[latS:latN, lonW:lonE]

    if plottype == 'vector':
        if field[0].interp_method == 'cgrid_velocity':
            logger.warning_once(
                'Plotting a C-grid velocity field is achieved via an A-grid projection, reducing the plot accuracy')
            d = np.empty_like(data[0])
            d[:-1, :] = (data[0][:-1, :] + data[0][1:, :]) / 2.
            d[-1, :] = data[0][-1, :]
            data[0] = d
            d = np.empty_like(data[0])
            d[:, :-1] = (data[0][:, :-1] + data[0][:, 1:]) / 2.
            d[:, -1] = data[0][:, -1]
            data[1] = d

        spd = data[0] ** 2 + data[1] ** 2
        speed = np.where(spd > 0, np.sqrt(spd), 0)
        vmin = speed.min() if vmin is None else vmin
        vmax = speed.max() if vmax is None else vmax
        # ncar_cmap = copy.copy(cmocean.cm.speed)   # green gradient
        ncar_cmap = copy.copy(cmocean.cm.matter)    # red gradient
        # ncar_cmap = copy.copy(plt.cm.gist_ncar)   # multicolour gradient
        ncar_cmap.set_over('k')
        ncar_cmap.set_under('w')
        if isinstance(field[0].grid, CurvilinearGrid):
            x, y = plotlon[0], plotlat[0]
        else:
            x, y = np.meshgrid(plotlon[0], plotlat[0])
        u = np.where(speed > 0., data[0] / speed, np.nan)
        v = np.where(speed > 0., data[1] / speed, np.nan)
        skip = (slice(None, None, int(1 / arrow_density)), slice(None, None, int(1 / arrow_density)))
        if cartopy:
            cs = ax.quiver(np.asarray(x)[skip], np.asarray(y)[skip], np.asarray(u)[skip], np.asarray(v)[skip], speed[skip], cmap=ncar_cmap, clim=[vmin, vmax], scale=100, transform=cartopy.crs.PlateCarree(), minshaft=3, pivot='mid')
        else:
            cs = ax.quiver(x[skip], y[skip], u[skip], v[skip], speed[skip], cmap=ncar_cmap, clim=[vmin, vmax], scale=50)
    else:
        vmin = data[0].min() if vmin is None else vmin
        vmax = data[0].max() if vmax is None else vmax
        pc_cmap = copy.copy(cmocean.cm.speed)
        pc_cmap.set_over('k')
        pc_cmap.set_under('w')
        assert len(data[0].shape) == 2
        if field[0].interp_method == 'cgrid_tracer':
            d = data[0][1:, 1:]
        elif field[0].interp_method == 'cgrid_velocity':
            if field[0].fieldtype == 'U':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][1:, :-1] + data[0][1:, 1:]) / 2.
            elif field[0].fieldtype == 'V':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][:-1, 1:] + data[0][1:, 1:]) / 2.
            else:  # W
                d = data[0][1:, 1:]
        else:  # if A-grid
            d = (data[0][:-1, :-1] + data[0][1:, :-1] + data[0][:-1, 1:] + data[0][1:, 1:]) / 4.
            d = np.where(data[0][:-1, :-1] == 0, np.nan, d)
            d = np.where(data[0][1:, :-1] == 0, np.nan, d)
            d = np.where(data[0][1:, 1:] == 0, np.nan, d)
            d = np.where(data[0][:-1, 1:] == 0, np.nan, d)
        if cartopy:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=pc_cmap, transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=pc_cmap)

    if cartopy is None:
        ax.set_xlim(np.nanmin(plotlon[0]), np.nanmax(plotlon[0]))
        ax.set_ylim(np.nanmin(plotlat[0]), np.nanmax(plotlat[0]))
    elif domain is not None:
        ax.set_extent([np.nanmin(plotlon[0]), np.nanmax(plotlon[0]), np.nanmin(plotlat[0]), np.nanmax(plotlat[0])],
                      crs=cartopy.crs.PlateCarree())
    if dominio_desplazado:
        # aqui se puede cambiar el dominio mostrado (mas allá de donde haya datos)
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    cs.set_clim(vmin, vmax)

    cartopy_colorbar(cs, plt, fig, ax)

    # Labels, etc
    timestr = parsetimestr(field[0].grid.time_origin, show_time)
    titlestr = kwargs.pop('titlestr', '')
    if field[0].grid.zdim > 1:
        if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.RectilinearZGrid]:
            gphrase = 'depth'
            depth_or_level = field[0].grid.depth[depth_level]
        else:
            gphrase = 'level'
            depth_or_level = depth_level
        depthstr = ' at %s %g ' % (gphrase, depth_or_level)
    else:
        depthstr = ''
    if plottype == 'vector':
        ax.set_title(titlestr + 'Velocity field' + depthstr + timestr)
    else:
        ax.set_title(titlestr + field[0].name + depthstr + timestr)

    if not spherical:
        ax.set_xlabel('Zonal distance [m]')
        ax.set_ylabel('Meridional distance [m]')

    plt.draw()

    if savefile:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()

    return plt, fig, ax, cartopy





def to_date(string):
    
    # Function that changes date from string to datetime object
    # Two different formats to deal with Excel date formatting
    
    try:
        return np.datetime64(datetime.strptime(string, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return np.datetime64(datetime.strptime(string, "%d/%m/%Y"))


class ParticleXLSLoader(object):
    
    # Function to read data from Excel files for particle release
    
    def __init__(self, filename, sheet_names, header, datetime_columns=[cfg.datetime_col_name]):
        
        """
        params:
        filename: path to excel file (can be an URL)
        sheet_names: list of sheets to open, None gets all sheets
        header: Row (0-indexed) to use for the column labels
        """
        
        self.datetime_columns = datetime_columns
        converters = {}
        for column_name in datetime_columns:
            converters[column_name] = str
        # uses pandas to read the excel file
        self.data = pd.read_excel(filename, sheet_name=sheet_names, header=header, converters=converters, engine='openpyxl')

        for sheet_name in self.data.keys():
            for data_column in self.datetime_columns:
                self.data[sheet_name][data_column] = self.data[sheet_name][data_column].map(to_date, na_action='ignore')
        logger.info(f"Excel file loaded: {filename}")

    def get_values(self, field_name, sheet_names=None, default_value=0, not_found_error=False):
        ret = []
        if sheet_names is None:
            sheet_names = self.data.keys()
        for name in sheet_names:
            if field_name in self.data[name].columns:
                for key, value in self.data[name][field_name].iteritems():
                    if field_name in self.datetime_columns:
                        ret.append(value.to_datetime64())
                    else:
                        ret.append(value)
            else:
                if not_found_error:
                    raise RuntimeError(f"Column {field_name} not present in data loaded from excel file")
                else:
                    ret += [default_value for _ in range(len(self.data[name]))]
        return ret


class ParticleXLSSampler(ParticleXLSLoader):
    
    # Function to generate particles from the data in the Excel spreadsheet
    
    """
    Classe especialitzada en generar particules a partir de les dades obtingudes de l'excel.
    Especialitza la classe pare afegint tota la part de generacio de dades a partir de la columna 'Total items/h'
    """

    def __init__(self, filename, sheet_names, generators_coords, header, particle_proportion, init_time, final_time):
        """
        :param filename: Excel file to load
        :param sheet_names: array of strings of the sheets to load
        :param generators_coords: array of tuples, one pair for each sheet
        :param header: row position of headers in excel (0-indexed)
        :param particle_proportion: float [0,1] to define the proportion of particles generated in front of total
        """
        super().__init__(filename, sheet_names, header)
        self.rnd_gen = np.random.default_rng(seed=1)
        self.lat = []
        self.lon = []
        self.time = []
        for i, name in enumerate(sheet_names):
            for index, row in self.data[name].iterrows():
                if row[cfg.particle_col_name] > 0:
                    init_day = row[cfg.datetime_col_name]
                    if init_day >= init_time and init_day <= final_time:
                        center_lat = generators_coords[i][0]
                        center_lon = generators_coords[i][1]
                        
                        # resolves error of scale if the values are negative (W or S)
                        center_lat_scale = abs(generators_coords[i][0])
                        center_lon_scale = abs(generators_coords[i][0]) 
                        
                        # add center_lat_scale and center_lon_scale to the scale+ argument if need be.
                        if cfg.particle_frequency == 'daily':
                            lat = self.rnd_gen.normal(loc=center_lat, scale=center_lat / 500000, size=round(row[cfg.particle_col_name]) * 1).astype(np.float32).tolist()
                            self.lat += lat
                            lon = self.rnd_gen.normal(loc=center_lon, scale=center_lon / 10000, size=round(row[cfg.particle_col_name]) * 1).astype(float).tolist()
                            self.lon += lon
                        elif cfg.particle_frequency == 'hourly':
                            lat = self.rnd_gen.normal(loc=center_lat, scale=center_lat / 500000, size=round(row[cfg.particle_col_name]) * 24).astype(np.float32).tolist()
                            self.lat += lat
                            lon = self.rnd_gen.normal(loc=center_lon, scale=center_lon / 10000, size=round(row[cfg.particle_col_name]) * 24).astype(float).tolist()
                            self.lon += lon                           
                        t = []                        
                        if cfg.particle_frequency == 'daily':
                            for j in range(1):
                                t += [datetime(year=init_day.year, month=init_day.month, day=init_day.day, hour=j)] * round(row[cfg.particle_col_name])
                        elif cfg.particle_frequency == 'hourly':
                            for j in range(24):
                                t += [datetime(year=init_day.year, month=init_day.month, day=init_day.day, hour=j)] * round(row[cfg.particle_col_name])

                        assert len(lat) == len(lon), 'Arrays are different sizes'
                        assert len(lon) == len(t), 'Arrays are different sizes'
                        self.time += t
        # Filtrat de particules en funcio de particle_proportion
        lat = np.asarray(self.lat)
        lon = np.asarray(self.lon)
        t = np.asarray(self.time)
        list_where = self.rnd_gen.choice([True, False], size=len(self.lat),
                                         p=[particle_proportion, 1 - particle_proportion])
        self.lat = lat[list_where].tolist()
        self.lon = lon[list_where].tolist()
        self.time = t[list_where].tolist()
        logger.info(f"Number of particles created: {len(self.lon)}")



class ParticleXLSCreator(ParticleXLSLoader):
    
    # Function to generate particles from an Excel file using lat and lon columns

    def __init__(self, filename, sheet_names, header, particle_proportion, init_time, final_time):

        super().__init__(filename, sheet_names, header)
        self.rnd_gen = np.random.default_rng(seed=1)
        self.lat = []
        self.lon = []
        self.time = []
        for i, name in enumerate(sheet_names):
            for index, row in self.data[name].iterrows():
                if row[cfg.particle_col_name] > 0:
                    init_day = row[cfg.datetime_col_name] # this now has the time included in the format YY/mm/dd hh:mm:ss

                    if init_day >= init_time and init_day <= final_time:
                        center_lat = float(row['Latitude'])
                        center_lon = float(row['Longitude'])
                        # resolves error of scale if the values are negative (W or S)
                        center_lat_scale = abs(float(row['Latitude']))
                        center_lon_scale = abs(float(row['Longitude'])) 
                        # removed * 
                        # there seems to be a problem with the longitude when it is a negative (W) so -0.5 would throw an error when creating the particles
                        lat = self.rnd_gen.normal(loc=center_lat, scale=center_lat_scale / 500000, size=round(row[cfg.particle_col_name])).astype(np.float32).tolist()
                        self.lat += lat
                        lon = self.rnd_gen.normal(loc=center_lon, scale=center_lon_scale / 10000, size=round(row[cfg.particle_col_name])).astype(float).tolist()
                        self.lon += lon
                                              
                        t = []                        
                        t += [datetime(year=init_day.year, month=init_day.month, day=init_day.day, hour=init_day.hour, minute=init_day.minute)] * round(row[cfg.particle_col_name])
                        
                        assert len(lat) == len(lon), 'Arrays are different sizes'
                        assert len(lon) == len(t), 'Arrays are different sizes'
                        self.time += t
        # Filtrat de particules en funcio de particle_proportion
        lat = np.asarray(self.lat)
        lon = np.asarray(self.lon)
        t = np.asarray(self.time)
  
        list_where = self.rnd_gen.choice([True, False], size=len(self.lat),
                                         p=[particle_proportion, 1 - particle_proportion])
        self.lat = lat[list_where].tolist()
        self.lon = lon[list_where].tolist()
        self.time = t[list_where].tolist()
        logger.info(f"Numer of particles created: {len(self.lon)}")



def plot_trajectory_distance(datos_xr):
    # Shows the distace travelled by the particle trajectories
    # First plot is distance (m) x obs number
    # Second plot is distance (m) x time
    
    x = datos_xr.lon.values
    y = datos_xr.lat.values
    distance = np.cumsum(np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y))), axis=1)
    real_time = datos_xr.time
    # Distancia x Observaciones
    time_since_release = (real_time.values.transpose() - real_time.values[:, 0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1.set_ylabel('Distance travelled [m]')
    ax1.set_xlabel('observation', weight='bold')
    # Distancia x Tiempo
    d_plot = ax1.plot(distance.transpose())
    ax2.set_ylabel('Distance travelled [m]')
    ax2.set_xlabel('time', weight='bold')
    d_plot_t = ax2.plot(real_time.T[1:], distance.transpose())
    plt.show()

 

def plot_trajectories(sim_data, field=None, domain=None, terrain=None, terrain_zoom=10, arrow_density=1 / 16,
                      arrows_paths=True, graph_type='lines', show_plot=False):    
   
    # graph type: lines, points
    land = True if terrain is None else False
    if field is not None:
        # Plot the field underneath the particles
        plt, fig, ax, cartopy = plotfield(field, domain=domain, titlestr="Trajectories ", land=land,
                                          arrow_density=arrow_density)  # vectors
    else:
        # Homogeneous background
        plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
        if domain is not None:
            new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
            ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
    # set figure size
    fig.set_size_inches(13, 6)
    fig.suptitle(cfg.figure_title, fontsize=16)
    plt.title(cfg.plot_title)

    lon = np.ma.filled(sim_data['lon'], np.nan)
    lat = np.ma.filled(sim_data['lat'], np.nan)
    t = np.ma.filled(sim_data['time'], np.nan)
    
    # zarr cannot seem to read attributes, or they are not written to the zarr file
    #mesh = sim_data.attrs['parcels_mesh'] if 'parcels_mesh' in sim_data.attrs else 'spherical'
    mesh = 'spherical'
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    
    if graph_type=='Points':
        outputdt = timedelta(hours=6) # default is 1. Changing this timedelta changes the frequency of the dots
        timerange = np.arange(np.nanmin(sim_data['time'].values),
                               np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                               outputdt)
        for idt in range(len(timerange)):
             time_id = np.where(sim_data['time'] == timerange[idt])
             ax.scatter(sim_data['lon'].values[time_id], sim_data['lat'].values[time_id], s=5) # s is the size of the dot
    
    else: # default is lines
        logger.info(f"Plotting trajectories using lines {'and arrows' if arrows_paths else ''}")
        color_index = np.arange(lon.shape[0])

        # Convert to RGB to use same colours as the scatter plot      
        phase = cmocean.cm.phase
        cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)
        
        for idx in color_index:
            colorVal = scalarMap.to_rgba(idx)
            ax.plot(lon[idx], lat[idx], '-', color=colorVal, lw=1)

            # shows arrows in Paths. Modify lw and width for thicker lines and arrows
            # if arrows_paths:
            if cfg.paths_with_arrows == True:
                for i in range(len(lon[idx]) - 1):
                    plt.arrow(lon[idx, i], lat[idx, i], (lon[idx, i + 1] - lon[idx, i]) / 2,
                              (lat[idx, i + 1] - lat[idx, i]) / 2, shape='full', length_includes_head=False, lw=0,
                              width=0.0005, color=colorVal)
        

    
        
    # Show the plot
    ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    
    # show one label of simulated particle trajectories
    ax.plot(lon[idx], lat[idx], '-', color=colorVal, lw=1, label = cfg.sim_trajectory_label)


    if show_plot == True:
        plt.show()
        
    plt.legend()
    
    # save the plot at hig hresolution by default    
    os.makedirs(cfg.plot_path, exist_ok=True)
    plt.savefig(cfg.plot_path + cfg.sim_name + '.jpg', format='jpg', dpi=300)

    


def plot_animation(sim_data, field=None, domain=None, terrain=None, terrain_zoom=10, arrow_density=1 / 16,
                   filename=None, show_plot=False):
    land = True if terrain is None else False
    if field is not None:
        
        # Field underneath the paths
        plt, fig, ax, cartopy = plotfield(field, domain=domain, titlestr="Trajectories ", land=land, arrow_density=arrow_density)  # vectores
    else:
        # Homogeneous background
        plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
        if domain is not None:
            new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
            ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
        
    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    mesh = sim_data.attrs['parcels_mesh'] if 'parcels_mesh' in sim_data.attrs else 'spherical'
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Convert to RGB to use same colours as the scatter plot
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(hours=1)
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                          np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                          outputdt)
    time_id = np.where(sim_data['time'] == timerange[0])  # Indices of the data where time = 0
    scatter = ax.scatter(sim_data['lon'].values[time_id], sim_data['lat'].values[time_id], s=10)
    t = np.datetime_as_string(timerange[0], unit='h')
    title = ax.set_title('Particles at t = ' + t)

    def animate(i):
        t = np.datetime_as_string(timerange[i], unit='h')
        title.set_text('Particles at t = ' + t)
        time_id = np.where(sim_data['time'] == timerange[i])
        scatter.set_offsets(np.c_[sim_data['lon'].values[time_id], sim_data['lat'].values[time_id]])

    anim = FuncAnimation(fig, animate, frames=len(timerange), interval=500)
    if show_plot:
        plt.show()
    if filename is not None:
        logger.info(f"Saving animation video to {filename}")
        writervideo = FFMpegWriter(fps=30)
        
        if cfg.plot_path:
            path = cfg.plot_path 
            os.makedirs(path, exist_ok=True)
        else:
            path = ''
        
        anim.save(path + filename, writer=writervideo)
        logger.info("Video saved")



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def plot_concentration_heatmap(sim_data, domain=None, terrain=None, terrain_zoom=10, sim_steps=24, show_plot=False):
    land = True if terrain is None else False
    
    # Homogeneous background
    plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
    if domain is not None:
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
        ax.stock_img()


    # set figure size
    fig.set_size_inches(13, 6)

    # removed variables as does not work with .zarr
    lon = np.ma.filled(sim_data['lon'], np.nan)
    lat = np.ma.filled(sim_data['lat'], np.nan)
    t = np.ma.filled(sim_data['time'], np.nan)
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Convert to RGB to use same colours as the scatter plot
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(hours=1)

    # removed conversion of time as integer. 
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                           np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                           outputdt)
    
    
    timewindow = (timerange[-2],timerange[-1])
    
    time_id = np.where(np.logical_and(sim_data['time'] >= timewindow[0], sim_data['time'] < timewindow[1])) 
    
    lon = sim_data['lon'].values[time_id]
    lat = sim_data['lat'].values[time_id]
    lon = lon.flatten()
    lon = lon[np.logical_not(np.isnan(lon))]
    lat = lat.flatten()
    lat = lat[np.logical_not(np.isnan(lat))]

    width_km = haversine(domain["W"], domain["N"], domain ["E"], domain["N"])
    height_km = haversine(domain["W"], domain["N"], domain ["W"], domain["S"])
    print(f'Map width={width_km:.3f} km')
    print(f'Map height={height_km:.3f} km')
    
    nbins=(round(1*width_km), round(1*height_km))
    print(f'Number of bins for 2d histogram: {nbins}')
    data, x_e, y_e = np.histogram2d(lon, lat, bins=nbins, range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])

    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([lon, lat]).T, method="splinef2d", bounds_error=False)
    idx = z.argsort()
    x,y,z = lon[idx], lat[idx], z[idx]
    scatter = ax.scatter(x , y, c=z)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel(f'Num particles per km$^2$')

    t = np.datetime_as_string(sim_data['time'].values[time_id][-1], unit='h')
    title = ax.set_title('Particles density at t = ' + t)
    ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)

    if show_plot == True:
        plt.show()
        
    # save figure as high resolution plot
    os.makedirs(cfg.plot_path, exist_ok=True)
    plt.savefig(cfg.plot_path + cfg.sim_name + '_heatmap.jpg', format='jpg', dpi=300)
    

