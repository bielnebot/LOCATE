# -*- coding: utf-8 -*-
"""
Created: 2021/03/29
@author: Jose M. Alsina (UPC) 
         with technical support from Deep Solutions (David Pérez, Oscar Serra)
"""

from config import sample_data as cfg
#import UPC_config as cfg

import numpy as np
import pandas as pd
import xarray as xr
import math
from matplotlib import pyplot as plt
import os
import progressbar
import datetime
from dask.diagnostics import ProgressBar

from skimage import measure
from matplotlib import path
from numpy import ma



def extiende_vector(v, minimo, maximo):
    pre = []
    incr = v[1]-v[0]
    cur = v[0] - incr
    while cur > minimo:
        pre.append(cur)
        cur = cur - incr
    pre.reverse()
    post = []
    cur = v[-1] + incr
    while cur < maximo:
        post.append(cur)
        cur = cur + incr
    return np.asarray(pre + v.tolist() + post)

def mix_xarrays(a,b, latitude:str='latitude', longitude:str='longitude', variables:list=['u', 'v']):
    """
    Superimposes the xarray data with spatial data (lat,lon)
    Does not take into account the time dimension, so it must be the same in both arrays
    The arrays can have different resolutions with the highest resolution one takiing priority
    Resolution: area(lat-lon)/num_data

    :param a: xr.DataArray o xr.Dataset with spatial data
    :param b: xr.DataArray o xr.Dataset with spatial data
    :param latitude: lat coordinate str label
    :param longitude: lon coordinate str label
    :param variables: list[str] with the names of variables to combine

    :return: NEW xarray with combined data.
    """
    
    # Set which array has a larger resolution and obtain the data fromthe maximum resolution
    res_a = (a[latitude][-1]-a[latitude][0])*(a[longitude][-1]-a[longitude][0])
    res_a = (a[latitude].size * a[longitude].size) / res_a
    res_b = (b[latitude][-1]-b[latitude][0])*(b[longitude][-1]-b[longitude][0])
    res_b = (b[latitude].size * b[longitude].size) / res_b
    if res_b>res_a:
        hi_res = b
        lo_res = a
    else:
        hi_res = a
        lo_res = b

    interp_lo = lo_res.interp(latitude=hi_res[latitude].data, longitude=hi_res[longitude].data, method='linear') #linear and nearest methods return arrays including NaN, while other methods such as cubic or
    for v in variables:
        ind_nan = np.isnan(hi_res[v].data)
        hi_res[v].data[ind_nan] = interp_lo[v].data[ind_nan]
    return hi_res

def mix_xarrays2(coastal, ibi, latitude:str= 'latitude', longitude:str= 'longitude', variables:list=['u', 'v']):
    """
    Mix information from two xarrays with spatial infromation (lat,lon)
    It does not account for the time dimension (time), therefore time needs to be the same in both arrays 
    Required if no interpolation (prepare files)

    :param coastal: xr.DataArray or xr.Dataset with spatial data
    :param ibi: xr.DataArray or xr.Dataset with spatial data 
    :param latitude: str label with infromation of the latitude coordinate 
    :param longitude: str label with infromation of the longitude coordinate
    :param variables: list[str] with names of the varibales to combine 

    :return: NEW xarray with combined data.
    """
    # get the shoreline configuration from initial velocity
    # we use the velocity modulus tod efine the limit. Load the initial data
    lon = np.array(coastal.longitude)
    lat = np.array(coastal.latitude)
    
    U = np.array(coastal.u[0,0,:,:])
    V = np.array(coastal.v[0,0,:,:])
    Umod = np.sqrt(np.power(U,2) + np.power(V,2))
    # Define the shoreline coordinate
    shore_mask, shore = Define_landidentifier(lon, lat, Umod)
   
    
    lo_res = ibi.compute() #we add the compute for indexing, see below. Otherwise we would need to make a loop covering the dataset chunks 
    hi_res = coastal.compute()
    interp_lo = lo_res.interp(latitude=hi_res[latitude].data, longitude=hi_res[longitude].data, method='nearest') #linear and nearest methods return arrays including NaN, while other methods such as cubic or quadratic return all NaN arrays
    
    # Search for values thata re not nana for the variables indicated in the list and assign the low resolution values by interpolation 
    for v in variables:
        ind_nan = np.isnan(hi_res[v].data)
        # Assigment here        
        hi_res[v].data[ind_nan] = interp_lo[v].data[ind_nan]
        # Here we use the shoreline mask to define shoreline values after interpolation 
        hi_res[v].data[:,:,shore_mask[0],shore_mask[1]]=0
        hi_res[v].data[:,:,shore[:,0],shore[:,1]]=0
    #return interp_lo
    return hi_res


def mix_files(files_a, files_b, new_dir):
    """
    Opens two folders with spatial data and superimposes them generating new folders in the specified directory
    Time dimension and the number of folders must be the same in both directories

    :param files_a: list(str) list of paths in folders a
    :param files_b: list(str) list of paths in folders b
    :param new_dir: str creates destination directory if does not exist

    """
    new_dir = Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    assert len(files_b)==len(files_a), "Different number of folders in the lists"
    print("Mixing files...")
    for i in progressbar.progressbar(range(len(files_a))):
        a = xr.open_dataset(files_a[i])
        b = xr.open_dataset(files_b[i])
        c = mix_xarrays(a, b, latitude='latitude', longitude='longitude', variables=['u', 'v'])
        base_filename_a = os.path.splitext(os.path.split(files_a[i])[1])[0]
        base_filename_b = os.path.splitext(os.path.split(files_b[i])[1])[0]
        res_filename = f'overlap_{base_filename_a}_{base_filename_b}.nc'
        c.to_netcdf(os.path.join(new_dir, res_filename))
    print(f'Job done. Files can be found at {os.path.abspath(new_dir)}')


def resample_files(files_a, files_b, new_dir):
    """
    Saves the new resampled variables in a new dirctory

    :param files_a: list(str) list of paths of folders a
    :param new_dir: str creates destination directory if does not exist

    """
    if not(os.path.isdir(new_dir)):
        os.makedirs(new_dir, exist_ok=True)
    #assert len(files_b)==len(files_a), "Numero de ficheros distinto en las listas"
    print("Storing resampled files in netcdf files...")
    for i in progressbar.progressbar(range(len(files_a))):
        a = xr.open_dataset(files_a[i])
        b = xr.open_dataset(files_b[i])
        c = mix_xarrays(a, b, latitude='latitude', longitude='longitude', variables=['u', 'v'])
        base_filename_a = os.path.splitext(os.path.split(files_a[i])[1])[0]
        base_filename_b = os.path.splitext(os.path.split(files_b[i])[1])[0]
        #res_filename = f'overlap_{base_filename_a}_{base_filename_b}.nc'
        res_filename = f'{base_filename_a}_resampled.nc'
        c.to_netcdf(os.path.join(new_dir, res_filename))
    print(f'Job done. Files can be found at {os.path.abspath(new_dir)}')

def resample_files_ibi(filenames_prt, filenames_cst, filenames_ibi, new_dir_ibi, new_dir_ci, new_dir_hc):
    """
    Keeps the three datasets in separate directories
    IBI: dataset grouped in folders by day and resampled to coastal dates
    ci: dataset grouped by day, corresponds to coastal filling NaNs with IBI data
    hc: dataset grouped by day, corresponds to coastal filling NaNs with ci data

    :param filenames_prt: list(str) list of paths in harbour folder
    :param filenames_cst: list(str) list of paths in coastal folder
    :param filenames_ibi:[str] list of paths in IBI folder
    :param new_dir_ibi: str directory where IBI files to be saved
    :param new_dir_ci: str directory where ci files to be saved
    :param new_dir_hc: str directory where hc files to be saved
    """
    print("Resampling datasets...")
    dir_ibi = Path(new_dir_ibi)
    dir_ci = Path(new_dir_ci)
    dir_hc = Path(new_dir_hc)
    dir_ibi.mkdir(parents=True, exist_ok=True)
    dir_ci.mkdir(parents=True, exist_ok=True)
    dir_hc.mkdir(parents=True, exist_ok=True)
    # Abre coastal y IBI
    coastal_mf = xr.open_mfdataset(filenames_cst, parallel=True)   # loads faster with multiproc
    ibidat_mf = xr.open_mfdataset(filenames_ibi, parallel=True)
    # Prepara IBI
    ibi_prep = prepare_ibi(ibidat_mf, coastal_mf)
    plt.pcolormesh(ibi_prep.longitude, ibi_prep.latitude, ibi_prep.u[0, 0, :, :])
    # Save IBI
    _, datasets_ibi = zip(*ibi_prep.groupby("time.day")) # groupby returns tuples (<dia>,<dataset>), paso del dia
    paths_ibi = []
    paths_ci = []
    paths_hc = []
    for dataset in datasets_ibi:
        first_date = pd.to_datetime(dataset.time[0].data).date()
        paths_ibi.append(dir_ibi/f'IBI_{first_date}_resampled.nc')
        paths_ci.append(dir_ci/f'coastal_IBI_{first_date}_resampled.nc')
        paths_hc.append(dir_hc/f'harbour_coastal_{first_date}_resampled.nc')
    # paths = [f'ibi_{day}_resampled.nc' for day in days]
    with ProgressBar():
        print("Saving IBI resampled files")
        xr.save_mfdataset(datasets_ibi, paths_ibi)
    # Fill coastal empty grids with IBI data and deallocate memory
    ci = mix_xarrays2(coastal_mf, ibi_prep, latitude='latitude', longitude='longitude', variables=['u', 'v'])
    plt.pcolormesh(ci.longitude, ci.latitude, ci.u[0, 0, :, :])
    del ibidat_mf
    del datasets_ibi
    # Resample coastal with IBI
    _, datasets_ci = zip(*ci.groupby("time.day"))
    with ProgressBar():
        print("Saving Coastal-IBI mixed files")
        xr.save_mfdataset(datasets_ci, paths_ci)
    del datasets_ci
    # Load harbour data
    harbour_mf = xr.open_mfdataset(filenames_prt, parallel=True)
    # Mix harbour y coastal files
    hc = mix_xarrays2(harbour_mf, coastal_mf, latitude='latitude', longitude='longitude', variables=['u', 'v'])
    plt.pcolormesh(hc.longitude, hc.latitude, hc.u[0, 0, :, :])
    plt.title("Three layers data in one image")
    # Save harbour and coastal data (resamples) in the same loop 
    _, datasets_hc = zip(*hc.groupby("time.day"))
    with ProgressBar():
        print("Saving Harbour-Coastal mixed files")
        xr.save_mfdataset(datasets_hc, paths_hc)
    print(f'Job done. Files can be found at {os.path.abspath(dir_ibi)}, {os.path.abspath(dir_ci)} and {os.path.abspath(dir_hc)}')


def prepare_ibi(dataset, dataset_no_ibi):
    time_prep = dataset.interp(time=dataset_no_ibi.time)
    name_prep = time_prep.rename({'vo':'v', 'uo':'u'})
    name_prep = name_prep.assign_coords(depth=1.0)
    name_prep = name_prep.expand_dims('depth')
    name_prep = name_prep.transpose('time','depth','latitude','longitude')
    return name_prep

def get_right_files(data_dir, file_search_string):
    
    #get Dates
    dl_i = cfg.download_init.split("/")
    dl_e = cfg.download_end.split("/")
    
    YYi = [int(dl_i[0]),int(dl_i[1]),int(dl_i[2])]
    YYe = [int(dl_e[0]),int(dl_e[1]),int(dl_e[2])]
    
    dt0 = datetime.datetime(YYi[0],YYi[1],YYi[2])
    dtend = datetime.datetime(YYe[0],YYe[1],YYe[2])
    filenames_list = pd.Series(os.listdir(data_dir))
    filenames_order = []
    for date in pd.date_range(dt0, dtend, freq = 'D').strftime('%Y%m%d'):
        for filenames in filenames_list[filenames_list.str.contains(date)]:
                filenames_order.append( Path(data_dir)/Path(filenames))        
    return filenames_order


def Define_landidentifier(lon,lat,Umod):
    # first we need to read the velocity field
    Umod[np.isnan(Umod)] = 0
    contours =  measure.find_contours(Umod, 0)

    ishow = False
    """
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax.plot(lon[contour[:, 1].astype(int)], lat[contour[:, 0].astype(int)], linewidth=2)
    """
    shore_aux = contours[0]
    index_min = np.argmin(lon[shore_aux[:, 1].astype(int)])
    # find edge points
    leftup = (np.where(lat == np.amax(lat) ), np.where( lon == lon[shore_aux[index_min, 1].astype(int)] ))
    leftup = np.array(leftup)
    leftup = np.reshape(leftup,(1,2))
    rightup = (np.where(lat == np.amax(lat) ), np.where(lon == lon[shore_aux[0, 1].astype(int) ] )  )  
    rightup = np.array(rightup)
    rightup = np.reshape(rightup,(1,2))

    # append the edge point to the shoreline
    shore =  shore_aux[0:index_min, :]
    shore = shore.astype(int)
    shore = np.append(shore,leftup,axis=0)
    shore = np.append(shore,rightup,axis=0)
    shore = np.append(shore,shore[0:1,0:2],axis=0)
    
    if ishow:
        fig, ax = plt.subplots()
        ax.pcolormesh(lon, lat, Umod) #port
        ax.plot(lon[shore[:, 1]], lat[shore[:, 0]] , linewidth=2)
        plt.show()
    
      
    closed_path = path.Path(shore, closed = True)
    
    
    idx = np.indices(Umod.shape)
    idx2 = idx.reshape(2,len(lon)*len(lat))
    inside = closed_path.contains_points(idx2.T)
    inside2 = inside.reshape(Umod.shape)
    Umod2 = Umod.copy()
    Umod2[inside2] = -1
    inside3= np.where(inside2)

   
    if ishow:
        fig1,(ax1,ax2)=plt.subplots(ncols=2, figsize=(6,3))
        cf = ax1.pcolormesh(lon, lat, Umod) #port
        fig.colorbar(cf, ax=ax1)
        cf2 = ax2.pcolormesh(lon, lat, Umod2) #port
        fig.colorbar(cf2, ax=ax2)
        plt.show()
    
    shoreline = shore
    in_mask = inside3
    return in_mask, shoreline


if __name__ == '__main__':
    from pathlib import Path

    # Directory paths
    data_dir_prt = Path(cfg.data_base_path) / Path(cfg.port_files_dir) # Port foles
    data_dir_cst = Path(cfg.data_base_path) / Path(cfg.coastal_files_dir) # Coastal files
    data_dir_reg = Path(cfg.data_base_path) / Path(cfg.regional_files_dir) # Regional files
    
    filenames_prt = sorted(data_dir_prt.glob(cfg.port_file_search_string)) # files
    filenames_prt_order = get_right_files(data_dir_prt, cfg.port_file_search_string)
    filenames_cst = sorted(data_dir_cst.glob(cfg.coastal_file_search_string))
    filenames_cst_order = get_right_files(data_dir_cst, cfg.coastal_file_search_string)
    filenames_reg = sorted(data_dir_reg.glob(cfg.regional_file_search_string))
    filenames_reg_order = get_right_files(data_dir_reg, cfg.regional_file_search_string)
    
    filenames_prt = filenames_prt_order
    filenames_cst = filenames_cst_order
    filenames_reg = filenames_reg_order
    
    
    # Plot mesh for initial files
    # IBI data initial file
    ibi1 = filenames_reg[0] # file inicial
    ibidat = xr.open_dataset(ibi1)
    ibidat_mf = xr.open_mfdataset(filenames_reg, parallel=True)
    # plt.pcolormesh(ibidat.longitude, ibidat.latitude, ibidat.uo[0, :, :]) 
    
    # Coastal data
    cst1 = filenames_cst[0] # file inicial
    coastal = xr.open_dataset(cst1)
    coastal_mf = xr.open_mfdataset(filenames_cst, parallel=True)
    # plt.pcolormesh(coastal.longitude, coastal.latitude, coastal.u[0, 0, :, :]) 
    
    # Harbour data
    prt1 = filenames_prt[0]
    port = xr.open_dataset(prt1)
    #plt.pcolormesh(port.longitude, port.latitude, port.u[0, 0, :, :]) 
    

    ibi_prepared = prepare_ibi(ibidat_mf, coastal_mf)
    d = mix_xarrays2(coastal_mf, ibi_prepared, latitude='latitude', longitude='longitude', variables=['u', 'v'])
    plt.pcolormesh(d.longitude, d.latitude, d.u[0, 0, :, :])

    # Creates new resampled folders
    new_dir_ibi = Path(cfg.data_base_path) / Path(cfg.regional_files_dir + '_resampled')
    new_dir_hc = Path(cfg.data_base_path) / Path(cfg.port_files_dir + '_resampled')
    new_dir_ci = Path(cfg.data_base_path) / Path(cfg.coastal_files_dir + '_resampled')
    resample_files_ibi(filenames_prt, filenames_cst, filenames_reg, new_dir_ibi, new_dir_ci, new_dir_hc)

    plt.show()
