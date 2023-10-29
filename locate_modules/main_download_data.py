"""
Created: 2021/03/29
@author: Jose M. Alsina (UPC)
         with technical support from Deep Solutions (David Pérez, Oscar Serra)
         Script to Download wave and current data for Lagrangian simulations (LOCATE model)
         Warning IBI data have time set in hour and 30 minutes i.e.: 12:30, 13:30 whereas coastal and harbour data
         have exact hour timing i.e. 12:00 13:00 . time interpolation is needed and performed in UPC_resample_dataset
"""

#############################################################
#############################################################
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import datetime
from utils.locate_operations import UPC_config as cfg


def Download_CMEMS_currents_v2(motuclient: str, product: str, product_id: str, geowin: str, nom: str, result0: list,
                               result: list):
    print(" -----------------------------------------------------------")
    print("----- Starting to download CMEMS MYOcean data products -----")
    print(" -----------------------------------------------------------")

    ibimfc_var_hourly = "--variable vo --variable uo --variable zos --variable thetao"  # var horaries

    data_dir = Path(cfg.data_base_path) / Path(cfg.IBI_files_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # crea tot el path i nomes si fa falta

    name_sortida = 'MyO-IBI_hm_' + nom + '_' + result0[0] + '_B' + result0[1] + '_HC01.nc'
    stringg = f'{motuclient}{product}{product_id}{geowin} --date-min "{result[0]} 00:30:00" --date-max "{result[1]} 23:30:00" {ibimfc_var_hourly} --out-dir {data_dir} --out-name {name_sortida} --user {os.environ["CMEMS_USER"]} --pwd {os.environ["CMEMS_PASSWD"]}'
    os.system('python' + stringg + ' > file_logging/download_success/CMEMS_problem.txt')

    # Check the download was successful
    f = open('file_logging/download_success/CMEMS_problem.txt', 'r')
    lineList = f.readlines()
    f.close()
    A = lineList[len(lineList) - 1]
    if A[32:36] == 'Done':  # tot ok
        print('File ' + name_sortida + ' downloaded....')
    else:  # something went wrong
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
        print('File NOT downloaded.....check CMEMS_problem.txt file!')
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')



def Download_CMEMS_waves_v2(motuclient: str, product: str, product_id: str, geowin: str, nom: str, result0: list,
                            result: list):
    print(' -----------------------------------------------------------')
    print('----- Starting to download CMEMS WAVE data products -----')
    print(' -----------------------------------------------------------')

    ibimfc_var_hourly = "--variable VHM0 --variable VMDR --variable VSDX --variable VSDY"

    data_dir = Path(cfg.data_base_path) / Path(cfg.Wave_IBI_files_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # crea tot el path i nomes si fa falta

    name_sortida = 'WAV-IBI_hm_' + nom + '_' + result0[0] + '_B' + result0[1] + '_HC01.nc'
    stringg = f'{motuclient}{product}{product_id}{geowin} --date-min "{result[0]} 00:30:00" --date-max "{result[1]} 23:30:00" {ibimfc_var_hourly} --out-dir {data_dir} --out-name {name_sortida} --user {os.environ["CMEMS_USER"]} --pwd {os.environ["CMEMS_PASSWD"]}'
    os.system('python' + stringg + ' > file_logging/download_success/WAVCMEMS_problem.txt')

    # Check the download was successful
    f = open('file_logging/download_success/WAVCMEMS_problem.txt', 'r')
    lineList = f.readlines()
    f.close()
    A = lineList[len(lineList) - 1]
    if A[32:36] == 'Done':  # tot ok
        print('File ' + name_sortida + ' downloaded....')
    else:  # something went wrong
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
        print('File NOT downloaded.....check WAVCMEMS_problem.txt file!')
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
        print(' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')


def download_main(date_start,
                  simulation_length,
                  CMEMS_download_limits,
                  data_base_path=cfg.data_base_path,  # this parameter could be made compulsory in the future
                  ):
    # Load CMEMS credentials
    load_dotenv()
    assert 'CMEMS_USER' in os.environ.keys(), "cmems_user variable not loaded in environment. Please set the variable or create a .env file with cmems_user and cmems_passwd"

    dt = date_start  # first day to download
    dtend = date_start + datetime.timedelta(
        days=simulation_length - 1)  # subtract one to download only simulation_length days
    tlimit = datetime.datetime(1993, 1, 1)  # 1/1/1993
    dtT = datetime.datetime(2020, 1, 1)  # ] 4/7/20019

    # dtT is the threshold time which is 2019-05-05:< (limitem hay una fecha a aprtir de la cual los datos s ealamcenan en un archivo u otro)
    # after that date the current data are storaged as IBI_ANALYISFORECAST_PHY_005_001 and waves as IBI_ANALYSIS_FORECAST_WAV_005_005
    # before that day the data are storage as IBI_MULTIYEAR_PHY_005_002 and waves as IBI_MULTIYEAR_WAV_005_006
    # This limit seems to change with time
    # before 1993-01-01 there is no data

    if dt >= dtT and dtend >= dtT:  # most recent storage (last 2 years)
        MyO_prod = 'IBI_ANALYSISFORECAST_PHY_005_001-TDS '
        MyO_product_id = "--product-id cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m "
        MyO_motuclient = ' -m motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id '
        WyO_prod = 'IBI_ANALYSIS_FORECAST_WAV_005_005-TDS '
        WyO_product_id = '--product-id dataset-ibi-analysis-forecast-wav-005-005-hourly '
    elif dt < dtT and dtend < dtT:  # data older than 2 years
        MyO_prod = 'IBI_MULTIYEAR_PHY_005_002-TDS '
        MyO_product_id = "--product-id cmems_mod_ibi_phy_my_0.083deg-2D_PT1H-m "
        MyO_motuclient = ' -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id '
        WyO_prod = 'IBI_MULTIYEAR_WAV_005_006-TDS '
        WyO_product_id = '--product-id cmems_mod_ibi_wav_my_0.05deg-2D_PT1H-i '
    elif dt < tlimit and dtend < tlimit:
        sys.exit('Error! there is no wave or current data before 1993')
    else:
        sys.exit(
            'Error! Your initial and end date corresponds to different data periods, think about a dividing your data downloading')

    ###############################################################################
    # DEFINEIX EL DOMINI QUE S'HA DE DESCARREGAR
    ###############################################################################
    lat_min, lat_max, lon_min, lon_max = CMEMS_download_limits
    ibimfc_geowin = f"--longitude-min {lon_min} --longitude-max {lon_max} --latitude-min {lat_min} --latitude-max {lat_max}"
    nom = 'BCN'

    # Preparem vector dies que servira per fer bucle descarrega

    result = [dt.strftime('%Y-%m-%d'), dtend.strftime('%Y-%m-%d')]
    result0 = [dt.strftime('%Y%m%d'), dtend.strftime('%Y%m%d')]

    ################
    # Creació del directori de data
    data_path = Path(data_base_path)
    data_path.mkdir(parents=True, exist_ok=True)

    print("result = ", result)
    print("result0 = ", result0)

    # Download currents data
    Download_CMEMS_currents_v2(MyO_motuclient, MyO_prod, MyO_product_id, ibimfc_geowin, nom, result0, result)
    # Download waves data
    Download_CMEMS_waves_v2(MyO_motuclient, WyO_prod, WyO_product_id, ibimfc_geowin, nom, result0, result)


if __name__ == '__main__':
    download_main(date_start=datetime.datetime(2022, 2, 3),
                  simulation_length=1,
                  CMEMS_download_limits=[38.1, 42.75, 0, 4.25])
