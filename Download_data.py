# coding: utf8
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

from config import sample_data as cfg

import urllib.error
from urllib.request import urlretrieve
from dotenv import load_dotenv

print (' -----------------------------------------------------------')
print (' -----------------------------------------------------------')
import os
import sys
from pathlib import Path
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

                       


def Download_CMEMS_currents(motuclient:str, product:str, product_id:str, geowin:str, nom:str, result0:list, result:list ):
    """
    Downloads current data from CMEMS
    :param motuclient: string defining the call to motuclient
    :param product: string to specify wich product to download
    :param product_id: string to specify wich product to download
    :param geowin:
    :param nom:
    :param result0: list of days to download
    :param result: list of days to download
    :return:
    """
    # In this def we proceed to download the current data
    # variables to download change by CMEMS product      
    ibimfc_var_hourly = "--variable vo --variable uo --variable zos --variable thetao" 
    
    
    
    #-------------------
    ###################################################
    # 	The loop starts here 
    ###################################################

    # In this loop we download the files one by one accroding to results0
    kk=0 
    #   n=1
    print (' -----------------------------------------------------------')
    print ('----- Starting to download CMEMS MYOcean data products -----')
    print (' -----------------------------------------------------------')
    data_dir = Path(cfg.data_base_path) / Path(cfg.regional_files_dir)
    data_dir.mkdir(parents=True, exist_ok=True) # create path and names if necessary
    count = len(result0)
    for i in range(1,len(result0)):

        print ('------------------------------------------------------')
        print ('Downloading day ',i,' of ',count-1)
        dayres=i-1
        name_sortida_hourly= 'MyO-IBI_hm_'+nom+'_'+result0[dayres]+'_B'+result0[i]+'_HC01.nc'  
        data_bucle=result[dayres]

        name_sortida = name_sortida_hourly
        print ('-- Downloading HOURLY values for ',result[dayres])
        stringg = f'{motuclient}{product}{product_id}{geowin} --date-min "{data_bucle} 00:30:00" --date-max "{data_bucle} 23:30:00" {ibimfc_var_hourly} --out-dir {data_dir} --out-name {name_sortida} --user {os.environ["CMEMS_USER"]} --pwd {os.environ["CMEMS_PASSWD"]}'
        os.system('python'+stringg +' > CMEMS_problem.txt')
        
        ###############################################################################
        # Check the downloading using CMEMS_problem.txt
        ###############################################################################
   
        f = open('CMEMS_problem.txt', 'r')
        lineList = f.readlines()
        f.close()
        A=lineList[len(lineList)-1]
        if A[32:36]=='Done': # everything ok
            print ('File '+name_sortida+' downloaded....')
        else: # if there are problems
            kk=kk+1
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
            print ('File NOT downloaded.....check CMEMS_problem.txt file!')
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
    status = 0
    return status


def Download_CMEMS_waves(motuclient:str, product:str, product_id:str, geowin:str, nom:str, result0:list, result:list):
    """
    Downloads waves data from CMEMS
    :param motuclient: string defining the call to motuclient
    :param product: string to specify wich product to download
    :param product_id: string to specify wich product to download
    :param geowin:
    :param nom:
    :param result0: list of days to download
    :param result: list of days to download
    :return:
    """
    # In this def we proceed to download the wave data
    #------------------
    # variables to download
    ibimfc_var_hourly = "--variable VHM0 --variable VMDR --variable VSDX --variable VSDY" # var horaries
    # Variables for wave downloading are VHM0= significant wave height, VMDR mean wave direction, VSDX= Stokes drfit (x-componnet), VSDY= Stokes drfit (y-componnet) 
    #-------------------
    
    # create loop to download files
    
    # i=1 # si peta en algun punt el re-iniciem desde aqui --> mirar valor iteracio prompt
    kk=0 
    print (' -----------------------------------------------------------')
    print ('----- Starting to download CMEMS WAVE data products -----')
    print (' -----------------------------------------------------------')
    data_dir = Path(cfg.data_base_path) / Path(cfg.wave_regional_files_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # create path and names if necessary
    count = len(result0)
    for i in range(1, len(result0)):

        print ('------------------------------------------------------')
        print ('Downloading day ',i,' of ',count-1)
        dayres=i-1
        name_sortida_hourly= 'WAV-IBI_hm_'+nom+'_'+result0[dayres]+'_B'+result0[i]+'_HC01.nc' 
        data_bucle=result[dayres]

        name_sortida = name_sortida_hourly
        print ('-- Downloading HOURLY values for ',result[dayres])
        # stringg =  motuclient + product  +  product_id + geowin + ' --date-min ' + data_bucle +' 00:30:00" --date-max ' +data_bucle+' 23:30:00" ' + ibimfc_var_hourly + ' --out-dir ' + cfg.dir_IBIDATA + ' --out-name ' + name_sortida + ' --user' + cfg.user +  ' --pwd' + cfg.passwd
        stringg = f'{motuclient}{product}{product_id}{geowin} --date-min "{data_bucle} 00:30:00" --date-max "{data_bucle} 23:30:00" {ibimfc_var_hourly} --out-dir {data_dir} --out-name {name_sortida} --user {os.environ["CMEMS_USER"]} --pwd {os.environ["CMEMS_PASSWD"]}'
        os.system('python'+stringg +' > WAVCMEMS_problem.txt')

        f = open('WAVCMEMS_problem.txt', 'r')
        lineList = f.readlines()
        f.close()
        A=lineList[len(lineList)-1]
        if A[32:36]=='Done': # tot ok
            print ('File '+name_sortida+' downloaded....')
        else: # algun problema
            kk=kk+1
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
            print ('File NOT downloaded.....check WAVCMEMS_problem.txt file!')
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
            print (' !!!!! !!!!!! !!!!!! !!!!!! !!!!!!!!')
    status = 0
    return status

"""
!!!! IMPORTANT !!!!
If copying the files manually  if data is no longer on the opendap server then must include:
the day prior to the dload dates
the two days after the dload dates
"""

def Download_opendap_currents(chkdap:str, result0:list, data_path:Path):
    """
    Downloads coastal and local data from opendap.puertos.es
    :param chkdap: 'coastal' or 'harbour'
    :param result0: list of days to download
    :param data_path: directory base to save the data
    """
    # Divide the url you get from the data portal into two parts
    # Everything before "catalog/"
    server_url = 'http://opendap.puertos.es/thredds/'
    count = len(result0)
    # Everything after "catalog/"
    if chkdap == 'coastal':
        Prod = 'fileServer/circulation_coastal_bcn/'
        file_pref = 'BCNCST-PdE-hm-' #move to config
        file_suf = '-HC.nc' #move to config
        dir = data_path / Path(cfg.coastal_files_dir)
    elif chkdap == 'harbour':   
        Prod = 'fileServer/circulation_local_bcn/'
        file_pref = 'BCNPRT-PdE-hm-' #move to config
        file_suf = '-HC.nc' #move to config
        dir = data_path / Path(cfg.port_files_dir)
    else:
        sys.error(f'Invalid parameter value chkdap: {chkdap} (expected: "coastal" or "port"')
    dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while i < count:
        date = result0[i]
        print ('------------------------------------------------------')
        file_url = server_url + Prod + date[0:4] + '/' + date[4:6] + '/' + file_pref + result0[i] + file_suf
        # filename to save the data
        file_name = dir / Path(file_pref + result0[i] + file_suf)
        print (f'Downloading file {i} of {count} ({date[0:4]}/{date[4:6]}) ')
        print(file_name)
        try:
            urlretrieve(file_url,file_name)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"Unable to find file: {file_url}")
        i = i + 1
        
if __name__ == '__main__':

    ###################################################

    # Preloads to memory CMEMS authentication credentials (if not there)
    load_dotenv()
    assert 'CMEMS_USER' in os.environ.keys(), "cmems_user variable not loaded in environment. Please set the variable or create a .env file with cmems_user and cmems_passwd"


    dl_i = cfg.download_init.split("/")
    dl_e = cfg.download_end.split("/")
    
    YYi = [int(dl_i[0]),int(dl_i[1]),int(dl_i[2])]
    YYe = [int(dl_e[0]),int(dl_e[1]),int(dl_e[2])]

    # CMEMS earliest data available
    # Earliest date of data available for CMEMS product
    CMEMS_o = "1993-1-1" 
    CMEMS_o_spl = CMEMS_o.split("-")
    
    # reanalysis product cutoff date is 2 years from current date
    # server date is in the format yyyy-mm-dd
    current_date = date.today()
    reanalysis_date = current_date - relativedelta(years=2)
    reanalysis_date_str = str(reanalysis_date)
    CMEMS_r_spl = reanalysis_date_str.split("-")
    
    # process CMEMS first availabla data
    CMEMS_o_d = [int(CMEMS_o_spl[0]),int(CMEMS_o_spl[1]),int(CMEMS_o_spl[2])]

    # process CMEMS data reanalysis product date, 2 years from current date
    CMEMS_r_d = [int(CMEMS_r_spl[0]),int(CMEMS_r_spl[1]),int(CMEMS_r_spl[2])]


    dt = datetime.datetime(YYi[0],YYi[1],YYi[2])
    dtend = datetime.datetime(YYe[0],YYe[1],YYe[2])
    tlimit = datetime.datetime(CMEMS_o_d[0], CMEMS_o_d[1], CMEMS_o_d[2])
    dtT = datetime.datetime(CMEMS_r_d[0], CMEMS_r_d[1], CMEMS_r_d[2])
       



    # specify product depending on the reanalysis product cutoff date
    if dt >= dtT and dtend >= dtT: 
        MyO_prod = 'IBI_ANALYSISFORECAST_PHY_005_001-TDS '
        MyO_product_id = "--product-id cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m "
        MyO_motuclient = ' -m motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id '
        WyO_prod = 'IBI_ANALYSIS_FORECAST_WAV_005_005-TDS '
        WyO_product_id = '--product-id dataset-ibi-analysis-forecast-wav-005-005-hourly '
        
    elif  dt < dtT and dtend < dtT:
        MyO_prod = 'IBI_MULTIYEAR_PHY_005_002-TDS '
        MyO_product_id = "--product-id cmems_mod_ibi_phy_my_0.083deg-2D_PT1H-m "
        MyO_motuclient = ' -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id '
        WyO_prod = 'IBI_MULTIYEAR_WAV_005_006-TDS '
        WyO_product_id = '--product-id cmems_mod_ibi_wav_my_0.05deg-2D_PT1H-i '
        
    elif dt < tlimit and dtend < tlimit: 
        sys.exit('Error! there is no wave or current data before 1993')
    else :
        sys.exit('Error! Your initial and end date corresponds to different data periods, think about a dividing your data downloading')
    

    ###############################################################################
    # DEFINE DOMAIN TO DOWNLOAD
    ###############################################################################

    # Specify coordinates
    # Barcelona

    ibimfc_geowin='--longitude-min ' + str(cfg.domain_W) + ' --longitude-max ' + str(cfg.domain_E) + ' --latitude-min ' + str(cfg.domain_S) + ' --latitude-max ' + str(cfg.domain_N) + '' 
      #  ibimfc_lev_daily="-z 0.49 -Z 2000 " # vertical levels per daily file
    nom = cfg.domain_name    
    # Opendap data
    print ('Data from: ', nom)

    ###################################################

    ###################################################
    #-------------------
    # Prepare vector days that will be used in the download loop

    step = datetime.timedelta(days=1)
    result = []
    result0 = []
    # count=0
    dt = dt-step
    # create vector days
    while dt <= dtend+step:
        result.append(dt.strftime('%Y-%m-%d'))
        result0.append(dt.strftime('%Y%m%d'))
        dt += step

    result.append(dt.strftime('%Y-%m-%d'))
    result0.append(dt.strftime('%Y%m%d'))


    ################
    # Create directory path
    data_path = Path(cfg.data_base_path)
    data_path.mkdir(parents=True, exist_ok=True)

    ##############################################################################
    # Here we call different scripts to download CMEMS data at the Regional scale
    ##############################################################################

    # Download velocity data!
    if cfg.download_regional ==True:
        status = Download_CMEMS_currents(MyO_motuclient, MyO_prod, MyO_product_id, ibimfc_geowin,  nom, result0, result)
    
    if cfg.download_waves == True:
        # Download Wave data
        status = Download_CMEMS_waves(MyO_motuclient, WyO_prod, WyO_product_id, ibimfc_geowin,  nom, result0, result)


    ########################################################################################
    # Here we call different scripts to download Opendap data at coastal and harbour scale
    ########################################################################################
        
    if cfg.download_coastal == True:
        Download_opendap_currents('coastal', result0, data_path)
    
    if cfg.download_port == True:
        Download_opendap_currents('harbour', result0, data_path)

    ########################################################################################
    # Overlap and resample of coastal and local data is needed but performed in a second stage
    ########################################################################################


