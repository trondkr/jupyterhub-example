import xarray as xr
import cftime
import numpy as np
import pandas as pd
import geopandas as gpd

from cmip6_preprocessing.preprocessing import combined_preprocessing

def weighted_mean_of_masked_data(data_in,data_mask,data_cond):
    #data_in = input xarray data to have weighted mean
    #data_mask = nan mask eg. land values
    #LME mask T or F values
    global_attrs = data_in.attrs
    R = 6.37e6 #radius of earth in m
    grid_dy,grid_dx = (data_in.lat[0]-data_in.lat[1]).data,(data_in.lon[0]-data_in.lon[1]).data
    dϕ = np.deg2rad(grid_dy)
    dλ = np.deg2rad(grid_dx)
    dA = R**2 * dϕ * dλ * np.cos(np.deg2rad(ds.lat)) 
    pixel_area = dA.where(data_cond)  #pixel_area.plot()
    pixel_area = pixel_area.where(np.isfinite(data_mask))
    total_ocean_area = pixel_area.sum(dim=('lon', 'lat'))
    data_weighted_mean = (data_in * pixel_area).sum(dim=('lon', 'lat'),keep_attrs=True) / total_ocean_area
    data_weighted_mean.attrs = global_attrs  #save global attributes
    for a in data_in:                      #set attributes for each variable in dataset
        gatt = data_in[a].attrs
        data_weighted_mean[a].attrs=gatt
    return data_weighted_mean

def weighted_mean_of_data(data_in,data_cond):
    #data_in = input xarray data to have weighted mean
    #data_mask = nan mask eg. land values
    #LME mask T or F values
    global_attrs = data_in.attrs
    R = 6.37e6 #radius of earth in m
    grid_dy,grid_dx = (data_in.lat[0]-data_in.lat[1]).data,(data_in.lon[0]-data_in.lon[1]).data
   # grid_dy,grid_dx = (data_in.lat[0,0]-data_in.lat[0,1]).data,(data_in.lon[0,0]-data_in.lon[1,0]).data

    dϕ = np.deg2rad(grid_dy)
    dλ = np.deg2rad(grid_dx)
    dA = R**2 * dϕ * dλ * np.cos(np.deg2rad(data_in.lat)) 
    pixel_area = dA.where(data_cond)  #pixel_area.plot()
  
    sum_data=(data_in*pixel_area).sum(dim=('lon', 'lat'),keep_attrs=True)
    total_ocean_area = pixel_area.sum(dim=('lon', 'lat'))

    data_weighted_mean = sum_data/total_ocean_area
    data_weighted_mean.attrs = global_attrs  #save global attributes
    for a in data_in:                      #set attributes for each variable in dataset
        gatt = data_in[a].attrs
        data_weighted_mean[a].attrs=gatt

    return data_weighted_mean

def get_and_organize_cmip6_data(conf):
    # Dictionary to hold the queried variables
  
    first = True
    for experiment_id in conf.experiment_ids:
        for grid_label in conf.grid_labels:
            for source_id in conf.source_ids:
                for member_id in conf.member_ids:
                    for variable_id, table_id in zip(conf.variable_ids, conf.table_ids):
                        
                        # Create unique key to hold dataset in dictionary
                        key="{}_{}_{}_{}_{}".format(variable_id,experiment_id,grid_label,source_id,member_id)
                        # Historical query string
                        query_string = "source_id=='{}'and table_id=='{}' and grid_label=='{}' and experiment_id=='historical' and variable_id=='{}'".format(source_id, 
                        table_id, 
                        grid_label,
                        variable_id)
                      
                        print(
                            "Running historical query on data: \n ==> {}\n".format(
                                query_string
                            )
                        )
                        ds_hist = perform_cmip6_query(conf,query_string)
                       
                        # Future projection depending on choice in experiment_id
                        query_string = "source_id=='{}'and table_id=='{}' and member_id=='{}' and grid_label=='{}' and experiment_id=='{}' and variable_id=='{}'".format(
                                source_id,
                                table_id,
                                member_id,
                                grid_label,
                                experiment_id,
                                variable_id,
                            )
                        print(
                            "Running projections query on data: \n ==> {}\n".format(
                                query_string
                            )
                        )
                        ds_proj = perform_cmip6_query(conf,query_string)
                        
                        # Concatentate the historical and projections datasets
                        ds = xr.concat([ds_hist, ds_proj], dim="time")
                        # Remove the duplicate overlapping times (e.g. 2001-2014)
                   #     _, index = np.unique(ds["time"], return_index=True)
                   #     ds = ds.isel(time=index)
               
                        # Extract the time period of interest
                        ds=ds.sel(time=slice(conf.start_date,conf.end_date))
                        print("{} => Dates extracted range from {} to {}\n".format(source_id,ds["time"].values[0], ds["time"].values[-1]))
                        # pass the preprocessing directly
                        dset_processed = combined_preprocessing(ds)
                        if (variable_id in ["chl"]):
                            if (source_id in ["CESM2","CESM2-FV2","CESM2-WACCM-FV2","CESM2-WACCM"]):
                                dset_processed = dset_processed.isel(lev_partial=conf.selected_depth)    
                            else:
                                dset_processed = dset_processed.isel(lev=conf.selected_depth)    
                       
                        # Save the dataset for variable_id in the dictionary
                        conf.dset_dict[key] = dset_processed
    return conf


def get_LME_records():
    lme_file='./data/LME/LME66.shp'
    return gpd.read_file(lme_file)

def get_LME_records_plot():
    lme_file='./data/LME66_180/LME66_180.shp'
    lme_file='./data/LME/LME66.shp'
    return gpd.read_file(lme_file)

    
def perform_cmip6_query(conf,query_string):
    df_sub = conf.df.query(query_string)
    if (df_sub.zstore.values.size==0):
        return df_sub
    
    mapper = conf.fs.get_mapper(df_sub.zstore.values[-1])
    ds = xr.open_zarr(mapper, consolidated=True)
    print("Time encoding: {} - {}".format(ds.indexes['time'],ds.indexes['time'].dtype))
    if not ds.indexes["time"].dtype in ["datetime64[ns]","object"]:
        
        time_object = ds.indexes['time'].to_datetimeindex() #pd.DatetimeIndex([ds["time"].values[0]])
        print(time_object,time_object.year)
        # Convert if necesssary
        if time_object[0].year == 1:

            times = ds.indexes['time'].to_datetimeindex()  # pd.DatetimeIndex([ds["time"].values])
            times_plus_2000 = []
            for t in times:
                times_plus_2000.append(
                    cftime.DatetimeNoLeap(t.year + 2000, t.month, t.day, t.hour)
                )
            ds["time"].values = times_plus_2000
            ds = xr.decode_cf(ds)                    
    return ds

def get_pices_cmip6_data(var, ilme, initial_date,final_date):
    
    file = get_filename(var)
    print('opening:',file)
    ds = xr.open_dataset(file)
    ds.close()
    
    #subset to time of interest
    ds = ds.sel(time=slice(initial_date,final_date))   
    
    if (str(var).lower()=='current') or (var==3):  #if current data need to mask
        m=ds.mask.sel(time=slice('1992-01-01','2010-01-01')).min('time')
        ds = ds.where(m==1,np.nan)
        ds = ds.drop('mask')
       
    #read in pices LME mask
    ds_mask = get_pices_mask()
    #interpolate mask
    mask_interp = ds_mask.interp_like(ds,method='nearest')

    #create mean for pices region
    cond = (mask_interp.region_mask==ilme)
    tem = weighted_mean_of_data(ds,cond)
    data_mean=tem.assign_coords(region=ilme)

    #make climatology and anomalies using .groupby method
    data_climatology = data_mean.groupby('time.month').mean('time',keep_attrs=True)
    data_anomaly = data_mean.groupby('time.month') - data_climatology
    global_attributes = ds.attrs
    data_anomaly.attrs = global_attributes
    
    return data_mean, data_climatology, data_anomaly

def get_pices_data(var, ilme, initial_date,final_date):
    import xarray as xr
    import numpy as np
    
    file = get_filename(var)
    print('opening:',file)
    ds = xr.open_dataset(file)
    ds.close()
    
    #subset to time of interest
    ds = ds.sel(time=slice(initial_date,final_date))   
    
    if (str(var).lower()=='current') or (var==3):  #if current data need to mask
        m=ds.mask.sel(time=slice('1992-01-01','2010-01-01')).min('time')
        ds = ds.where(m==1,np.nan)
        ds = ds.drop('mask')
       
    #read in pices LME mask
    ds_mask = get_pices_mask()
    #interpolate mask
    mask_interp = ds_mask.interp_like(ds,method='nearest')

    #create mean for pices region
    cond = (mask_interp.region_mask==ilme)
    tem = weighted_mean_of_data(ds,cond)
    data_mean=tem.assign_coords(region=ilme)

    #make climatology and anomalies using .groupby method
    data_climatology = data_mean.groupby('time.month').mean('time',keep_attrs=True)
    data_anomaly = data_mean.groupby('time.month') - data_climatology
    global_attributes = ds.attrs
    data_anomaly.attrs = global_attributes
    
    return data_mean, data_climatology, data_anomaly

def get_lme_data(var, ilme, initial_date,final_date):
    import xarray as xr
    
    file = get_filename(var)
    #print('opening:',file)]
    ds = xr.open_dataset(file)
    ds.close()
    
    #subset to time of interest
    ds = ds.sel(time=slice(initial_date,final_date))   
    
    #read in mask
    ds_mask = get_lme_mask()
    #interpolate mask
    mask_interp = ds_mask.interp_like(ds,method='nearest')

    #create mean for pices region
    cond = (mask_interp.region_mask==ilme)
    tem = weighted_mean_of_data(ds,cond)
    data_mean=tem.assign_coords(region=ilme)

    #make climatology and anomalies using .groupby method
    data_climatology = data_mean.groupby('time.month').mean('time')
    data_anomaly = data_mean.groupby('time.month') - data_climatology

    return data_mean, data_climatology, data_anomaly
