import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import os

def read_cubed_sphere_data(file_path):
    with xr.open_dataset(file_path, engine="netcdf4", decode_coords=False) as ds:
        
        if "anchor" in ds:
            anchor_dims = list(ds["anchor"].dims)  
            
            if anchor_dims.count("ncontact") > 1:
                rename_dict = {"ncontact": "ncontact_2"} 
                ds = ds.rename_dims(rename_dict) 
                
                print(f"Renamed dimensions: {rename_dict}") 

        data = ds["QV"].isel(time=0, lev=0)
        lons = ds["lons"]
        lats = ds["lats"]

    return data, lons, lats

def adjust_longitudes(lons):
    lons = xr.where(lons > 180, lons - 360, lons)
    return lons

def interpolate_to_latlon_grid(lat, lon, data, resolution=1.0):
    lat_grid = np.arange(-90, 90 + resolution, resolution)
    lon_grid = np.arange(-180, 180 + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    data_grid = griddata((lat.flatten(), lon.flatten()), data.flatten(), (lat_grid, lon_grid), method='nearest')
    return lat_grid, lon_grid, data_grid


def plot_data(lat_grid, lon_grid, data_grid):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.contourf(lon_grid, lat_grid, data_grid * 1e+3, 60, transform=ccrs.PlateCarree(), cmap='GnBu')
    plt.colorbar(label='(g kg-1)')
    plt.title('Specific Humidity on Latitude-Longitude Grid')
    plt.show()

def cube2latlon(file_path):
    file_path = os.path.abspath(file_path)

    data, lons, lats = read_cubed_sphere_data(file_path)
    lons = adjust_longitudes(lons)

    all_lat, all_lon, all_data = [], [], []

    for face_index in range(data.shape[0]): 
        all_lat.append(lats.isel(nf=face_index).values)  
        all_lon.append(lons.isel(nf=face_index).values) 
        all_data.append(data.isel(nf=face_index).values)

    lat = np.concatenate([lat.flatten() for lat in all_lat])
    lon = np.concatenate([lon.flatten() for lon in all_lon])
    data = np.concatenate([d.flatten() for d in all_data])


    lat_grid, lon_grid, data_grid = interpolate_to_latlon_grid(lat, lon, data)
    plot_data(lat_grid, lon_grid, data_grid)
    
    return lat_grid, lon_grid, data_grid
