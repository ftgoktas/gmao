"""
data_handler.py

CubedSphereData class, designed to read, process, interpolate, 
and visualize 2D cubed-sphere data from NetCDF files.

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import tarfile
from datetime import datetime, timedelta
import multiprocessing
import icechunk
from icechunk.xarray import to_icechunk
import warnings
import os

warnings.filterwarnings("ignore")

class CubedSphereData:
    def __init__(self, file_path, time=0, lev=0, variable="QV", resolution=1.0):
        """
        Handles reading, processing, interpolating, and visualizing 2D cubed-sphere data from cubed-sphere (.nc4) files.

        Parameters:
        - file_path (str): Path to the NetCDF file.
        - time (int): Time index to extract, default 0
        - lev (int): Level index to extract, default 0
        - variable (str): The variable to extract, default "QV"
        - resolution (float): Grid resolution in degrees for interpolation, default 1.0
        """
        self.file_path = file_path
        self.time = time
        self.lev = lev
        self.variable = variable
        self.resolution = resolution
        self.lats = None
        self.lons = None
        self.data = None
        self.lat_grid = None
        self.lon_grid = None
        self.data_grid = None
        self.raw_data = None
        self.raw_data_cleaned = None

        self.load_data()
        
    # Load data
    def load_data(self):
        """Reads NetCDF file and extracts data (lats, lons, and a user asigned
        variable at a given time and level), handling duplicate 'anchor' dimensions."""
        try:
            with xr.open_dataset(self.file_path, engine="h5netcdf") as ds:
                # Rename duplicate dimensions as the 'anchor' variable have duplicate dimensions - (nf, ncontact, ncontact)
                self.raw_data = ds.copy()
                anchor = ds['anchor']
                anchor_corrected = xr.DataArray(data=anchor.values, dims=('nf', 'ncontact1', 'ncontact2'), attrs=anchor.attrs)
                ds['anchor'] = anchor_corrected

                self.raw_data_cleaned = ds
                self.lats = ds["lats"].values
                self.lons = ds["lons"].values
                self.data = ds[self.variable].isel(time=self.time, lev=self.lev).values
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.file_path}")
            
        # Adjust longitudes
        self.lons = self.adjust_longitudes(self.lons)

        # Aggregate data from the cubed sphere
        self.all_lats, self.all_lons, self.all_data = self.aggregate_data()
            

    @staticmethod
    def adjust_longitudes(lons):
        """Adjusts longitudes to be within the range [-180, 180]."""
        return np.where(lons > 180, lons - 360, lons)

    def aggregate_data(self):
        """Aggregates the data across the six faces of the cubed sphere into flat lists.
        Returns:
        tuple: Lists of latitudes, longitudes, and corresponding data values."""
        all_lats, all_lons, all_data = [], [], []
        for face_index in range(self.data.shape[0]):  
            face_lats = self.lats[face_index, :, :].flatten()
            face_lons = self.lons[face_index, :, :].flatten()
            face_data = self.data[face_index, :, :].flatten()

            all_lats.extend(face_lats)
            all_lons.extend(face_lons)
            all_data.extend(face_data)

        return all_lats, all_lons, all_data

    def interpolate_to_latlon_grid(self, method='linear'):
        """Interpolates the data to a regular latitude-longitude grid.
        Returns:
        tuple: Interpolated latitude grid, longitude grid, and data grid arrays."""
        lat_grid = np.arange(-90, 90 + self.resolution, self.resolution)
        lon_grid = np.arange(-180, 180 + self.resolution, self.resolution)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

        # Perform interpolation
        data_grid = griddata((self.all_lats, self.all_lons), self.all_data, (lat_grid, lon_grid), method=method)

        self.lat_grid, self.lon_grid, self.data_grid = lat_grid, lon_grid, data_grid
        return lat_grid, lon_grid, data_grid

    @staticmethod
    def plot_data(lat_grid, lon_grid, data_grid):
        """Plots the interpolated data on a latitude-longitude grid with labeled axes."""

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()

        contour = plt.contourf(
            lon_grid, lat_grid, data_grid * 1e3, 
            60, transform=ccrs.PlateCarree(), cmap='GnBu'
        )

        plt.colorbar(contour, label='Specific Humidity (g kg⁻¹)')
        
        # Set axis labels and ticks
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        lon_ticks = range(-180, 181, 30)  # Longitude from -180 to 180, step 30°
        lat_ticks = range(-90, 91, 30)    # Latitude from -90 to 90, step 30°
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels(lon_ticks, fontsize=10)
        ax.set_yticklabels(lat_ticks, fontsize=10)

        plt.title('Specific Humidity on Latitude-Longitude Grid', fontsize=14)
        fig.savefig('specific_humidity.png')
        plt.show()


# === Ensemble Monitoring & Icechunk Integration ===

def get_target_file(date_in_tar):

    #For now DO include spinup year  
    year = date_in_tar.year
    if year < 2007 :
        strm = 'm21c_ens_strm1'
        expid = 'e5303_m21c_jan98'
    elif year >= 2007 and year < 2017:
        strm = 'm21c_ens_strm2'
        expid = 'e5303_m21c_jan08'
    else:
        strm = 'm21c_ens_strm3'
        expid = 'e5303_m21c_jan18'
    
    # Files 
    var_file_prefix = f"{expid}.bkg.eta"
    tar_file_prefix = f"{expid}.atmens_stat"
    
    tar_file_template = f"../data/{strm}/Y%Y/M%m/{tar_file_prefix}.%Y%m%d_%Hz.tar"
    
    parentdir = date_in_tar.strftime(f"{tar_file_prefix}.%Y%m%d_%Hz")

    target_datetime = date_in_tar + timedelta(hours=3)
    target_file_name = target_datetime.strftime(f"{parentdir}/ensrms/{var_file_prefix}.%Y%m%d_%H00z.nc4")

    return tar_file_template, target_file_name

def get_variables():
    var3d_list = ['tv', 'u', 'v', 'sphu', 'ozone']
    var2d_list = ['ps'] # Check if ts can be added here.

    return var3d_list, var2d_list


"""Open NetCDF file from tarball using Dask"""
def read_netcdf_from_tar_dask(tar_file, date_in_tar, target_file_name):
    with tarfile.open(tar_file, 'r') as tar:
        if target_file_name in tar.getnames():
            tar_file = tar.extractfile(target_file_name)
            if tar_file is None:
                return None

            ds = xr.open_dataset(tar_file, chunks={"time": 1})
            ds.load()  # Force load while tar is open
            return ds
        return None


def process_tar_file_2d3d(tar_file, date_in_tar, target_file_name, var3d_list, var2d_list):
    """
    Process a single tar file and return the lat-lon average for 3D variables,
    and longitude-only average for 2D variables, along with the corresponding time.
    """
    ds = read_netcdf_from_tar_dask(tar_file, date_in_tar, target_file_name)
    if ds is None:
        return None, None

    latlon_avg_results = {}  # Dictionary to hold results for each variable
    time_coord = ds['time'].values[0]  # Get the time coordinate (assumes 'time' dimension is present)
    # Process 3D variables: Average over latitude and longitude
    for var in var3d_list:
        if var in ds:
            spread_data = ds[var]
            latlon_avg = spread_data.mean(dim=["lat", "lon"])
            latlon_avg_results[var] = latlon_avg

    # Process 2D variables: Average over longitude only
    for var in var2d_list:
        if var in ds:
            spread_data = ds[var]
            lon_avg = spread_data.mean(dim="lon")  # Average over longitude only
            latlon_avg_results[var] = lon_avg

    return latlon_avg_results, time_coord


def parallel_process_files_2d3d(start_date, end_date, var3d_list, var2d_list, skip_times=None, num_workers=30):
    current_date = start_date
    tar_files = []

    while current_date <= end_date:
        target_dt = current_date + timedelta(hours=3)
        if skip_times and np.datetime64(target_dt) in skip_times:
            current_date += timedelta(hours=6)
            continue

        tar_file_template, target_file_name = get_target_file(current_date)
        tar_file_path = current_date.strftime(tar_file_template)

        if not os.path.exists(tar_file_path):
            print(f"Tarball not found: {tar_file_path}")
        else:
            print(f"Processing: {tar_file_path}")

        tar_files.append((tar_file_path, current_date, target_file_name))
        current_date += timedelta(hours=6)


    if not tar_files:
        return {}

    # Process files in parallel using multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        process_args = [(tar_file, date, target_file_name, var3d_list, var2d_list)
                        for tar_file, date, target_file_name in tar_files]
        results = pool.starmap(process_tar_file_2d3d, process_args)

    valid_results = [r for r in results if r[0] is not None]
    combined_averages = {var: [] for var in var3d_list + var2d_list}
    time_coords = []

    for avg_dict, time_coord in valid_results:
        for var, avg in avg_dict.items():
            combined_averages[var].append(avg)
        time_coords.append(time_coord)

    if not time_coords:
        return {}

    time_coords = np.array(time_coords)
    for var in combined_averages:
        if combined_averages[var]:  # skip empty
            combined_averages[var] = xr.concat(combined_averages[var], dim='time')
            combined_averages[var]['time'] = time_coords

    return combined_averages


def plot_hovmoeller_3d(latlon_avg: xr.DataArray, var):
    """
    Plot the Hovmöller diagram (lat-lon averaged ensemble spread over time and vertical levels).
    """
    time = latlon_avg['time'].values
    vertical_levels = latlon_avg['lev'].values
    spread_data = latlon_avg.values

    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    levels = np.linspace(0, 2.5, 32)  # 
    
    # Create contour plot with fixed color range
    cf = plt.contourf(time, vertical_levels, spread_data.T, 
                     cmap='YlGnBu', 
                     levels=levels,
                     extend='max')  # 'extend' handles values beyond the range
    
    # Add colorbar with specified range
    cbar = plt.colorbar(cf, label='(m/s)')
    
    # Set axis labels and title
    plt.xlabel('Time')
    plt.ylabel('Vertical Levels')
    plt.title(f'Ensemble Spread - {var}')
    
    # Adjust time axis to be less crowded
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Year-Month format
    
    # Optional: Add minor ticks for months in between
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate and align the tick labels so they look better
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.gca().invert_yaxis()
    
    plt.savefig(f'../plots/M21C_EnsSpread_hovm_{var}_{datetime.now().year}.png')
    plt.show()
    


def plot_hovmoeller_2d(lon_avg: xr.DataArray, var):
    """
    Plot the Hovmöller diagram (zonally averaged ensemble spread over time and latitudes ).
    """
    time = lon_avg['time'].values
    latitudes = lon_avg['lat'].values
    spread_data = lon_avg.values

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set specific levels for colorbar
    levels = np.linspace(0, 250, 32)  # 
    
    # Create contour plot with fixed color range
    cf = plt.contourf(time, latitudes, spread_data.T, 
                     cmap='YlGnBu', 
                     levels=levels,
                     extend='max')  # 'extend' handles values beyond the range
    
    # Add colorbar with specified range
    cbar = plt.colorbar(cf, label='[Unit]')
    
    # Set axis labels and title
    plt.xlabel('Time')
    plt.ylabel('Latitudes')
    plt.title(f'Ensemble Spread - {var}')
    
    # Adjust time axis to be less crowded
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every 1 month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Year-Month format
    
    # Optional: Add minor ticks for months in between
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate and align the tick labels so they look better
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f'../plots/M21C_EnsSpread_hovm_{var}_{datetime.now().year}.png', bbox_inches='tight', dpi=300)
    plt.show()


def save_to_icechunk(repo, combined_averages, commit_message):
    """
    Save the combined averages to the Icechunk repository.
    """
    session = repo.writable_session("main")
    for var_name, data_array in combined_averages.items():
        # Convert DataArray to Dataset (check this)
        if isinstance(data_array, xr.DataArray):
            data_array = data_array.to_dataset(name=var_name)
        to_icechunk(data_array, session, mode="a")  # or mode="w" to overwrite

    session.commit(commit_message)


def load_from_icechunk(repo, var_name):
    """
    Load a variable from the Icechunk repository.
    """
    # Start a read-only session from the "main" branch
    session = repo.readonly_session("main")

    # Open the dataset using xarray and return the specified variable
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds[var_name]

def get_existing_times(repo, var_name):
    try:
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)
        if var_name not in ds:
            return set()
        return set(ds[var_name].coords['time'].values)
    except FileNotFoundError:
        # Happens if no data has been written yet
        return set()
