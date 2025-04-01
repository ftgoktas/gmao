import tarfile
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import multiprocessing
import icechunk
from icechunk.xarray import to_icechunk
import os
import time

start_time = time.time()
print(start_time)

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
    
    # tar_file_template = f"mock_data/{strm}/Y%Y/M%m/{tar_file_prefix}.%Y%m%d_%Hz.tar"
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


# TODO: make faster, SEE HOW IT'S BEING CALLED
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
    print("start Process 3D variables: Average over latitude and longitude")
    # Process 3D variables: Average over latitude and longitude
    for var in var3d_list:
        if var in ds:
            spread_data = ds[var]
            latlon_avg = spread_data.mean(dim=["lat", "lon"])
            latlon_avg_results[var] = latlon_avg

    print("start Process 2D variables: Average over longitude only")
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

    # while current_date <= end_date:
    #     if skip_times and np.datetime64(current_date + timedelta(hours=3)) in skip_times:
    #         current_date += timedelta(hours=6)
    #         continue

    #     tar_file_template, target_file_name = get_target_file(current_date)
    #     tar_file_path = current_date.strftime(tar_file_template)
    #     tar_files.append((tar_file_path, current_date, target_file_name))
    #     current_date += timedelta(hours=6)

    while current_date <= end_date:
        target_dt = current_date + timedelta(hours=3)
        if skip_times and np.datetime64(target_dt) in skip_times:
            print(f"Skipping, already processed: {target_dt}")
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

    print("done!")

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
    
#    plt.yscale('log')

#    Fix the x-axis to show readable time labels
#    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  
#    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
#    plt.gcf().autofmt_xdate()  
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


if __name__ == '__main__':
    # Setting the time range to process
    # works with generate_mock_1.py
    # start_date = datetime(2010, 1, 1, 0) 
    # end_date = datetime(2010, 1, 2, 0) # next 24 hours (5 steps if 6-hourly)
    start_date = datetime(2018, 6, 1, 21)
    end_date = datetime(2018, 9, 30, 21)

    # test first mock generation
    # works with generate_mock_1.py
    # start_date = datetime(2010, 1, 2, 12)  # starts from where previous test ended
    # end_date = datetime(2010, 1, 3, 12)    

    # test second mock generation
    # works with generate_mock_2.py
    # start_date = datetime(2010, 1, 4, 0)
    # end_date = datetime(2010, 1, 5, 0)

    var3d_list, var2d_list = get_variables()
    var3d_list = ['u']
    var2d_list = ['ps']

    # Initialize Icechunk storage and repository
    repo_path = "ensemble_store"
    storage = icechunk.local_filesystem_storage(repo_path)


    # Open the existing repository; if it doesn't exist, create a new one
    repo = icechunk.Repository.open_or_create(storage)

    # Get already processed timestamps to skip
    existing_times = get_existing_times(repo, 'u')  # any var is fine — 'u' is good

    print("Existing times in Icechunk:", sorted(existing_times))

    # Process new data only
    combined_averages = parallel_process_files_2d3d(
        start_date, end_date, var3d_list, var2d_list, skip_times=existing_times, num_workers=50
    )

    if not combined_averages:
        print("No new time steps to process.")
        
    else:
        commit_message = f"Processed data from {start_date} to {end_date}"
        save_to_icechunk(repo, combined_averages, commit_message)

        # Load full data and plot
        u_data = load_from_icechunk(repo, 'u')
        ps_data = load_from_icechunk(repo, 'ps')
    
        plot_hovmoeller_3d(u_data, var='u')
        plot_hovmoeller_2d(ps_data, var='ps')
        
    print(time.time() - start_time, "seconds")