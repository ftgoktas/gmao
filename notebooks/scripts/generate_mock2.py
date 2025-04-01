import os
import tarfile
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta

folder_prefix = "e5303_m21c_jan08"
stream = "m21c_ens_strm2"
base_path = Path("mock_data")
tar_output_path_base = base_path / stream

start_datetime = datetime(2010, 1, 2, 12)  # Next 6-hour interval
# Generate data for: 12z, 18z, 00z, 06z, 12z (next day)
num_steps = 5  # Add 5 more 6-hour time steps 

for step in range(num_steps):
    base_time = start_datetime + timedelta(hours=6 * step)
    target_time = base_time + timedelta(hours=3)

    date_str = base_time.strftime("%Y%m%d_%Hz")
    tar_parent = f"{folder_prefix}.atmens_stat.{date_str}"
    netcdf_subdir = f"{tar_parent}/ensrms"
    tar_output_path = tar_output_path_base / f"Y{base_time.year}" / f"M{base_time.month:02d}"
    tar_filename = f"{folder_prefix}.atmens_stat.{date_str}.tar"
    tar_full_path = tar_output_path / tar_filename

    netcdf_filename = f"{folder_prefix}.bkg.eta.{target_time.strftime('%Y%m%d_%H00z')}.nc4"
    netcdf_path = base_path / netcdf_subdir / netcdf_filename

    os.makedirs(netcdf_path.parent, exist_ok=True)
    os.makedirs(tar_output_path, exist_ok=True)

    # Create fake data
    time = np.array([np.datetime64(target_time)], dtype="datetime64[h]")
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20, endpoint=False)
    lev = np.linspace(1000, 100, 5)

    shape_3d = (1, len(lev), len(lat), len(lon))
    shape_2d = (1, len(lat), len(lon))

    ds = xr.Dataset(
        {
            "u": (["time", "lev", "lat", "lon"], np.random.rand(*shape_3d)),
            "tv": (["time", "lev", "lat", "lon"], np.random.rand(*shape_3d)),
            "v": (["time", "lev", "lat", "lon"], np.random.rand(*shape_3d)),
            "sphu": (["time", "lev", "lat", "lon"], np.random.rand(*shape_3d)),
            "ozone": (["time", "lev", "lat", "lon"], np.random.rand(*shape_3d)),
            "ps": (["time", "lat", "lon"], np.random.rand(*shape_2d)),
        },
        coords={
            "time": time,
            "lev": lev,
            "lat": lat,
            "lon": lon,
        }
    )

    ds.to_netcdf(netcdf_path)

    with tarfile.open(tar_full_path, "w") as tar:
        tar.add(netcdf_path, arcname=netcdf_path.relative_to(base_path))

    print(f"Wrote: {tar_full_path}")

