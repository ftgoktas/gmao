import pytest
import numpy as np
import xarray as xr
from m21ctools.cube2latlon import adjust_longitudes, interpolate_to_latlon_grid

def test_adjust_longitudes():
    lons = xr.DataArray([0, 90, 180, 270, 360])
    expected = xr.DataArray([0, 90, 180, -90, 0])
    result = adjust_longitudes(lons)
    assert np.array_equal(result, expected), f"Expected {expected} but got {result}"

def test_interpolate_to_latlon_grid():
    lat = np.array([0, 1, -1, 2, -2])
    lon = np.array([0, 1, -1, 2, -2])
    data = np.array([1, 2, 3, 4, 5])
    
    lat_grid, lon_grid, data_grid = interpolate_to_latlon_grid(lat, lon, data, resolution=1.0)

    assert lat_grid.shape == lon_grid.shape, "Lat and Lon grids should have the same shape"
    assert data_grid.shape == lat_grid.shape, "Data grid should match lat-lon grid shape"

def test_mock_cubed_sphere_data():
    nf = 6  
    ny, nx = 10, 10

    mock_lats = xr.DataArray(np.random.uniform(-90, 90, size=(nf, ny, nx)), dims=("nf", "y", "x"))
    mock_lons = xr.DataArray(np.random.uniform(-180, 180, size=(nf, ny, nx)), dims=("nf", "y", "x"))
    mock_data = xr.DataArray(np.random.rand(nf, ny, nx), dims=("nf", "y", "x"))

    lat = np.concatenate([mock_lats.isel(nf=i).values.flatten() for i in range(nf)])
    lon = np.concatenate([mock_lons.isel(nf=i).values.flatten() for i in range(nf)])
    data = np.concatenate([mock_data.isel(nf=i).values.flatten() for i in range(nf)])

    lat_grid, lon_grid, data_grid = interpolate_to_latlon_grid(lat, lon, data, resolution=1.0)
    
    assert lat_grid.shape == lon_grid.shape, "Lat and lon grids should match"
    assert data_grid.shape == lat_grid.shape, "Data grid should match lat-lon shape"

