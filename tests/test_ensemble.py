import pytest
from datetime import datetime
import icechunk
import os
import numpy as np
import xarray as xr
import m21ctools.data_handler as m21c

@pytest.fixture
def test_repo():
    repo_path = "test_ensemble_store"
    storage = icechunk.local_filesystem_storage(repo_path)
    repo = icechunk.Repository.open_or_create(storage)
    yield repo
    # Cleanup after tests
    if os.path.exists(repo_path):
        import shutil
        shutil.rmtree(repo_path)

def test_get_variables():
    var3d_list, var2d_list = m21c.get_variables()
    assert isinstance(var3d_list, list)
    assert isinstance(var2d_list, list)
    assert 'u' in var3d_list
    assert 'ps' in var2d_list

def test_get_target_file():
    test_date = datetime(2010, 1, 4, 0)
    tar_template, target_file = m21c.get_target_file(test_date)
    assert isinstance(tar_template, str)
    assert isinstance(target_file, str)
    assert '.tar' in tar_template
    assert '.nc4' in target_file

def test_save_and_load_icechunk(test_repo):
    # Create sample data
    times = [datetime(2010, 1, 4, 0)]
    data = np.random.rand(1, 10, 10)  # time, lat, lon
    test_data = xr.DataArray(
        data,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': times,
            'lat': np.linspace(-90, 90, 10),
            'lon': np.linspace(-180, 180, 10)
        }
    )
    
    test_averages = {'test_var': test_data}
    
    # Test save
    m21c.save_to_icechunk(test_repo, test_averages, "Test commit")
    
    # Test load
    loaded_data = m21c.load_from_icechunk(test_repo, 'test_var')
    assert isinstance(loaded_data, xr.DataArray)
    assert loaded_data.shape == test_data.shape
    np.testing.assert_array_equal(loaded_data.values, test_data.values)
