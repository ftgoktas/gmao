#!/usr/bin/env python3
"""
Example script demonstrating how to use m21ctools for ensemble data processing.
This script shows how to:
1. Process ensemble data from tar files
2. Use icechunk for data versioning
3. Skip already processed timestamps
4. Visualize the results
"""

from datetime import datetime
import icechunk
import os
import m21ctools.data_handler as m21c
import matplotlib.pyplot as plt

def main():
    # Define the time range to process
    start_date = datetime(2010, 1, 4, 0)
    end_date = datetime(2010, 1, 5, 0)

    # Get variables to process
    var3d_list, var2d_list = m21c.get_variables()
    # For this example, we'll just process 'u' and 'ps'
    var3d_list = ['u']
    var2d_list = ['ps']

    print(f"Processing data from {start_date} to {end_date}")
    print(f"3D variables: {var3d_list}")
    print(f"2D variables: {var2d_list}")

    # Initialize or open Icechunk repository
    repo_path = "ensemble_store"
    storage = icechunk.local_filesystem_storage(repo_path)
    repo = icechunk.Repository.open_or_create(storage)

    # Get already processed timestamps to avoid reprocessing
    existing_times = m21c.get_existing_times(repo, 'u')
    if existing_times:
        print(f"\nFound {len(existing_times)} existing timestamps in repository")
        print(f"First timestamp: {min(existing_times)}")
        print(f"Last timestamp: {max(existing_times)}")
    else:
        print("\nNo existing data found in repository")

    # Process new data
    print("\nProcessing new data...")
    combined_averages = m21c.parallel_process_files_2d3d(
        start_date, end_date,
        var3d_list, var2d_list,
        skip_times=existing_times,
        num_workers=4  # Adjust based on your system
    )

    if not combined_averages:
        print("No new time steps to process.")
        return

    # Save new data
    commit_msg = f"Processed data from {start_date} to {end_date}"
    print(f"\nSaving data to icechunk repository: {commit_msg}")
    m21c.save_to_icechunk(repo, combined_averages, commit_msg)

    # Load and visualize
    print("\nLoading data for visualization...")
    u_data = m21c.load_from_icechunk(repo, 'u')
    ps_data = m21c.load_from_icechunk(repo, 'ps')

    print("\nCreating visualizations...")
    # Plot 3D variable (u)
    m21c.plot_hovmoeller_3d(u_data, var='u')

    # Plot 2D variable (ps)
    m21c.plot_hovmoeller_2d(ps_data, var='ps')

    print("\nDone! Plots saved in ./plots directory")

if __name__ == '__main__':
    main()
