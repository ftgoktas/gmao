import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_data(lat_grid, lon_grid, data_grid, title="Interpolated Data"):
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    plt.contourf(lon_grid, lat_grid, data_grid * 1e+3, 60, transform=ccrs.PlateCarree(), cmap='GnBu')
    plt.colorbar(label='(g kg-1)')
    plt.title(title)
    plt.show()
