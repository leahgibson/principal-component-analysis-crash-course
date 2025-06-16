import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_animation(ds, cbar_name, cbar_lims, title_name, save_name):
    """
    Code to animate sst

    Parameters
    - ds: xarray dataset of sst with coordinates T, X, Y
    - cbar_name (str): name for colorbar
    - title (str): title name
    - save_name (str): name for save file, must end with .gif
    - cbar_mins (list) (optional): [vmin, vmax]
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.5)

    lon_min, lon_max = float(ds.X.min()), float(ds.X.max())
    lat_min, lat_max = float(ds.Y.min()), float(ds.Y.max())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False

    vmin = cbar_lims[0]
    vmax = cbar_lims[1]

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(
        ds.X,
        ds.Y,
        ds.isel(T=0),
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        norm=norm,
        shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label(cbar_name, fontsize=12)

    title = ax.set_title("", fontweight="bold")

    def animate(frame):
        current_data = ds.isel(T=frame)

        im.set_array(current_data.values.ravel())

        time_str = pd.to_datetime(ds["T"].values[frame]).strftime("%Y-%m")
        title.set_text(f"{title_name}, {time_str}")

        return [im, title]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(ds["T"]), interval=150, blit=True, repeat=True
    )
    save_path = os.path.join("./figures", save_name)
    anim.save(save_path, writer="pillow", fps=3, dpi=80)

    return anim, fig
