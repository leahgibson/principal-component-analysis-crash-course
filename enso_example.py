"""
Deriving ENSO from sea surface temperature using PCA.

Inspired by: https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis/chapters/Week7/Intro_to_PCA.html
"""

# import numpy as np
import xarray as xr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Access dataset
iri_url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/"
T_convert = "T/[(days)(since)(1960-01-01)]sconcat/streamgridunitconvert/"
url = iri_url + T_convert + "dods"

ds = xr.open_dataset(url)  # vars are sea surface temp:sst, lat:Y, lon:X, time:T
ds = ds.sel(T=slice("1854", "2024"))

# PCA approach
sst_data = ds.sst.sel(zlev=0, X=slice(120, 300), Y=slice(-30, 30)).dropna(
    dim="T", how="all"
)

sst_flat = sst_data.stack(space=["Y", "X"]).fillna(0)  # time x lat*lon
sst_array = sst_flat.values

scaler = StandardScaler()
sst_scaled = scaler.fit_transform(sst_array)

pca = PCA(n_components=5)
pca_result = pca.fit_transform(
    sst_scaled
)  # projection of data onto eigenvectors (dim reduction)
components = pca.components_  # eigenvectors

# Plot eigenvectors mapped back to lat lon grid
# Eigenvectors capture spatial patterns of where sst varies together


def plot_component(i):
    component = components[i].reshape(len(sst_data.Y), len(sst_data.X))
    titles = [
        "Eigenvector 1",
        "Eigenvector 2",
        "Eigenvector 3",
    ]

    vmax = np.max(np.abs(component))
    vmin = -vmax

    im = axes[i].pcolormesh(
        lon_grid,
        lat_grid,
        component,
        cmap="PiYG",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    axes[i].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[i].add_feature(cfeature.BORDERS, linewidth=0.5)
    axes[i].add_feature(cfeature.LAND, color="lightgray", alpha=0.5)
    axes[i].gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5
    )
    axes[i].axhline(y=0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    axes[i].set_title(titles[i], fontsize=14, fontweight="bold")
    axes[i].set_extent([120, 300, -30, 30], crs=ccrs.PlateCarree())

    # Add ENSO region box for second component (i=1)
    if i == 1:
        # Define ENSO region boundaries
        enso_lon_min, enso_lon_max = 190, 240
        enso_lat_min, enso_lat_max = -5, 5

        # Create rectangle for ENSO region
        from matplotlib.patches import Rectangle

        enso_box = Rectangle(
            (enso_lon_min, enso_lat_min),
            enso_lon_max - enso_lon_min,
            enso_lat_max - enso_lat_min,
            linewidth=3,
            edgecolor="black",
            facecolor="none",
            linestyle="-",
            transform=ccrs.PlateCarree(),
        )
        axes[i].add_patch(enso_box)

        # Add text label
        axes[i].text(
            215,
            7,
            "ENSO 3.4 Region",
            transform=ccrs.PlateCarree(),
            fontsize=12,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.colorbar(im, ax=axes[i], orientation="horizontal", pad=0.05, shrink=0.8)


fig, axes = plt.subplots(
    nrows=3,
    figsize=(6, 12),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
)


lons = sst_data.X.values
lats = sst_data.Y.values
lon_grid, lat_grid = np.meshgrid(lons, lats)


first_component = components[0].reshape(len(sst_data.Y), len(sst_data.X))
second_component = components[1].reshape(len(sst_data.Y), len(sst_data.X))
third_component = components[2].reshape(len(sst_data.Y), len(sst_data.X))

# Plot 1: Seasonality (First Component)
plot_component(0)

# Plot 2: ENSO (Second Component)
plot_component(1)

# Plot 3: Third Component
plot_component(2)

plt.tight_layout()
plt.show()
# plt.savefig("evectors.png", dpi=300)


def extract_patterns(lon_bounds, lat_bounds):
    region = ds.sst.sel(
        zlev=0,
        X=slice(lon_bounds[0], lon_bounds[1]),
        Y=slice(lat_bounds[0], lat_bounds[1]),
    )

    regional_avg = region.mean(dim=("X", "Y"))

    # 1950 - 1979 climatology as baseline
    monthly_climatology = (
        regional_avg.sel(T=slice("1950", "1979")).groupby("T.month").mean()
    )

    sst_anomly = regional_avg.groupby("T.month") - monthly_climatology
    sst_anomly_sd = sst_anomly.sel(T=slice("1950", "1979")).std()
    sst_anomly_normalized = sst_anomly / sst_anomly_sd

    return sst_anomly_normalized


normal = extract_patterns(lon_bounds=[140, 160], lat_bounds=[-10, 10])
enso = extract_patterns(lon_bounds=(190, 240), lat_bounds=(-6, 6))


fig = plt.figure(figsize=(12, 6), dpi=300)
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1])

ax1 = fig.add_subplot(gs[0:4])
ax2 = fig.add_subplot(gs[0, 4])
ax3 = fig.add_subplot(gs[1, :])
ax4 = fig.add_subplot(gs[2, :])

# Extract seasonal cycle from climatology
region = ds.sst.sel(zlev=0, X=slice(180, 230), Y=slice(16, 26))
regional_avg = region.mean(dim=("X", "Y"))

monthly_climatology = (
    regional_avg.sel(T=slice("1950", "1979")).groupby("T.month").mean()
)
climatology_std = monthly_climatology.std()
climatology_mean = monthly_climatology.mean()
seasonal_cycle = (monthly_climatology - climatology_mean) / climatology_std


# Plot 1: PC1 (seasonality)
pc1_normalized = (pca_result[:, 0] - pca_result[:, 0].mean()) / pca_result[:, 0].std()
ax1.plot(ds.T.values, pc1_normalized, "b-", linewidth=1.5)
ax1.set_title("PC1", fontweight="bold")

pc1_da = xr.DataArray(
    pc1_normalized, coords={"T": ds.T.values}, dims=["T"], name="pc1_standardized"
)
pc1_monthly = pc1_da.groupby("T.month").mean()

# Plot 2: Seasonality of Pacific
seasonal_cycle.plot(ax=ax2, color="green", alpha=0.8, label="Seasonality")
pc1_monthly.plot(ax=ax2, color="blue", alpha=0.8, label="PC1 Avg")
ax2.set_ylabel("")
ax2.set_title("Seasonal Cycle", fontsize=12, fontweight="bold")
ax2.legend()

# Plot 3: PC2 (ENSO)
pc2_std = (pca_result[:, 1] - pca_result[:, 1].mean()) / pca_result[:, 1].std()
ax3.plot(ds.T.values, pc2_std, "r-", linewidth=1.5, label="PC2", alpha=0.8)
enso.plot(ax=ax3, color="orange", linewidth=2, label="ENSO Index", alpha=0.7)
ax3.set_title("PC2 vs ENSO Index", fontsize=12, fontweight="bold")
ax3.set_ylabel("Index Value")
ax3.legend()

# Plot 4: PC3 (seasonality in ENSO conditions)
pc3_std = (pca_result[:, 2] - pca_result[:, 2].mean()) / pca_result[:, 2].std()
ax4.plot(ds.T.values, pc3_std, "m-", linewidth=1.5, alpha=0.8)  # label="PC3"
# normal.plot(ax=ax4, color="cyan", linewidth=2, label="Normal Index", alpha=0.7)
ax4.set_title("PC3", fontsize=12, fontweight="bold")
ax4.set_ylabel("Index Value")
# ax4.legend()

plt.tight_layout()
plt.savefig("PC.png")

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))
