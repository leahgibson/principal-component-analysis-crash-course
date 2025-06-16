"""
Deriving ENSO from sea surface temperature using PCA.

Inspired by: https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis/chapters/Week7/Intro_to_PCA.html
"""

# import numpy as np
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from visualization import create_animation

# Access dataset
iri_url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/"
T_convert = "T/[(days)(since)(1960-01-01)]sconcat/streamgridunitconvert/"
url = iri_url + T_convert + "dods"

ds = xr.open_dataset(url)  # vars are sea surface temp:sst, lat:Y, lon:X, time:T
ds = ds.sel(T=slice("1854", "2024"))


# PCA analysis
sst_data = ds.sst.sel(zlev=0, X=slice(120, 300), Y=slice(-30, 30)).dropna(
    dim="T", how="all"
)

sst_flat = sst_data.stack(space=["Y", "X"]).fillna(0)  # time x lat*lon
sst_array = sst_flat.values

scaler = StandardScaler()
sst_scaled = scaler.fit_transform(sst_array)  # data scaled to mean 0 and unit variance

pca = PCA(n_components=5)
pca_result = pca.fit_transform(
    sst_scaled
)  # dimension reduction (these are the principal components)
components = pca.components_  # principal axes


def projection(i):
    component = components[i].reshape(-1, 1)

    projected_matrix = []
    dots = []
    # project each row of sst data onto principal axis
    for row in sst_scaled:
        row = row.reshape(1, -1)
        dot = row @ component
        proj = dot * component

        dots.append(dot[0][0])
        projected_matrix.append(proj)

    projected_matrix = np.squeeze(np.array(projected_matrix))

    return projected_matrix


def reshape_to_xarray(proj, i=None):
    """Takes projection, reshpaes to grid, and puts in xarray ds"""

    time_coords = sst_data["T"]
    lat_coords = sst_data["Y"]
    lon_coords = sst_data["X"]

    n_lat = len(lat_coords)
    n_lon = len(lon_coords)

    reshaped = proj.reshape(len(time_coords), n_lat, n_lon)

    if i is not None:
        name = "projection_{i}_sst"
    else:
        name = "sst_xr"

    projected_sst = xr.DataArray(
        reshaped,
        coords={"T": time_coords, "Y": lat_coords, "X": lon_coords},
        dims=["T", "Y", "X"],
        name=name,
    )

    return projected_sst


# Project first 3 principal axes
cumulative_proj = None
for i in range(3):
    proj = reshape_to_xarray(projection(i), i)

    if cumulative_proj is None:
        cumulative_proj = proj.copy()
    else:
        cumulative_proj += proj
    proj_subset = proj.sel(X=slice(120, 300), Y=slice(-30, 30), T=slice("2009", "2024"))
    vmin = float(proj_subset.min())
    vmax = float(proj_subset.max())
    anim, fig = create_animation(
        proj_subset,
        cbar_name="Sea Surface Temperature Standard Deviation",
        cbar_lims=[vmin, vmax],
        title_name=f"Projection onto PA{i + 1}",
        save_name=f"projection{i}.gif",
    )

# Plot standardized SST data
sst_standardized = reshape_to_xarray(sst_scaled)
sst_standardized_subset = sst_standardized.sel(
    X=slice(120, 300), Y=slice(-30, 30), T=slice("2009", "2024")
)
vmin = float(sst_standardized_subset.min())
vmax = float(sst_standardized_subset.max())
anim, fig = create_animation(
    sst_standardized_subset,
    cbar_name="Sea Surface Temperature Standard Deviation",
    cbar_lims=[vmin, vmax],
    title_name="Standardized SST",
    save_name="standardized_sst.gif",
)

# Plot the reconstructed sst from first 3 principal axes
cumulative_subset = cumulative_proj.sel(
    X=slice(120, 300), Y=slice(-30, 30), T=slice("2009", "2024")
)
anim, fig = create_animation(
    cumulative_subset,
    cbar_name="Sea Surface Temperature Standard Deviation",
    title_name="Reconstructed Standardized SST",
    save_name="cumulative_projection.gif",
    cbar_lims=[vmin, vmax],  # lims from sst_standardized_subset
)


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


enso = extract_patterns(lon_bounds=(190, 240), lat_bounds=(-6, 6))

# PC1
fig, ax = plt.subplots(figsize=(12, 4))
pc1_normalized = (pca_result[:, 0] - pca_result[:, 0].mean()) / pca_result[:, 0].std()
ax.plot(ds["T"].values[1100:], pc1_normalized[1100:], color="blue")
ax.set_ylabel("SST SD")
ax.set_title("Principal Component 1", fontweight="bold")
plt.savefig("./figures/pc1.png")
# plt.show()

# PC2
fig, ax = plt.subplots(figsize=(12, 4))
pc2_std = (pca_result[:, 1] - pca_result[:, 1].mean()) / pca_result[:, 1].std()
ax.plot(
    ds["T"].values[1100:], pc2_std[1100:], "r-", linewidth=1.5, label="PC2", alpha=0.8
)
enso.isel(T=slice(1100, None)).plot(
    ax=ax, color="orange", linewidth=2, label="ENSO Index", alpha=0.7
)
ax.set_title("Principal Component 2", fontsize=12, fontweight="bold")
ax.set_ylabel("SST SD")
ax.legend()
plt.savefig("./figures/pc2.png")
# plt.show()

# PC3
fig, ax = plt.subplots(figsize=(12, 4))
pc3_std = (pca_result[:, 2] - pca_result[:, 2].mean()) / pca_result[:, 2].std()
ax.plot(
    ds["T"].values[1100:], pc3_std[1100:], "m-", linewidth=1.5, alpha=0.8
)  # label="PC3"
ax.set_title("Principal Component 3", fontsize=12, fontweight="bold")
ax.set_ylabel("SST SD")
plt.savefig("./figures/pc3.png")
# plt.show()
