
#!/usr/bin/env python3

"""
Single-day PoF forecast test using existing GRIB inputs.

Does NOT read training_data.parquet.
Does NOT retrieve any data.

Uses the same GRIB input processing, feature definitions and
pixel masks as the training workflow in pof_utils.py.

Outputs:
    outputs/POF_forecast_YYYYMMDD.png
    outputs/POF_forecast_YYYYMMDD.grib2
    outputs/POF_forecast_YYYYMMDD.parquet
"""

from pathlib import Path
from contextlib import ExitStack
import json

import joblib
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pof_utils as pof


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

MODEL_PATH = DATA_DIR / "POF_model.joblib"
FEATURES_PATH = DATA_DIR / "POF_features.json"

GRID_TEMPLATE = DATA_DIR / "CLIMATE" / "POP_2020.grib"
STATIC_FIRE_MASK = DATA_DIR / "static_fire_mask.npy"

# Date to test: YYYYMMDD
TEST_DATE = 20100715

# Prediction batch size
PREDICTION_BATCH_SIZE = 500_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load model
# =============================================================================

print("=" * 65)
print("PoF Single-Day Forecast Test")
print("=" * 65)

test_date = pd.to_datetime(str(TEST_DATE), format="%Y%m%d")

print(f"\nTest date: {test_date.date()}")

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

feature_info = json.loads(FEATURES_PATH.read_text())
feature_cols = feature_info["features"]

print(f"Model features: {len(feature_cols)}")
print(f"Model trained on: {feature_info.get('trained_on', 'unknown')}")


# =============================================================================
# Check existing input files
# =============================================================================

year = test_date.year
month = test_date.month

stem = f"{year}_{month:02d}"

print(f"\nChecking existing GRIB inputs for {stem}...")

required_files = set()

for column, (pattern, filters) in pof.TRAINING_INPUTS.items():
    required_files.add(DATA_DIR / pattern.format(stem=stem))

required_files.update([
    GRID_TEMPLATE,
    DATA_DIR / "CLIMATE" / "road_density_2015_agg_r.grib",
    STATIC_FIRE_MASK,
])

missing_files = [
    path for path in sorted(required_files)
    if not path.exists()
]

if missing_files:
    print("\nMissing required files:")
    for path in missing_files:
        print(f"  {path}")

    raise FileNotFoundError(
        "Some prepared input files are missing. "
        "No retrieval has been attempted."
    )

print(f"All {len(required_files)} required files found.")


# =============================================================================
# Load static fields and grid
# =============================================================================

print("\nLoading static fields...")

population = pof.read_grib(
    DATA_DIR / "CLIMATE" / "POP_2020.grib",
    decode_times=False,
)

roads = pof.read_grib(
    DATA_DIR / "CLIMATE" / "road_density_2015_agg_r.grib",
    decode_times=False,
)

pof.check_same_grid(population, roads, "road_density")

static = {
    "PO": np.asarray(population.values, dtype=np.float32).ravel(),
    "RD": np.asarray(roads.values, dtype=np.float32).ravel(),
}

lats = np.asarray(population.latitude.values)
lons = np.asarray(population.longitude.values)

nj = len(lats)
ni = len(lons)

print(f"Grid dimensions: {ni} x {nj}")
print(f"Grid cells: {ni * nj:,}")
print(
    f"Latitude: {lats.min():.2f} to {lats.max():.2f}"
)
print(
    f"Longitude: {lons.min():.2f} to {lons.max():.2f}"
)

lat_flat = np.repeat(lats, ni)
lon_flat = np.tile(lons, nj)


# =============================================================================
# Load static fire mask
# =============================================================================

print("\nLoading static fire-source mask...")

static_fire_mask = np.load(STATIC_FIRE_MASK).astype(bool).ravel()

if static_fire_mask.size != ni * nj:
    raise ValueError(
        "Static fire mask does not match the template grid."
    )

print(
    f"Static fire-source pixels excluded: "
    f"{static_fire_mask.sum():,}"
)


# =============================================================================
# Read the selected day's GRIB fields
# =============================================================================

print(f"\nOpening GRIB inputs for {stem}...")

day_fields = None

with ExitStack() as stack:

    inputs = pof._open_month_inputs(
        stack,
        str(DATA_DIR),
        stem,
        population,
    )

    for day, fields in pof._month_fields_by_day(inputs, stem):

        if day.normalize() == test_date.normalize():
            day_fields = fields
            break

if day_fields is None:
    raise ValueError(
        f"No active-fire observations found for {test_date.date()}"
    )

print(f"Successfully read GRIB fields for {test_date.date()}")

for name, values in day_fields.items():
    print(
        f"  {name:8s}: "
        f"{np.count_nonzero(np.isfinite(values)):,} finite values"
    )


# =============================================================================
# Apply the same training-data masks
# =============================================================================

print("\nApplying training-data masks...")

# Minimum latitude and valid static fields.
mask = (
    (lat_flat >= pof.MIN_LATITUDE)
    & np.isfinite(static["PO"])
    & np.isfinite(static["RD"])
)

# Exclude static heat sources.
mask &= ~static_fire_mask

# Total fuel load.
total_fuel = (
    day_fields["FU_LL"]
    + day_fields["FU_LW"]
    + day_fields["FU_DF"]
    + day_fields["FU_DW"]
)

mask &= total_fuel > pof.MIN_TOTAL_FUEL

# Require all model input variables to be finite.
feature_inputs = [
    column
    for column in pof.TRAINING_INPUTS
    if column != "AF"
]

for column in feature_inputs:
    mask &= np.isfinite(day_fields[column])

index = np.flatnonzero(mask)

if index.size == 0:
    raise ValueError("No valid pixels remain after applying the masks.")

print(f"Eligible pixels: {len(index):,}")

# Observed fire occurrence, using exactly the same definition as training.
observed = np.isin(
    day_fields["AF"][index],
    pof.FIRE_CODES,
).astype(np.uint8)

print(f"Observed fires: {observed.sum():,}")
print(f"Observed fire rate: {observed.mean():.6f}")


# =============================================================================
# Construct model feature matrix
# =============================================================================

print("\nConstructing model features...")

feature_data = {}

for column in feature_inputs:
    feature_data[column] = day_fields[column][index].astype(
        np.float32
    )

for column, values in static.items():
    feature_data[column] = values[index].astype(np.float32)

X = pd.DataFrame(feature_data)

# Ensure exactly the same feature order as during training.
missing_features = [
    column for column in feature_cols
    if column not in X.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

X = X[feature_cols]

print(f"Feature matrix: {X.shape}")


# =============================================================================
# Generate predictions
# =============================================================================

print("\nGenerating PoF predictions...")

predictions = np.empty(len(X), dtype=np.float32)

for start in range(0, len(X), PREDICTION_BATCH_SIZE):

    end = min(start + PREDICTION_BATCH_SIZE, len(X))

    predictions[start:end] = model.predict_proba(
        X.iloc[start:end]
    )[:, 1]

    print(
        f"  Predicted {end:,} / {len(X):,} pixels"
    )

del X

print("\nPrediction statistics:")
print(f"Mean PoF:   {predictions.mean():.6f}")
print(f"Median PoF: {np.median(predictions):.6f}")
print(f"Maximum:    {predictions.max():.6f}")
print(f"> 0.01:     {(predictions > 0.01).sum():,}")
print(f"> 0.1:      {(predictions > 0.1).sum():,}")


# =============================================================================
# Calculate test-day metrics
# =============================================================================

print("\nComparison with observations:")

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

if np.unique(observed).size == 2:

    print(
        f"ROC-AUC:           "
        f"{roc_auc_score(observed, predictions):.4f}"
    )

    print(
        f"Average precision: "
        f"{average_precision_score(observed, predictions):.6f}"
    )

print(
    f"Brier score:       "
    f"{brier_score_loss(observed, predictions):.6f}"
)

print(
    f"Observed fire rate: {observed.mean():.6f}"
)

print(
    f"Mean predicted PoF: {predictions.mean():.6f}"
)


# =============================================================================
# Populate global grids
# =============================================================================

print("\nPopulating global grids...")

pof_grid = np.full(
    (nj, ni),
    np.nan,
    dtype=np.float32,
)

observed_grid = np.full(
    (nj, ni),
    np.nan,
    dtype=np.float32,
)

# The flat index is the same row-major grid ordering used by pof_utils.
pof_grid.ravel()[index] = predictions
observed_grid.ravel()[index] = observed

populated = np.count_nonzero(np.isfinite(pof_grid))

print(f"Populated cells: {populated:,}")
print(f"Grid coverage: {100 * populated / (ni * nj):.2f}%")


# =============================================================================
# Plot predictions against observations
# =============================================================================

print("\nGenerating global map...")

# Convert longitude to -180..180 and sort.
plot_lons = (lons + 180.0) % 360.0 - 180.0
sort_idx = np.argsort(plot_lons)

plot_lons = plot_lons[sort_idx]

pof_plot = pof_grid[:, sort_idx]
observed_plot = observed_grid[:, sort_idx]

pof_max = np.nanpercentile(predictions, 99.5)
pof_max = max(pof_max, 1e-6)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(19, 8),
    subplot_kw={
        "projection": ccrs.Robinson()
    },
)

plot_fields = [
    (
        observed_plot,
        "Observed Fire Occurrence",
        "YlOrRd",
        0,
        1,
    ),
    (
        pof_plot,
        "Predicted Probability of Fire",
        "inferno",
        0,
        pof_max,
    ),
]

for ax, (values, title, cmap, vmin, vmax) in zip(
    axes,
    plot_fields,
):

    ax.set_global()

    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.5,
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.25,
        alpha=0.5,
    )

    mesh = ax.pcolormesh(
        plot_lons,
        lats,
        values,
        cmap=cmap,
        norm=colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        ),
        transform=ccrs.PlateCarree(),
        shading="auto",
        rasterized=True,
    )

    ax.set_title(title, fontsize=14)

    cb = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.8,
    )

    if title == "Observed Fire Occurrence":
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["No fire", "Fire"])

    cb.ax.tick_params(labelsize=9)

plt.suptitle(
    f"PoF Forecast Test: {test_date:%Y-%m-%d}",
    fontsize=17,
    y=0.96,
)

plt.tight_layout()

png_path = (
    OUTPUT_DIR / f"POF_forecast_{TEST_DATE}.png"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved map: {png_path}")


# =============================================================================
# Save predictions as parquet
# =============================================================================

print("\nSaving predictions...")

prediction_df = pd.DataFrame({
    "date": TEST_DATE,
    "lat": lat_flat[index],
    "lon": lon_flat[index],
    "AF": observed,
    "PoF": predictions,
})

parquet_path = (
    OUTPUT_DIR / f"POF_forecast_{TEST_DATE}.parquet"
)

prediction_df.to_parquet(
    parquet_path,
    index=False,
)

print(f"Saved parquet: {parquet_path}")


# =============================================================================
# Write PoF to GRIB2
# =============================================================================

print("\nWriting global PoF GRIB2...")

grib_path = (
    OUTPUT_DIR / f"POF_forecast_{TEST_DATE}.grib2"
)

output_ds = xr.Dataset(
    {
        "fire_probability": (
            ("latitude", "longitude"),
            pof_grid,
        )
    },
    coords={
        "latitude": lats,
        "longitude": lons,
    },
)

pof.write_grib_dataset(
    output_ds,
    grib_path,
    pof.PARAM_IDS,
    str(TEST_DATE),
    template_path=str(GRID_TEMPLATE),
)

print(f"Saved GRIB: {grib_path}")


# =============================================================================
# Finished
# =============================================================================

print("\n" + "=" * 65)
print("Single-day PoF forecast test completed successfully.")
print("=" * 65)

