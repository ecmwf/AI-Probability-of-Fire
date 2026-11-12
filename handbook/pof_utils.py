# (C) Copyright 2025 - ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Shared helpers for the PoF handbook notebooks."""

import datetime
import os
import subprocess
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import earthkit.data as ekd

DATA_DIR = "data"

ECBOX_URL = "https://sites.ecmwf.int/ecbox/POF_DATA/s/dav/data"
ECBOX_AUTH = (
    "ecbox",
    "1vXbp0THtxZkla015skARAqTnf2DxxoQS0qlwrgpKiIGpPRKLtXklYO39tOJTXtnZrxP16WZs99iSoFgTTAy3orpmjykOBAO7CVk5RQIWFvZFhiSv0Ev3O3",
)

# POP_2020 is GRIB2 on the ERA5-Land 0.1 degree grid (90N -> 90S, 0 -> 359.9E).
# Every GRIB we write for training is regridded to, and encoded from, this file.
GRID_TEMPLATE = f"{DATA_DIR}/CLIMATE/POP_2020.grib"

# GRIB encoding for each variable we write. Integers are local ECMWF
# parameters (discipline 0, category 191); dicts are full key sets.
PARAM_IDS = {
    "LFMC": 1,
    "LFMC_low": 2,
    "LFMC_high": 3,
    "DFMC_Foliage": 4,
    "DFMC_Wood": 5,
    "DFMC_1h": 6,
    "DFMC_10h": 7,
    "DFMC_100h": 8,
    "DFMC_1000h": 9,
    # ECMWF local active-fire parameter (paramId 210097).
    "ACTIVE_FIRE": {"discipline": 192, "parameterCategory": 210, "parameterNumber": 97},
    "ws": 11,
    "rh": 12,
    "t2m": 13,
    "tp": 14,
    "Live_Leaf": {"discipline": 2, "parameterCategory": 4, "parameterNumber": 27},
    "Live_Wood": {"discipline": 2, "parameterCategory": 4, "parameterNumber": 28},
    "Dead_Foliage": {"discipline": 2, "parameterCategory": 4, "parameterNumber": 29},
    "Dead_Wood": {"discipline": 2, "parameterCategory": 4, "parameterNumber": 30},
    "fire_probability": {"discipline": 2, "parameterCategory": 4, "parameterNumber": 43},
}

ERA5_LAND_VARIABLES = {
    "T2M": "2m_temperature",
    "D2M": "2m_dewpoint_temperature",
    "10U": "10m_u_component_of_wind",
    "10V": "10m_v_component_of_wind",
    "P": "total_precipitation",
}

XDS_VARIABLES = {
    "DFMC_MAP": "dead_fuel_moisture_content_group",
    "LFMC_MAP": "live_fuel_moisture_content_group",
    "FUEL_MAP": "fuel_group",
}

SPATIAL_DIMS = {"latitude", "longitude", "lat", "lon"}


############################################################
# DOWNLOADS
############################################################

def retrieve_ecbox_file(path, data_dir=DATA_DIR):
    output_path = Path(data_dir) / path
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{ECBOX_URL}/{path}", auth=ECBOX_AUTH)
    if response.status_code == 200:
        output_path.write_bytes(response.content)
    else:
        print(f"Failed to retrieve file: {ECBOX_URL}/{path}")
        print(response.status_code, response.text)
    return output_path


# Daily output -> (GRIB variable name, hourly ERA5-Land inputs, function of the hourly fields)
ERA5_LAND_DAILY = {
    "T2M": ("t2m", ("T2M",), lambda t2m: t2m),
    "RH": ("rh", ("T2M", "D2M"), lambda t2m, d2m: relative_humidity(t2m, d2m)),
    "WS": ("ws", ("10U", "10V"), lambda u10, v10: wind_speed(u10, v10)),
}


def retrieve_era5_land_month(client, year, month, data_dir=DATA_DIR, keep_hourly=False):
    """Retrieve hourly ERA5-Land fields as GRIB and store daily means of T2M, P, RH and WS as GRIB2."""
    stem = f"{year}_{month:02d}"
    daily_paths = {shortname: f"{data_dir}/{shortname}_{stem}.grib2" for shortname in [*ERA5_LAND_DAILY, "P"]}
    hourly_paths = {shortname: f"{data_dir}/{shortname}_{stem}_hourly.grib" for shortname in ERA5_LAND_VARIABLES}
    # Precipitation for the last day of the month ends at 00 UTC on the 1st of the next month
    next_month = datetime.date(year, month, 28) + datetime.timedelta(days=4)
    precip_end_path = f"{data_dir}/P_{stem}_end.grib"
    if all(os.path.exists(path) for path in daily_paths.values()):
        return

    def era5_land_request(variable, year, month, days, times):
        return {
            "variable": [variable],
            "year": f"{year}",
            "month": f"{month:02d}",
            "day": days,
            "time": times,
            "data_format": "grib",
            "download_format": "unarchived",
        }

    for shortname, variable in ERA5_LAND_VARIABLES.items():
        print(f"Retrieving hourly {shortname} for {year}-{month:02d}")
        if not os.path.exists(hourly_paths[shortname]):
            request = era5_land_request(
                variable, year, month, [f"{d:02d}" for d in range(1, 32)], [f"{h:02d}:00" for h in range(24)]
            )
            client.retrieve("reanalysis-era5-land", request, target=hourly_paths[shortname])
    if not os.path.exists(precip_end_path):
        request = era5_land_request(ERA5_LAND_VARIABLES["P"], next_month.year, next_month.month, ["01"], ["00:00"])
        client.retrieve("reanalysis-era5-land", request, target=precip_end_path)

    first_day = f"{year}{month:02d}01"
    for shortname, (name, inputs, func) in ERA5_LAND_DAILY.items():
        if os.path.exists(daily_paths[shortname]):
            continue
        print(f"Daily mean {shortname} for {year}-{month:02d}")
        with ExitStack() as stack:
            hourly = [first_variable(stack.enter_context(open_grib(hourly_paths[i]))) for i in inputs]
            daily = daily_mean(func, *hourly, year=year, month=month).rename(name)
        write_grib_dataset(daily.to_dataset(), daily_paths[shortname], PARAM_IDS, first_day, template_path=GRID_TEMPLATE)

    if not os.path.exists(daily_paths["P"]):
        print(f"Daily mean P for {year}-{month:02d}")
        with ExitStack() as stack:
            tp = [first_variable(stack.enter_context(open_grib(path))) for path in (hourly_paths["P"], precip_end_path)]
            daily = daily_precipitation(*tp, year=year, month=month).rename("tp")
        write_grib_dataset(daily.to_dataset(), daily_paths["P"], PARAM_IDS, first_day, template_path=GRID_TEMPLATE)

    if not keep_hourly:
        for path in [*hourly_paths.values(), precip_end_path]:
            os.remove(path)


def hourly_positions(variable):
    """(valid time, isel index) for every hourly slot, whatever its time/step layout."""
    dims = [dim for dim in variable.dims if dim not in SPATIAL_DIMS]
    valid_time = variable.valid_time.transpose(*dims).values
    for position in np.ndindex(valid_time.shape):
        yield pd.Timestamp(valid_time[position]), dict(zip(dims, position))


def daily_array(fields_by_day, like):
    days = sorted(fields_by_day)
    return xr.DataArray(
        np.stack([fields_by_day[day] for day in days]).astype(np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": days, "latitude": like.latitude, "longitude": like.longitude},
    )


def daily_mean(func, *variables, year, month):
    """Calendar-day mean of func(*hourly fields) for each day of one month."""
    hours_by_day = {}
    for valid, index in hourly_positions(variables[0]):
        if valid.year == year and valid.month == month:
            hours_by_day.setdefault(valid.normalize(), []).append(index)

    means = {}
    for day, indexes in hours_by_day.items():
        total, count = 0.0, 0
        for index in indexes:
            hour = [np.asarray(v.isel(index).values, dtype=float) for v in variables]
            # Missing time/step combinations are represented as NaN.
            if not np.isfinite(hour[0]).any():
                continue
            total, count = total + func(*hour), count + 1
        if count:
            means[day] = total / count
    return daily_array(means, variables[0])


def daily_precipitation(*variables, year, month):
    """Mean hourly precipitation for each calendar day of one month.

    ERA5-Land tp accumulates from 00 UTC, so the field valid at 00 UTC on day D + 1
    is the total for day D. Dividing by 24 gives the mean of the hourly amounts.
    """
    totals = {}
    for variable in variables:
        for valid, index in hourly_positions(variable):
            day = (valid - pd.Timedelta(days=1)).normalize()
            if valid.hour == 0 and day.year == year and day.month == month:
                total = np.asarray(variable.isel(index).values, dtype=float)
                if np.isfinite(total).any():
                    totals[day] = total / 24
    return daily_array(totals, variables[0])


def to_training_grid(variable):
    """Open Data runs longitude -180..180, the training grid runs 0..360."""
    return variable.assign_coords(longitude=variable.longitude.round(4) % 360).sortby("longitude")


def forecast_daily_means(t2m, d2m, u10, v10, tp, run_date, forecast_days):
    """Daily means over forecast days 1..N from 3/6-hourly Open Data, on the training grid.

    tp is accumulated from the start of the run, so it is de-accumulated over the
    calendar day (00 to 24 UTC) into a mean hourly amount, matching daily_precipitation.
    """
    t2m, d2m, u10, v10, tp = (to_training_grid(v) for v in (t2m, d2m, u10, v10, tp))

    def at(variable, hours):
        return np.asarray(variable.sel(step=np.timedelta64(hours, "h")).values, dtype=float)

    available_steps = sorted(int(pd.Timedelta(step) / pd.Timedelta(hours=1)) for step in t2m.step.values)
    fields = {name: [] for name in ("t2m", "tp", "rh", "ws")}
    for day in range(1, forecast_days + 1):
        steps = [step for step in available_steps if 24 * day <= step < 24 * (day + 1)]
        if not steps:
            raise ValueError(f"No forecast fields available for day {day}")
        end_step = 24 * (day + 1)
        if end_step not in available_steps:
            raise ValueError(f"No precipitation accumulation endpoint at step {end_step}")
        fields["t2m"].append(np.mean([at(t2m, s) for s in steps], axis=0))
        fields["rh"].append(np.mean([relative_humidity(at(t2m, s), at(d2m, s)) for s in steps], axis=0))
        fields["ws"].append(np.mean([wind_speed(at(u10, s), at(v10, s)) for s in steps], axis=0))
        fields["tp"].append(np.maximum(at(tp, end_step) - at(tp, 24 * day), 0.0) / 24)

    times = [np.datetime64(run_date + datetime.timedelta(days=day)) for day in range(1, forecast_days + 1)]
    return xr.Dataset(
        {name: (("time", "latitude", "longitude"), np.stack(values)) for name, values in fields.items()},
        coords={"time": times, "latitude": t2m.latitude.values, "longitude": t2m.longitude.values},
    )


def retrieve_xds_month_to_grib(client, year, month, data_dir=DATA_DIR):
    """Retrieve DFMC, LFMC and fuel load from the XDS and store them as GRIB2 on the training grid."""
    stem = f"{year}_{month:02d}"
    template = read_grib(GRID_TEMPLATE, decode_times=False)
    target = xr.Dataset(coords={"lat": template.latitude.values, "lon": template.longitude.values})

    for shortname, variable in XDS_VARIABLES.items():
        grib_path = f"{data_dir}/{shortname}_{stem}.grib2"
        if os.path.exists(grib_path):
            continue
        zip_path = f"{data_dir}/{shortname}_{stem}.zip"
        nc_path = f"{data_dir}/{shortname}_{stem}.nc"
        request = {
            "variable": [variable],
            "version": ["1"],
            "year": [f"{year}"],
            "month": [f"{month:02d}"],
        }
        client.retrieve("derived-fire-fuel-biomass", request, target=zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        print(f"Regridding {shortname} for {year}-{month:02d}")
        with xr.open_dataset(nc_path) as ds:
            ds = ds[[name for name in ds.data_vars if name in PARAM_IDS]]
            # XDS longitudes run -180..180, the training grid runs 0..360
            ds = ds.assign_coords(lon=ds.lon % 360).sortby("lat").sortby("lon")
            ds_regrid = ds.interp_like(target, method="linear")
            write_grib_dataset(ds_regrid, grib_path, PARAM_IDS, f"{year}{month:02d}01", template_path=GRID_TEMPLATE)
        os.remove(zip_path)
        os.remove(nc_path)


############################################################
# GRIB INPUT / OUTPUT
############################################################

def grib_filter(name):
    """GRIB metadata keys for a variable in PARAM_IDS."""
    parameter = PARAM_IDS[name]
    if isinstance(parameter, dict):
        return dict(parameter)
    return {"discipline": 0, "parameterCategory": 191, "parameterNumber": parameter}


def open_grib(path, filters=None, decode_times=True):
    fields = ekd.from_source("file", os.fspath(path)).to_fieldlist()
    if filters:
        fields = fields.sel({f"metadata.{key}": value for key, value in filters.items()})
    if not len(fields):
        raise ValueError(f"No GRIB fields in {path} match filters {filters or {}}")

    dataset = fields.to_xarray(
        engine="earthkit",
        decode_times=decode_times,
        time_dims=["forecast_reference_time", "step"],
        squeeze=False,
        rename_dims={"forecast_reference_time": "time"},
        add_valid_time_coord=True,
        allow_holes=True,
    )
    singleton_dims = [
        dim for dim, size in dataset.sizes.items() if size == 1 and dim not in SPATIAL_DIMS
    ]
    return dataset.squeeze(dim=singleton_dims, drop=True) if singleton_dims else dataset


def read_grib(path, filters=None, variable_name=None, decode_times=True, select=None):
    with open_grib(path, filters, decode_times) as dataset:
        if variable_name is None:
            variable_name = first_variable(dataset).name
        variable = dataset[variable_name]
        if select:
            variable = variable.sel(select)
        return variable.load()


def first_variable(dataset):
    return dataset[next(iter(dataset.data_vars))]


def time_dim(variable):
    dims = [dim for dim in variable.dims if dim not in SPATIAL_DIMS]
    for dim in ("time", "step"):
        if dim in dims and variable.sizes[dim] > 1:
            return dim
    return dims[0]


def field_at(variable, index):
    return np.squeeze(variable.isel({time_dim(variable): index}).values)


def regrid_with_cdo(source_path, output_path, grid_file, method="remapnn"):
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    temp_path = output_path.with_suffix(".tmp.grib2")
    try:
        subprocess.run(
            ["cdo", "-f", "grb2", f"{method},{grid_file}", str(source_path), str(temp_path)],
            check=True,
        )
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return output_path


def write_grib_dataset(ds, path, parameter_ids, valid_date, template_path=None):
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat = np.asarray(ds[lat_name].values)
    lon = np.asarray(ds[lon_name].values)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("GRIB output currently requires 1D latitude and longitude coordinates")

    lat_step = abs(lat[1] - lat[0]) if len(lat) > 1 else 0
    lon_step = abs(lon[1] - lon[0]) if len(lon) > 1 else 0
    if (len(lat) > 2 and not np.allclose(np.diff(lat), np.diff(lat)[0])) or (
        len(lon) > 2 and not np.allclose(np.diff(lon), np.diff(lon)[0])
    ):
        raise ValueError("GRIB output currently requires a regular latitude/longitude grid")

    output_date = datetime.datetime.strptime(valid_date, "%Y%m%d")
    template = "regular_ll_sfc_grib2"
    if template_path and os.path.exists(template_path):
        templates = ekd.from_source("file", template_path).to_fieldlist()
        if not len(templates):
            raise ValueError(f"No GRIB messages found in {template_path}")
        template = templates[0]
        if template.shape != (len(lat), len(lon)):
            raise ValueError(f"Template grid {template.shape} does not match output grid {(len(lat), len(lon))}")

    encoder = ekd.create_encoder("grib")
    with open(path, "wb") as output_file:
        for variable_name, variable in ds.data_vars.items():
            time_dims = [dim for dim in variable.dims if dim not in SPATIAL_DIMS]
            if len(time_dims) > 1:
                raise ValueError(f"GRIB output supports at most one time dimension, found {time_dims}")
            if time_dims:
                field = variable.transpose(time_dims[0], lat_name, lon_name).values
            else:
                field = variable.transpose(lat_name, lon_name).values[np.newaxis, ...]

            parameter = parameter_ids[variable_name]
            parameter_metadata = (
                dict(parameter)
                if isinstance(parameter, dict)
                else {"discipline": 0, "parameterCategory": 191, "parameterNumber": parameter}
            )
            grid_metadata = {}
            if isinstance(template, str):
                grid_metadata = {
                    "gridType": "regular_ll",
                    "Ni": len(lon),
                    "Nj": len(lat),
                    "latitudeOfFirstGridPointInDegrees": float(lat[0]),
                    "longitudeOfFirstGridPointInDegrees": float(lon[0]),
                    "latitudeOfLastGridPointInDegrees": float(lat[-1]),
                    "longitudeOfLastGridPointInDegrees": float(lon[-1]),
                    "iDirectionIncrementInDegrees": float(lon_step),
                    "jDirectionIncrementInDegrees": float(lat_step),
                    "iScansNegatively": int(lon[-1] < lon[0]),
                    "jScansPositively": int(lat[-1] > lat[0]),
                    "bitsPerValue": 24,
                }

            for index, values in enumerate(field):
                field_date = output_date + datetime.timedelta(days=index)
                metadata = parameter_metadata | grid_metadata | {
                    "dataDate": int(field_date.strftime("%Y%m%d")),
                    "dataTime": 0,
                    "forecastTime": 0,
                }
                encoded = encoder.encode(
                    values=np.asarray(values, dtype=float),
                    template=template,
                    metadata=metadata,
                    missing_value=9999.0,
                )
                encoded.to_file(output_file)


############################################################
# DERIVED VARIABLES
############################################################

def relative_humidity(t2m, d2m):
    """Relative humidity (%) from 2m temperature and dewpoint in Kelvin."""
    t2m_c = t2m - 273.15
    d2m_c = d2m - 273.15
    es = 10 ** (7.5 * t2m_c / (237.7 + t2m_c)) * 6.11
    e = 10 ** (7.5 * d2m_c / (237.7 + d2m_c)) * 6.11
    return (e / es) * 100


def wind_speed(u10, v10):
    # Squared wind speed (no square root): the trained model expects this feature.
    return u10**2 + v10**2


############################################################
# TRAINING DATA
############################################################

# Column -> (file name pattern, GRIB metadata filter)
TRAINING_INPUTS = {
    "AF": ("ACTIVE_FIRE_MAP_{stem}_R.grib2", grib_filter("ACTIVE_FIRE")),
    "PR": ("P_{stem}.grib2", grib_filter("tp")),
    "T2": ("T2M_{stem}.grib2", grib_filter("t2m")),
    "RH": ("RH_{stem}.grib2", grib_filter("rh")),
    "WS": ("WS_{stem}.grib2", grib_filter("ws")),
    "FU_LL": ("FUEL_MAP_{stem}.grib2", grib_filter("Live_Leaf")),
    "FU_LW": ("FUEL_MAP_{stem}.grib2", grib_filter("Live_Wood")),
    "FU_DF": ("FUEL_MAP_{stem}.grib2", grib_filter("Dead_Foliage")),
    "FU_DW": ("FUEL_MAP_{stem}.grib2", grib_filter("Dead_Wood")),
    "DF": ("DFMC_MAP_{stem}.grib2", grib_filter("DFMC_Foliage")),
    "DW": ("DFMC_MAP_{stem}.grib2", grib_filter("DFMC_Wood")),
    "LF": ("LFMC_MAP_{stem}.grib2", grib_filter("LFMC")),
}


def check_same_grid(reference, variable, name):
    for coord in ("latitude", "longitude"):
        if not np.allclose(reference[coord].values, variable[coord].values, atol=1e-3):
            raise ValueError(f"{name} is not on the same {coord} grid as {GRID_TEMPLATE}")


def build_training_samples(years, months, data_dir=DATA_DIR, sample_frac=1.0):
    """Combine climate, vegetation and anthropogenic fields into one row per fuelled pixel per day."""
    population = read_grib(f"{data_dir}/CLIMATE/POP_2020.grib", decode_times=False)
    roads = read_grib(f"{data_dir}/CLIMATE/road_density_2015_agg_r.grib", decode_times=False)
    check_same_grid(population, roads, "road_density")
    static = {"PO": population.values.flatten(), "RD": roads.values.flatten()}

    all_samples = []
    for year in years:
        print(f"Processing year {year}...")
        for month in months:
            stem = f"{year}_{month:02d}"
            with ExitStack() as stack:
                inputs = {}
                for column, (pattern, filters) in TRAINING_INPUTS.items():
                    dataset = stack.enter_context(open_grib(f"{data_dir}/{pattern.format(stem=stem)}", filters))
                    inputs[column] = first_variable(dataset)
                    check_same_grid(population, inputs[column], column)

                days = inputs["AF"].sizes[time_dim(inputs["AF"])]
                for i in range(days):
                    fields = {column: field_at(variable, i).flatten() for column, variable in inputs.items()}
                    # Any detected fire (possible or probable) is a fire; no fire and missing are not
                    fields["AF"] = np.isin(fields["AF"], (1, 2)).astype(float)

                    # Keep only pixels with some fuel
                    total_fuel = fields["FU_LL"] + fields["FU_LW"] + fields["FU_DF"] + fields["FU_DW"]
                    mask = total_fuel > 0.0

                    dfx = pd.DataFrame(
                        {column: values[mask] for column, values in (fields | static).items()},
                        dtype=float,
                    )
                    dfx.dropna(inplace=True)

                    if len(dfx) > 0:
                        n_samples = max(1, int(len(dfx) * sample_frac))
                        all_samples.append(dfx.sample(n=n_samples, random_state=1))

    if not all_samples:
        raise ValueError("No samples generated. Check data masks or paths.")
    return pd.concat(all_samples, ignore_index=True)
