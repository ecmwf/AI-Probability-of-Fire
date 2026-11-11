---
title: The data behind PoF
---

![logo](./images/ECMWF.png)

__A  data driven model needs data__

:::{note}
You will not be able to reproduce the exact configuration run at ECMWF, as it uses some proprietary data that we are unable to share. However, most of the data used to train PoF are publicly available and free to use. Below, we provide guidance on the datasets we recommend for getting started on your PoF journey.
:::

:::{figure} ./images/data-sources-pof.png
:align: center
:width: 70%
Data sources to train the PoF
:::
    
<div style="text-align:center; margin:40px 0; font-weight:bold;">
    Weather Data
</div>

<a href="https://doi.org/10.1175/JAMC-D-15-0297.1" target="_blank">
In fuel-rich environments, weather is the dominant control of fire behaviour.</a>

To begin your data collection, we recommend focusing on four key environmental variables: __temperature, precipitation, relative humidity, and wind speed.__ These variables represent the fundamental controls of how fire behaviour, and they are at the core of nearly all fire-weather indices used operationally around the world. Each variable plays a distinct role in shaping landscape flammability. 

__Temperature__ is one of the most immediate drivers: as temperatures rise, vegetation and soils lose moisture more quickly, and both live and dead fuels become more prone to ignition. Prolonged warm conditions can push even healthy vegetation into moisture stress, making it more combustible.

__Precipitation__ acts in the opposite direction. It replenishes moisture in fuels and slows the drying process, reducing the likelihood that a spark will result in a fire. But precipitation also shapes fire risk over longer periods. <a href="https://doi.org/10.1111/gcb.70481" target="_blank"> Wet springs or wet years stimulate vegetation growth, building up large fuel loads which can later burn if drought conditions return. </a>

__Relative humidity__ is closely linked to fuel moisture, particularly for fine fuels like grasses and leaf litter. When the air is dry, these fuels lose moisture quickly and ignite much more easily. When humidity is higher, even small increases in atmospheric moisture can keep fine fuels from reaching the critical dryness needed to sustain fire spread. Because humidity changes rapidly over the course of a day, it introduces fast variations in fire danger. The data used here derives relative humidity by first computing the saturation vapor pressure at the air temperature and the actual vapor pressure at the dew point (both via a Clausius–Clapeyron/Magnus approximation), then taking the ratio of actual to saturation vapor pressure multiplied by 100.

Finally, __wind speed__ is the most dynamic and often the most dangerous factor. Wind supplies oxygen to flames, drives heat into unburnt areas, and can carry embers far ahead of the main fire front, creating spot fires and accelerating spread. Strong winds can transform a small ignition into a fast-moving, intense wildfire within minutes.

Together, these __four variables__  describe the conditions under which fires can ignite and spread. They provide a solid, intuitive foundation for understanding landscape flammability and are a suitable starting point for anyone beginning their PoF modelling journey. They also provide a data-driven alternative to the Fire Weather Index.

---

<div style="text-align:center; margin:40px 0; font-weight:bold;">
    Fuel Data
</div>

<a href="https://doi.org/10.1038/s41467-025-58097-7" target="_blank">
In absence of any other source, fuel is the most important control globally on fire activity.</a>

To model how vegetation influences wildfire behaviour, two key aspects must be captured, how much fuel is present and how dry that fuel is.

<a href="https://doi.org/10.5194/bg-21-279-2024" target="_blank"> __Fuel load__ </a> represents the total mass of above-ground biomass available to burn. It includes:

🌿 Foliage (live and dead)
🌳 Wood (live and dead)

We estimate fuel load by combining Satellite-derived Above-Ground Biomass (AGB) from ESA-CCI (2010 baseline), and Daily Net Ecosystem Exchange (NEE) from the ECLand land-surface model (forced by ERA5) with <a href="https://doi.org/10.5194/acp-16-10399-2016" target="_blank"> bias correction from atmospheric inversions. </a>

This approach allows us to reconstruct daily biomass evolution at ~9 km resolution. AGB is then partitioned into live and dead components using vegetation-type-specific ratios and leaf area index. This gives a dynamic estimate of foliage and wood fuel loads, consistent with fire-modelling practices.

<a href="https://doi.org/10.5194/bg-21-279-2024" target="_blank"> __Fuel moisture__ </a>  represents the total mass of biomass with respect to water. It is usually divided in the moisture of the linving plant and the moisture in dead biomass 

__Live Fuel Moisture Content (LFMC)__

💦🍃 LFMC expresses how much water is contained in living vegetation relative to its dry mass. It determines how easily plants ignite and how fast a fire can spread.
We use a semi-empirical model, trained on the <a href="https://doi.org/10.1038/s41597-019-0164-9" target="_blank"> Globe-LFMC in-situ dataset</a>, to estimate daily LFMC from:

🌿 **Leaf Area Index (LAI)**

💧 **Soil moisture (root-weighted)**

🌳 **Vegetation type**

The model ensures physically realistic moisture ranges and captures seasonal vegetation responses to drought and growth cycles.

<a href="https://doi.org/10.5194/bg-21-279-2024" target="_blank"> __Dead Fuel Moisture Content (DFMC)__ </a>

DFMC describes the moisture content of dead leaves, litter, and woody debris—fuels that respond directly to weather.
We generalize the <a href="https://doi.org/10.1139/x00-032" target="_blank">Nelson (2000)</a> physical model to estimate DFMC for standard dead-fuel classes (1h, 10h, 100h, 1000h). Short-lag fuels (1h, 10h), which represent dead foliage, respond to fast humidity changes. Long-lag fuels (100h, 1000h), which represent dead wood, respond to multi-day weather patterns. DFMC for foliage and wood is derived by weighting appropriate fuel classes based on vegetation type.

---

<div style="text-align:center; margin:40px 0; font-weight:bold;">
    Sources of Ignitions
</div>

**Road Density**

Roads facilitate human access to wildland areas, increasing the likelihood of ignitions through activities such as agriculture, forestry, and recreation. However, they may also contribute to fire suppression efficiency and serve as firebreaks, creating a complex relationship between road density and fire outcomes.

The <a href="https://doi.org/10.1088/1748-9326/aabd42" target="_blank"> Global Roads Inventory Project (GRIP) dataset</a>, which provides a global map of road networks at ~8 km resolution, has been regridded for use here.

**Population Density**

Around 90% of wildfires are human‑ignited, making human presence a useful indicator of potential fire occurrence. Fire incidence generally rises with population density but eventually saturates and at very high densities, suppression efforts typically limit the potential for large wildfires.

The data provided here are derived from the <a href="https://doi.org/10.7927/H49C6VHW" target="_blank"> Gridded Population of the World, Version 4 (GPWv4) dataset</a>, which provides gridded population estimates at ~1 km resolution in 5-year intervals from 2000 to 2020.

---

<div style="text-align:center; margin:40px 0; font-weight:bold;">
    Fire Data
</div>

Various sensors and fire products are available for use in a PoF-style system. Here we attempt a binary classifier model, that is an indication of the probability of a fire yes/no. As such we use hotspot detections rather than burned area.

__Active fire (AF)__ detections were taken from the <a href="https://doi.org/10.1016/j.rse.2016.02.054" target="_blank"> MODIS MCD14ML product</a>, which provides daily fire hotspot locations based on thermal anomalies detected at 1 km resolution. These data were gridded to coarser daily resolution and represented as binary values (1 = at least one hotspot detected, 0 = none). We applied quality assurance flags to exclude low-confidence detections and removed spurious signals when possible.

:::{important}
  <h3 style="margin-top:0; color:#003e74;">🎯 Final result</h3>
  <p style="margin:0; font-size:1.05em; color:#003e74;">
    A set of environmental predictors and fire activity collocated dataset 
  </p>
</div>
