---
title: 🔥 Probability of Fire Jupyter Book
# subtitle: Reproducible notebooks for building local PoF models with your own data
---

![logo](./images/ECMWF.png)

<br>
<!-- 
<div style="
    background: linear-gradient(90deg, #003e74 0%, #006bb3 100%);
    padding: 35px 30px;
    border-radius: 10px;
    margin: 40px 0;
    color: white;
    text-align: center;
">
  <h1 style="margin:0; font-size: 2.5em; font-weight:600;">
    🔥 Probability of Fire — Toolbox
  </h1>
  <p style="margin-top:10px; font-size:1.3em; opacity:0.95;">
    Reproducible notebooks for building local PoF models with your own data
  </p>
</div>

:::{figure} ./images/PoF-light.png
:align: center
:width: 70%
POF Toolbox
:::


__The Probability of Fire (PoF) Toolbox provides a collection of notebooks, datasets and workflows to help users explore, reconstruct and run a PoF model.__

Developed around the machine-learning framework used at ECMWF, the toolbox provides two complementary approaches to working with PoF. Whether you want to investigate historical fire predictions or build a workflow to generate predictions using current data, the notebooks provide a practical starting point.

Two ways to work with PoF

1. Historical PoF reconstruction

Explore how PoF would have performed over previous years using historical environmental conditions.

This workflow combines ERA5 reanalysis data, satellite-informed land-surface information and historical fuel datasets to reconstruct the input conditions required by PoF. By applying the model to these historical conditions, users can generate spatially and temporally consistent estimates of fire probability.

2. Real-time PoF forecasting

Build and run a PoF workflow using openly available datasets.

This workflow provides a template for preparing the required input data and running PoF with current environmental conditions. It brings together the data preparation and modelling steps needed to generate fire probability predictions, providing a starting point for users interested in developing their own operational or near-real-time applications.

Unlike the historical reconstruction workflow, this approach is designed around the availability of current input data rather than retrospective reanalysis. It therefore provides a foundation for experimenting with real-time fire prediction using openly accessible data sources.

# Getting started

Both workflows are designed to make the PoF framework more accessible, providing practical examples of how the model can be applied beyond its implementation at ECMWF.

The notebooks guide users through the relevant data preparation, model execution and analysis steps. They are intended to be adaptable, allowing users to explore the framework, experiment with alternative datasets and develop their own applications.

# Wildfire prediction  

Given their chaotic nature, how can we know when and where wildfires will occur? That is a question that has puzzled wildfire forecasters for decades and now, thanks to advances in <a href="https://doi.org/10.1029/2023GL107929" target="_blank"> **machine learning**</a>, we are one step closer to answering it, although we still have some way to go.
 
The topic of **wildfire forecasting** is not a new one. However, in recent years media attention around the subject has grown as unprecedented wildfire seasons in Australia (2019/2020) and Canada (2023) and Europe (2025) have resulted in widespread devastation of local ecosystems and communities.

These events have far-reaching consequences for both air quality and greenhouse gas emissions

# Traditional fire forecasting

For nearly half a century, fire danger forecasts have relied on a method that links weather conditions with fire activity to create an index of fire risk, with the <a href="https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi" target="_blank"> **Canadian Fire Weather Index (FWI)**</a>
 being the most widely used. However, this approach has its limitations. 
 
:::{figure} ./images/fire-triangle.png
:align: center
:width: 70%
The wildfire triangle: fuel × dryness × ignition
:::

**A fire needs fuel**, and for wildfires that fuel is both **living** and **dead** vegetation. The abundance and arrangement of that fuel is known as the ‘**fuel bed**’. If all else is equal, the drier the fuel bed, the higher the fire risk. The FWI primarily estimates the state of fuel moisture based on meteorological conditions affecting a predetermined typical Canadian forest fuel bed. However, a typical fuel bed does not capture controls such as the moisture levels in living vegetation, the composition of vegetation types, or the actual abundance of available fuel. Consequently, the FWI tends to overestimate fire risk in areas with limited fuel. Moreover, because the FWI was originally developed for Canadian forests, its applicability becomes complex when extrapolated to different ecosystems.

All this is before we even consider the factors that might start a wildfire in the first place (known as ignitions). Ninety per cent of ignitions are caused by the unpredictable behaviour of humans, making them chaotic in nature and hard to predict.

```{note}
These shortcomings highlight the need for more advanced forecasting techniques to accurately assess wildfire risk across diverse landscapes.
```

# Machine learning evolution

In recent months, there has been a remarkable growth in the integration of machine learning into weather forecasting systems. **With the intricate dynamics governing wildfires, it seems only natural to explore similar applications of machine learning in fire forecasting.**

We have developed a new tool, known as <a href="https://doi.org/10.1029/2023GL107929" target="_blank">**Probability of Fire, or PoF,**</a> which uses machine learning techniques to effectively forecast fire occurrence globally at high resolution, up to ten days in advance. 
```{note}
A key strength of PoF is found not only in the accurate predictions but also in computational cost. The model itself is extremely cheap to run compared with more traditional physical models, which allows us to perform global 1 km forecasts.
```

:::{figure} ./images/pof.png
:align: center
:width: 70%
Data in and out of the PoF
:::
 

The foundation of PoF lies in its utilisation of diverse datasets, including information from the ECMWF Integrated Forecasting System (IFS), land cover data, and a <a href="https://doi.org/10.5194/bg-21-279-2024" target="_blank"> **newly developed fuel characteristic model**</a>.

The training of PoF is made possible thanks to the wealth of historic observations of active fires from satellites. The model mimics what it anticipates satellites will detect in the next few days. Consequently, it could only ever hope to perform as well as the satellite, which emphasises the importance of accurate satellite data for training. 

