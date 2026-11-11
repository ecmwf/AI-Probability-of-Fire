---
title: "Forecasting With PoF"
---
![logo](./images/ECMWF.png)

# Using Your PoF Model

Now that you have gathered your data and trained your model, you can use it for historical or future fire–occurrence predictions.

In this example, we generate predictions for an entire month using historical inputs. This type of prediction is often referred to as an **analysis prediction**, because it relies on observation-based data. An alternative—mirroring how PoF is used operationally at ECMWF—is to use *forecast* inputs. These are typically short-range (up to ~10 days), but the same workflow can be extended to seasonal, annual, or even decadal predictions.

---

## Loading Data

We begin by loading the previously trained model **POF_model.joblib**.

As with the training workflow, you must then load the required input data:

- **Static data** (e.g., population, roads), which do not change with time.
- **Time-varying data** (e.g., meteorological inputs, vegetation indices).  
  This section also demonstrates how to load active fire data, which can be useful for evaluation.

---

## Running the Prediction


After constructing a dataframe that matches the structure used during training, you can run the model in probability mode to estimate the likelihood of fire occurrence (`1`) using: model.predict_proba(X_pred)


---

## Saving Prediction as NetCDF

The final step is to save your prediction as a NetCDF file for further analysis or visualisation.  
In this example, metadata is not included, but you can add variable attributes, CRS information, or global metadata as required.

Your output file should be named:

**POF_prediction_YYYY_MM.nc**