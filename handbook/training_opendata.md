---
title: How we train an XGBoost Model for PoF
# subtitle: Reproducible notebooks for building local PoF models with your own data
---

![logo](./images/ECMWF.png)

PoF uses XGBoost, a powerful gradient-boosted decision-tree algorithm widely applied to tabular environmental data.

The training procedure follows a simple, reproducible workflow based on a probabilistic classifier:

### Prepare the training dataset 

You will assemble a table where each row represents a gridcell in space and time (daily & 9km grid) and includes:

---

#### Predictors (features)
  Fuel variables, meteorological variables, and ignition proxies. All of which are described in the retrieving_data documentation.


---

#### Target (label)
  Binary fire occurrence within the gridcell on the given day, where a single or multiple counts equate to fire detection: <code>1</code> = fire detected, <code>0</code> = no fire.


The data generation script should have synthesised your data into a DataFrame stored in a Parquet file, which is now ready for training.

### Split the data

---

##### Dataset Splits

- **Training set** → used to fit the model
- **Test set** → final skill evaluation

---

We typically use a random stratified split.

### Define the XGBoost model

We configure the key XGBoost Hyperparameters:

- **max_depth** – tree complexity
- **learning_rate** – how fast the model learns
- **n_estimators** – number of boosting rounds
- **objective="binary:logistic"** – required to output probabilities

---

You will need to adjust these parameters depending on your region and data volume.

### Generate PoF predictions

Once trained, the model outputs a probability between 0 and 1 representing the likelihood that at least one fire will occur under the given conditions withing a gridcell on a given day.

These are the core PoF predictions you will visualise and evaluate.

---

### Evaluate the model 

We can assess the model skill using tools such as:
- ROC curve
- AUC score
- Reliability diagrams
- Confusion matrix

However we only provide examples for a subset of these metrics. Reliability methods are highly recommended given the unbalanced nature of PoF, however it is important to note these can be more computationally expensive than some other methods. 

Interpretation of these diagnostics in the context of fire risk needs to be carefully considered.

The script saves the trained model for reuse (POF_model.joblib) . Later versions of XGboost may save as json or other file formats but can be saved and reused in the same or similar way.

---

### 🎯 Final result

A trained, validated XGBoost model providing daily probability-of-fire estimates
from environmental predictors
