# Feature Selection and Classification Pipelines in Scikit-Learn

This repository contains the implementation of two advanced Data Mining workflows focused on data preparation, feature engineering, and building robust machine learning pipelines using `scikit-learn`. 

The methodologies and structure follow the guidelines from the book *Learning Data Mining with Python* (Layton, 2015).

## 🚀 Project Overview

The project is structured into two main experimental sections:

### 1. Feature Engineering & Selection (Adult Dataset)
* **Objective:** Predict whether an individual's annual earnings exceed \$50K based on demographic and employment data.
* **Key Implementation Steps:**
  * **Data Preprocessing & Discretization:** Loaded the raw `adult.data`, assigned manual headers, handled missing values, and created a custom categorical feature `LongHours` (True if `Hours-per-week` > 40).
  * **Variance Thresholding:** Demonstrated feature filtering using `VarianceThreshold` on an artificial matrix to eliminate zero-variance columns.
  * **Feature Selection Comparison:** Evaluated and compared two statistical selection methods via `SelectKBest` (reducing from 5 numerical features to $k=3$):
    * **Chi-Squared ($\chi^2$)** test.
    * **Pearson Correlation Coefficient** using a custom-built multivariate function.
  * **Evaluation & Inference:** Trained a `DecisionTreeClassifier` validated with a 5-fold cross-validation scheme. The Chi² feature selection yielded higher accuracy (**82.9%**) compared to Pearson (**77.2%**). The finalized model was then used to predict the salary class of a custom synthetic profile.

### 2. Dimensionality Reduction & Custom Pipelines (Internet Ads Dataset)
* **Objective:** Classify whether a web image is an advertisement or normal content within a high-dimensional feature space (>1,500 features).
* **Key Implementation Steps:**
  * **Data Cleaning & Imputation:** Implemented rigorous type conversion to handle corrupted values (`NaN`) using `SimpleImputer` with a median strategy.
  * **Dimensionality Reduction (PCA):** Applied Principal Component Analysis (PCA) to project the 1,558 features down to 5 principal components (explaining ~85.6% and ~14.2% variance in the first two components). Visualized class separation using a Matplotlib scatter plot.
  * **Custom Transformer (`MeanDiscrete`):** Engineered a custom scikit-learn compatible transformer by extending `TransformerMixin`. This class binarizes continuous columns using their column-specific mean. Includes formal unit testing verified via `assert_array_equal`.
  * **Robust Pipeline Integration:** Enchained the Imputer, custom `MeanDiscrete` transformer, and a `DecisionTreeClassifier` into a unified `Pipeline` to strictly prevent data leakage during cross-validation.
  * **Results:** The end-to-end pipeline achieved an exceptional cross-validation accuracy of **95.1%**.

---

## 🛠️ Tech Stack & Libraries

* **Python 3.x**
* **Pandas** & **NumPy** — Data ingestion, cleaning, and matrix manipulations.
* **Scikit-Learn** — Preprocessing, feature selection, PCA, pipelines, and evaluation.
* **SciPy** — Statistical calculations for Pearson correlation.
* **Matplotlib** — 2D visualization of principal components.

---

## 📂 Repository Structure

```text
├── Data/
│   ├── Adult/
│   │   └── adult.data
│   └── Ads/
│       └── ad.data
├── src/
│   ├── feature_selection_adult.py
│   └── ads_pipeline_pca.py
├── README.md
