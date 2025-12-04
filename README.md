# MSc Project: Premier League Match Outcome Prediction
## (University of Hertfordshire - Data Science)

### 1. Project Overview

This repository contains the complete analytical pipeline for the MSc Data Science project focused on predicting outcomes (Home Win, Draw, Away Win) in the English Premier League.

**Goal:** To determine the predictive power of engineered performance statistics (post-match) and identify the primary drivers of victory using interpretable Machine Learning.

### 2. Key Findings & Results

The final, optimized model successfully demonstrated significant predictive power against the baseline.

* **Final Model:** Optimized **Logistic Regression** Classifier (chosen via GridSearchCV).
* **Accuracy:** **56%** (Significantly outperforms the random baseline of 33%).
* **Primary Driver:** **ShotsOnTargetDiff** (Correlation of 0.44), proving that shooting accuracy is the most influential metric.
* **Advanced Analysis:** Utilized **SHAP** (SHapley Additive exPlanations) to prove the model's logic.

### 3. Data Source and Preprocessing

The primary dataset (`England CSV.csv`) contains over 12,000 historical match records (2010–2024).

* **Data Strategy:** Features were engineered into difference metrics (e.g., `ShotsDiff`) to capture *relative dominance* between Home and Away teams.
* **Bias Mitigation:** Team names were intentionally excluded from training to prevent the model from learning historical bias, forcing it to focus solely on performance metrics.

### 4. Requirements and Setup

To run this project, you need the following libraries installed in your Python environment:

```bash
# Install the core libraries
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap
