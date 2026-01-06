# MSc Project: Premier League Match Outcome Prediction
## MSc Data Science Final Project | University of Hertfordshire

# 1. Project Overview

This research addresses the complex challenge of predicting professional football match outcomes using historical performance data only. The project investigates the predictive power of machine learning to forecast results (Home Win, Draw, Away Win) and identifies the most significant quantifiable indicators of success.

**Goal:** To determine the predictive power of engineered performance statistics (post-match) and identify the primary drivers of victory using interpretable Machine Learning.

# 2. Key Findings & Results

**Champion Model:** Optimized Logistic Regression, selected via GridSearchCV for its superior generalization on noisy sports data.

**Accuracy:** 56.97% — significantly outperforming the random baseline of 33.3%.

**Primary Predictor:** Shots on Target Difference (Correlation: 0.44), proving that offensive efficiency is the most significant driver of outcomes.

**The "Draw" Paradox:** The model achieved a recall of 0.00 for Draws, confirming that draws represent "statistical parity" and act as random noise in performance-driven models.

# 3. Data & Feature Engineering

**Dataset:** The "England CSV" dataset containing 12,000+ matches from 1993 to 20257777.

**Relative Metrics:** Raw stats were transformed into "Difference" features (e.g., ShotsOnTargetDiff) to capture game flow and dominance8888.

**Formula:** $Metric_{Diff} = Home_{Stat} - Away_{Stat}$9.

**Bias Mitigation:** Team names and specific player identities were excluded to force the model to learn strictly from on-field performance rather than historical reputation10101010.

# 4. Visualization:

**The analysis utilizes several diagnostic plots to interpret model logic:**

**Correlation Matrix:** Identifying the significance of offensive vs. disciplinary metrics.

**Confusion Matrix:** Highlighting the high recall (0.87) for Home Wins.

**SHAP Summary Plot:** Providing "Explainable AI" (XAI) to show feature impacts on predictions.

# 5. Repository Structure

**England.csv:** Historical EPL Dataset (1993-2025) 

**Final Project Report.pdf:** Full Academic Project Report

Main_Code.py: Python Source Code implementation 

**README.md:** Project Documentation

# 6. Setup and Installation
**To replicate this study, ensure you have Python 3.8+ installed, then run:**

### Clone the repository
git clone https://github.com/sriharimohan/Premier-League-Match-Prediction-using-ML-Models.git

### Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap

# 7. Future Work

**Team Form:** Implementing Rolling Averages to capture transient states of momentum and tactical execution.

**Granular Data:** Integrating Player-Level Data (injuries, suspensions, and starting lineups) to refine predictive sensitivity.



### Student: Srihari Mohan (SRN: 23069726) 
### Supervisor: Ralf Napiwotzki 
### University: University of Hertfordshire
