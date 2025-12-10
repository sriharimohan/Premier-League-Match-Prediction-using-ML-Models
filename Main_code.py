import pandas as pd
data = pd.read_csv("England.csv")

data.head()
data.info()

required_cols = ['FT Result', 'H Shots', 'A Shots', 'H SOT', 'A SOT', 'H Fouls', 'A Fouls', 'H Corners', 'A Corners', 'H Yellow', 'A Yellow', 'H Red', 'A Red']

df_cleaned = df.dropna(subset=required_cols).copy()

df_cleaned['Result'] = df_cleaned['FT Result'].map({'A': 0, 'D': 1, 'H': 2})


import numpy as np

df_cleaned['ShotsOnTargetDiff'] = df_cleaned['H SOT'] - df_cleaned['A SOT']
df_cleaned['ShotsDiff'] = df_cleaned['H Shots'] - df_cleaned['A Shots']
df_cleaned['CornersDiff'] = df_cleaned['H Corners'] - df_cleaned['A Corners']
df_cleaned['FoulsDiff'] = df_cleaned['H Fouls'] - df_cleaned['A Fouls']
df_cleaned['YellowCardDiff'] = df_cleaned['H Yellow'] - df_cleaned['A Yellow']
df_cleaned['RedCardDiff'] = df_cleaned['H Red'] - df_cleaned['A Red']

features = ['ShotsOnTargetDiff', 'ShotsDiff', 'CornersDiff', 
            'FoulsDiff', 'YellowCardDiff', 'RedCardDiff']
X = df_cleaned[features]
y = df_cleaned['Result']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
corr_matrix = df_cleaned[features + ['Result']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix: Feature Relationships")
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

models_to_test = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

for name, model in models_to_test.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"{name} Baseline Accuracy: {acc:.4f}")


from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  
    'solver': ['lbfgs']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), 
                           param_grid, cv=5, n_jobs=-1, verbose=0)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_


from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import shap

y_pred_final = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_final, target_names=['Away', 'Draw', 'Home']))

ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test, 
                                      display_labels=['Away', 'Draw', 'Home'], cmap='Blues')
plt.title("Plot 2: Confusion Matrix (Optimised Model)")
plt.show()

importance = best_model.coef_[2] 
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Plot 3: Feature Coefficients (Drivers of Home Wins)")
plt.axvline(x=0, color='black', linestyle='--')
plt.show()

probs = best_model.predict_proba(X_test_scaled)
confidence_scores = probs.max(axis=1)

plt.figure(figsize=(8, 5))
plt.hist(confidence_scores, bins=20, color='purple', edgecolor='black', alpha=0.7)
plt.title("Plot 4: Model Confidence Distribution")
plt.xlabel("Confidence Score")
plt.axvline(x=0.5, color='red', linestyle='--')
plt.show()

masker = shap.maskers.Independent(data=X_test_scaled)
explainer = shap.LinearExplainer(best_model, masker=masker)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled, feature_names=features, plot_type="bar")

results_table = df_cleaned.loc[X_test.index].copy()
results_table['Predicted_Code'] = y_pred_final
outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
results_table['Prediction'] = results_table['Predicted_Code'].map(outcome_map)
results_table['Actual'] = results_table['Result'].map(outcome_map)
results_table['Correct?'] = results_table['Prediction'] == results_table['Actual']
final_view = results_table[['Date', 'HomeTeam', 'AwayTeam', 'H Shots', 'A Shots', 'Prediction', 'Actual', 'Correct?']]
print(final_view.head(10))

arsenal_games = final_view[(final_view['HomeTeam'] == 'Arsenal') | (final_view['AwayTeam'] == 'Arsenal')]
print(arsenal_games.head())

arsenal_accuracy = arsenal_games['Correct?'].mean()
print(f"\nModel Accuracy for Arsenal games: {arsenal_accuracy:.2%}")



