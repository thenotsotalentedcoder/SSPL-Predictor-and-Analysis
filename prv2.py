import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('AirfoilSelfNoise.csv')

# Separate features and target
X = df.drop('SSPL', axis=1)
y = df['SSPL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a bagging regressor with Random Forest as the estimator
rf = RandomForestRegressor(n_estimators=100, random_state=42)
bagging_rf = BaggingRegressor(estimator=rf, n_estimators=10, random_state=42)

# Fit the model
bagging_rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = bagging_rf.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Perform cross-validation
cv_scores = cross_val_score(bagging_rf, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print(f"Cross-validation RMSE scores: {cv_rmse}")
print(f"Average CV RMSE: {np.mean(cv_rmse)}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Test Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual SSPL')
plt.ylabel('Predicted SSPL')
plt.title('Actual vs Predicted SSPL')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Plot feature importances
feature_importance = bagging_rf.estimators_[0].feature_importances_
feature_names = X.columns
plt.figure(figsize=(12, 8))
bars = plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(12, 8))
plt.scatter(y_pred, residuals, alpha=0.5, color='green', label='Residuals')
plt.xlabel('Predicted SSPL')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted SSPL')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual Line')
plt.legend()
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png')
plt.close()

# Plot error distribution
plt.figure(figsize=(12, 8))
plt.hist(residuals, bins=30, color='purple', alpha=0.7, label='Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.legend()
plt.tight_layout()
plt.savefig('residuals_distribution.png')
plt.close()

# Generate new data for predictions
new_data = X.sample(n=100, random_state=42)
new_data_scaled = scaler.transform(new_data)
new_predictions = bagging_rf.predict(new_data_scaled)

# Plot actual data vs new predictions
plt.figure(figsize=(12, 8))
plt.scatter(y, X['f'], alpha=0.5, color='blue', label='Actual Data')
plt.scatter(new_predictions, new_data['f'], alpha=0.5, color='red', label='New Predictions')
plt.xlabel('SSPL')
plt.ylabel('Frequency (Hz)')
plt.title('Actual Data vs New Predictions')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_new_predictions.png')
plt.close()

print("All plots have been saved as PNG files.")