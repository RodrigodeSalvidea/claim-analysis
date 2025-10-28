import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt



data = r.engineered_data

X = data.drop('charges', axis = 1)
Y = data["charges"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 13)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

models = {
  "LinearRidge": Ridge(alpha = 10.0),
  "RandomForest": RandomForestRegressor(
    max_depth=6,
    min_samples_leaf = 10,
    random_state = 13
  ),
  "XGBoost": xgb.XGBRegressor(
    n_estimators = 100,
    max_depth = 4,
    learning_rate = .05,
    min_child_weight = 5,
    random_state = 13
  )
}
for name, model in models.items():
  if name == "LinearRidge":
    X_tr, X_te = X_train_scaled, X_test_scaled
  else:
    X_tr, X_te = X_train, X_test
    
  cross_validation_scores = cross_val_score(
    model, 
    X_tr, 
    Y_train,
    cv = 5,
    scoring = 'neg_mean_absolute_error'
  )
  model.fit(X_tr, Y_train)
  Y_pred = model.predict(X_te)
    
  mse = mean_squared_error(Y_test, Y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(Y_test, Y_pred)
  r2 = r2_score(Y_test, Y_pred)
  results = {
        'CV MAE': -cross_validation_scores.mean(),
        'CV Std': cross_validation_scores.std(),
        'Test RMSE': rmse,
        'Test MAE': mae,
        'Test R^2': r2
  }
  print(f"\n{name}:")
  print(f"  CV MAE: ${-cross_validation_scores.mean():,.2f} (±${cross_validation_scores.std():,.2f})")
  print(f"  Test RMSE: ${rmse:,.2f}")
  print(f"  Test MAE: ${mae:,.2f}")
  print(f"  Test R²: {r2:.3f}")
  
  
  #----
linear_model = models["LinearRidge"]
random_forest_model = models["RandomForest"]
xgboost_model = models["XGBoost"]
linear_model.fit(X_train, Y_train)
Y_pred = linear_model.predict(X_test)
  
  
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], 
         [Y_test.min(), Y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Predicted vs Actual Charges -- Linear Model')
plt.savefig('predicted-vs-actual-changes-lin.png')

coefficients = linear_model.coef_
intercept = linear_model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

plt.close()
random_forest_model.fit(X_train, Y_train)
Y_pred = random_forest_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], 
         [Y_test.min(), Y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Predicted vs Actual Charges -- Random Forest Model')
plt.savefig('predicted-vs-actual-changes-random-forest.png')

importances = random_forest_model.feature_importances_
print(f"random forest importances: {importances}")

plt.close()
xgboost_model.fit(X_train, Y_train)
Y_pred = xgboost_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], 
         [Y_test.min(), Y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Predicted vs Actual Charges --XGBoost Model')
plt.savefig('predicted-vs-actual-changes-xgboost.png')

importances = xgboost_model.feature_importances_
print(f"XGBoost: {importances}")

    
    
