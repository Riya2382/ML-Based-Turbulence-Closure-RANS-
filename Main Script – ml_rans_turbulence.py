import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load synthetic RANS-like dataset
df = pd.read_csv("rans_ml_dataset.csv")
X = df.drop("mu_t", axis=1)
y = df["mu_t"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Eddy Viscosity (mu_t)')
plt.ylabel('Predicted Eddy Viscosity (mu_t)')
plt.title('Random Forest: Eddy Viscosity Prediction')
plt.grid(True)
plt.tight_layout()
plt.savefig("rans_mu_t_prediction.png")
plt.show()
