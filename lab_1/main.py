import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print()

df = pd.read_csv('Lab1_Data.csv')

x = df.iloc[:, 0].values.reshape(-1, 1)  # Displacement (cm)
force = df.iloc[:, 1].values  # Experimental Force (N)
force_t = df.iloc[:, 2].values  # Theoretical Force (N)

plt.figure()
plt.scatter(x, force, label="Measured Force", color='blue')
plt.plot(x, force_t, label="Theoretical Force", color='red')
plt.xlabel("Displacement (cm)")
plt.ylabel("Force (N)")
plt.title("Force vs Displacement")
plt.legend()
plt.grid()

x_train, x_test, y_train, y_test = train_test_split(x, force, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

F_pred = model.predict(x_test)

k_estimated = model.coef_[0]
intercept = model.intercept_

print(f"Estimated Spring Constant (k): {k_estimated:.2f} N/cm")
print(f"Intercept: {intercept:.2f} N")

plt.figure()
plt.scatter(x_train, y_train, label="Training Data", color='blue')
plt.scatter(x_test, y_test, label="Testing Data", color='red')
plt.plot(x, force_t, label="Theoretical Force", color='orange')
plt.plot(x_test, model.predict(x_test), label="Predicted Force", color='purple', linestyle='--')
plt.xlabel("Displacement (cm)")
plt.ylabel("Force (N)")
plt.title("Force vs Displacement")
plt.legend()
plt.grid()

print(f"R2 Score: {r2_score(y_test, F_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, F_pred)):.4f}")
plt.show()