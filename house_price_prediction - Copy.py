# HOUSE PRICE PREDICTION
# Task 01 - Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Create house price dataset
data = {
    "Square_Footage": [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600],
    "Bedrooms": [2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    "Bathrooms": [1, 1, 2, 2, 2, 2, 3, 3, 3, 4],
    "Price": [2000000, 2500000, 3000000, 3500000, 4000000,
              4500000, 5000000, 5500000, 6000000, 6500000]
}

df = pd.DataFrame(data)

print("HOUSE PRICE DATASET")
print(df)


# 2. Select features and target
X = df[["Square_Footage", "Bedrooms", "Bathrooms"]]
y = df["Price"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Create Linear Regression model
model = LinearRegression()


# 5. Train the model
model.fit(X_train, y_train)


# 6. Predict house prices
y_pred = model.predict(X_test)


# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# 8. Predict a new house price
new_house = pd.DataFrame({
    "Square_Footage": [1800],
    "Bedrooms": [3],
    "Bathrooms": [2]
})

prediction = model.predict(new_house)

print("\nNEW HOUSE")
print("Square Footage: 1800")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Predicted House Price: ₹", round(prediction[0], 2))


# 9. Display graph
# 9. Actual vs Predicted graph

plt.scatter(y_test, y_pred, label="Predicted values")

# Perfect prediction reference line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    label="Perfect Prediction Line"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)

plt.show()
