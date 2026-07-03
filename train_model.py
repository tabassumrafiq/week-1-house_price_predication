# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("data/housing.csv")

# Display first 5 rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Select features and target
X = data[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.pkl")

print("✅ Model trained and saved successfully!")