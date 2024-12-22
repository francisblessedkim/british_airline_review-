# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/kimbl/Downloads/britsih-airways-2/data/customer_booking.csv', encoding='ISO-8859-1')

# Step 2: Data Exploration and Feature Engineering
# For simplicity, let's preview the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check the types of columns
print(df.dtypes)

# If there are categorical columns, they need to be encoded (if not already)
# Convert categorical columns (if any) to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 3: Split the dataset into train and test sets
X = df.drop('booking_complete', axis=1)  # Features (independent variables)
y = df['booking_complete']               # Target variable (dependent variable)

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Machine Learning Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Step 6: Visualize Feature Importance
# Get feature importances from the trained model
feature_importances = model.feature_importances_

# Create a DataFrame for the feature importances
feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort by importance
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Plot only the top 10 most important features
top_n = 10
top_features_df = feature_importances_df.head(top_n)

# Plot the top 10 feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_df)
plt.title(f'Top {top_n} Feature Importance for Customer Booking Prediction')
plt.tight_layout()
plt.show()
