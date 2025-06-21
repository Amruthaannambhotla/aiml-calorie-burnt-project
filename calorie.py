import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')
df = exercise.merge(calories, on='User_ID')
df
df.isnull()
df.isnull().sum()
#checking the data is clear or not
sns.heatmap(df.isnull())
###GENDER DISTRIBUTION
df['Gender']=df['Gender'].map({0:'Female',1:"Male"}).astype('str')
plt.figure(figsize=(8, 6))
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightpink', 'lightblue'])
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()
#### AGE DISTRIBUTION
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
###HEIGHT DISTRIBUTION BY GENDER
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Height', data=df)
plt.title('Height Distribution by Gender')
plt.show()
#### WEIGHT DISTRIBUTION BY GENDER
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Weight', data=df)
plt.title('Weight Distribution by Gender')
plt.show()
###CALORIES BURNED VS. DURATION
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Duration', y='Calories', hue='Gender', data=df)
plt.title('Calories Burned vs. Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories Burned')
plt.show()
###HEART RATE VS. BODY TEMPERATURE
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Body_Temp', y='Heart_Rate', hue='Gender', data=df)
plt.title('Heart Rate vs. Body Temperature')
plt.xlabel('Body Temperature (°C)')
plt.ylabel('Heart Rate (bpm)')
plt.show()
### CALORIES BURNED BY GENDER
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Calories', data=df)
plt.title('Calories Burned by Gender')
plt.show()
# Encoding
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Train test split
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

models = {
    'lr': LinearRegression(),
    'rd': Ridge(),
    'ls': Lasso(),
    'dtr': DecisionTreeRegressor(),
    'rfr': RandomForestRegressor()
}

for name, mod in models.items():
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)

    print(f"{name}  MSE: {mean_squared_error(y_test, y_pred)}, Score: {r2_score(y_test, y_pred)*100}")

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

import pickle
pickle.dump(rfr, open('rfr.pkl', 'wb'))
df.to_csv('X_train.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




# Drop User_ID column since it is not needed for prediction
data = df.drop(columns=['User_ID'])

# Split data into features and target variable
X = data.drop(columns=['Calories'])
y = data['Calories']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature columns (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(16, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on test data: {mae}")

# Make predictions
predictions = model.predict(X_test)

# Calculate R-squared score
r2 = r2_score(y_test, predictions)
print(f"R-squared score on test data: {r2}")

# If you really need an 'accuracy' like metric for demonstration
# We can compute a percentage accuracy based on the predictions
accuracy = 1 - (mae / y_test.mean())
print(f"Accuracy (based on MAE): {accuracy * 100:.2f}%")
