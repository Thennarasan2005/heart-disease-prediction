import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # NEW

# Data collection
heart_dataset = pd.read_csv(r'C:\Users\thenn\OneDrive\Desktop\heart.csv')

print(heart_dataset.head())
print(heart_dataset.shape)
print(heart_dataset.isnull().sum())
print(heart_dataset.describe())
print(heart_dataset['target'].value_counts())

# Splitting the features and target
x = heart_dataset.drop(columns='target', axis=1)
y = heart_dataset['target']

# Splitting into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# Feature scaling using StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

# Accuracy evaluation
x_train_pred = model.predict(x_train_scaled)
print(f'Accuracy of training  : {accuracy_score(y_train, x_train_pred)}')

x_test_pred = model.predict(x_test_scaled)
print(f'Accuracy of testing   : {accuracy_score(y_test, x_test_pred)}')

# Predicting for a single input (first row of training data)
input_data = np.array(x_train.iloc[2]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
if prediction[0] == 0:
    print('No heart disease predicted')
else:

    print('Heart disease predicted')
