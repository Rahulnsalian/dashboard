import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from catboost import CatBoostClassifier
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv', encoding='latin-1')

# Remove 'Price' column from the features
X = data.drop(columns=['Rank', 'Ingredients', 'Price'])

# Separate the target variable
y = data['Sensitive']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Specify categorical features
cat_features = list(range(0, X.shape[1]))

# Initialize and train the CatBoost classifier
clf = CatBoostClassifier(iterations=10)
clf.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=True)

# Make predictions on the test set
pred_clf = clf.predict(X_test)

# Save the trained model
pickle.dump(clf, open('model.pkl', 'wb'))

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Example prediction using the loaded model