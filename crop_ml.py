"""
    Feature Engineering:
        Interaction Terms: Create interaction terms between features. For example, temperature * rainfall could represent a combined effect on crop growth.
        Polynomial Features: Introduce polynomial features to capture nonlinear relationships.
        Feature Scaling: Scale features, especially for models sensitive to feature magnitudes like SVM.

    Model Selection and Tuning:
        Hyperparameter Tuning: Tune hyperparameters of models using techniques like Grid Search or Random Search.
        Model Ensemble: Combine predictions from multiple models (like Decision Trees, SVM, Random Forest) through techniques like Voting Classifier or Stacking.

    Cross-Validation:
        Use cross-validation for more reliable estimation of the model's performance. It helps in identifying overfitting and selecting models that generalize well.

    Evaluation Metrics:
        Besides accuracy, consider other evaluation metrics like precision, recall, F1-score, or ROC-AUC score, especially if the classes are imbalanced.

    Handling Imbalanced Data:
        If there's class imbalance in the target variable, use techniques like oversampling, undersampling, or Synthetic Minority Over-sampling Technique (SMOTE).

    Regularization:
        Apply regularization techniques like L1 or L2 regularization in models like Logistic Regression to prevent overfitting.

"""
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset
df = pd.read_csv("/home/0221csds213/webapp/Hack-X/crop_recommendation.csv")

# Define features and target
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Feature Engineering: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model Selection and Hyperparameter Tuning
models = [
    ('Decision Tree', DecisionTreeClassifier(), {'max_depth': [3, 5, 7]}),
    ('SVM', SVC(probability=True), {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.1, 1, 10]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [10, 20, 30]})
]

best_estimators = {}

for name, model, params in models:
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train_poly if name == 'Decision Tree' else X_train, Y_train)
    best_estimators[name] = grid_search.best_estimator_
    predicted_probabilities = grid_search.predict_proba(X_test_poly if name == 'Decision Tree' else X_test)
    top_3_crops_indices = np.argsort(predicted_probabilities, axis=1)[:, -3:]  # Indices of top 3 crops
    top_3_crops = [grid_search.classes_[idx] for idx in top_3_crops_indices]
    #print(f"{name} Top 3 Recommended Crops: {top_3_crops}")
    
    # Cross-validation score
    cv_score = cross_val_score(model, features, target, cv=5)
    #print(f"{name} Cross-validation Score: {np.mean(cv_score)}")

# Model Ensemble: Voting Classifier
voting_clf = VotingClassifier(estimators=[(name, model) for name, model in best_estimators.items()], voting='soft')
voting_clf.fit(X_train_poly, Y_train)
predicted_probabilities = voting_clf.predict_proba(X_test_poly)
top_3_crops_indices = np.argsort(predicted_probabilities, axis=1)[:, -3:]  # Indices of top 3 crops
top_3_crops = [voting_clf.classes_[idx] for idx in top_3_crops_indices]
#print(f"Voting Classifier Top 3 Recommended Crops: {top_3_crops}")

# Cross-validation score for Voting Classifier
cv_score_voting = cross_val_score(voting_clf, features, target, cv=5)
#print(f"Voting Classifier Cross-validation Score: {np.mean(cv_score_voting)}")

# Function to predict top 3 crops
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_probabilities = voting_clf.predict_proba(poly.transform(data))
    top_3_crops_indices = np.argsort(predicted_probabilities, axis=1)[:, -3:]  # Indices of top 3 crops
    top_3_crops = [voting_clf.classes_[idx] for idx in top_3_crops_indices]
    return top_3_crops

