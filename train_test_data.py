import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_prepared_data(train_path, test_path):
    try:
        training_data = pd.read_csv(train_path, index_col=False)
        testing_data = pd.read_csv(test_path, index_col=False)
    except Exception as e:
        raise FileNotFoundError(f"Error reading file: {e}")
    
    X_train = training_data.drop('Survived', axis=1)
    X_test = testing_data.drop('Survived', axis=1)
    y_train = training_data['Survived']
    y_test = testing_data['Survived']


    return X_train, X_test, y_train, y_test

def train_test(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(),
        "RidgeClassfier": RidgeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "decision tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),

    }

    for i in range(len(list(models))):
        model_name = list(models.keys())[i]
        model = models[model_name]
        model.fit(X_train, y_train)

        # make predictions
        y_pred = model.predict(X_test)

        # metrics
        print(f"Model: {model_name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\n")

def main(train_path, test_path):
    X_train, X_test, y_train, y_test = load_prepared_data(train_path, test_path)
    train_test(X_train, X_test, y_train, y_test)

if __name__=="__main__":
    main('processed_data/Train.csv', 'processed_data/Test.csv')