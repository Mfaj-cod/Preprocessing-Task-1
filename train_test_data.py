import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

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

def train_test_write(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(),
        "RidgeClassfier": RidgeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "decision tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
    }

    pdf_path = 'reports/report.pdf'

    # Move PdfPages context manager OUTSIDE the loop to open the file once and write all pages to it.
    with PdfPages(pdf_path) as pdf: 
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # metrics
            aac = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # figure for the current model's report
            fig = plt.figure(figsize=(8.27, 11.69)) # A4 size
            
            # Prepare the text for the report
            txt = f"Model: {model_name}\n\n"
            txt += f"Accuracy: {aac:.4f}\n\n"
            txt += "Classification Report:\n"
            txt += str(cr) + "\n\n"
            txt += "Confusion Matrix:\n"
            txt += str(cm)
            
            # Display the text on the figure (you'd typically use ax.text for this)
            # A simple way to display this much text might be to use plt.text
            plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', family='monospace')
            plt.title(f"Model Performance Report: {model_name}")
            
            pdf.savefig(fig) 
            plt.close(fig)


def main(train_path, test_path):
    X_train, X_test, y_train, y_test = load_prepared_data(train_path, test_path)
    train_test_write(X_train, X_test, y_train, y_test)

if __name__=="__main__":
    main('processed_data/Train.csv', 'processed_data/Test.csv')