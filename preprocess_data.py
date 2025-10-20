import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=False)
    except Exception as e:
        raise FileNotFoundError(f"Error reading file: {e}")

    df.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

    return df


def transform(df):
    df['Age'] = df['Age'].fillna(np.mean(df['Age']))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    encoder = OneHotEncoder(drop='first')
    scaler = StandardScaler()

    numerical_columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_columns=['Sex', 'Embarked']

    preprocessor = ColumnTransformer([
        ('num', scaler, numerical_columns),
        ('cat', encoder, categorical_columns)
    ])

    
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    training_dataset = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    os.makedirs('processed_data', exist_ok=True)
    training_dataset.to_csv('processed_data/Train.csv')
    test_data.to_csv('processed_data/Test.csv')

    print("Train and Test data saved successfully in the 'processed_data' directory.")




def make_plots(df):
    os.makedirs('plots', exist_ok=True)

    #survival count plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=df, palette='coolwarm')
    plt.title('Survival Count')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig('plots/survival_count.png', dpi=300, bbox_inches='tight')
    plt.close()

    #survival by sex
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
    plt.title('Survival by Sex')
    plt.savefig('plots/survival_by_sex.png', dpi=300, bbox_inches='tight')
    plt.close()

    #survival by PClass
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='viridis')
    plt.title('Survival by Passenger Class')
    plt.savefig('plots/survival_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    #Age distribution by survival
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='cool')
    plt.title('Age Distribution by Survival')
    plt.savefig('plots/age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    #correlation heatmap (of numeric columns)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plots saved successfully in the 'plots' directory.")



def main(path):
    data = load_data(file_path=path)
    transform(data)
    make_plots(data)

if __name__=="__main__":
    main('Data/Titanic-Dataset.csv')

    