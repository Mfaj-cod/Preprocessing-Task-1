#### To Do's

1.Import the dataset and explore basic info (nulls, data types).

=> Imported the dataset and explored the dataset. The dataset had 12 columns and both numerical and categorical features.
=> The columns ['Name', 'Cabin', 'Ticket'] had no significant value in the dataset because the column ['Fare'] can alone depict the quality of Ticket and Cabin through the amount of fare. Dropped them using .drop method.

2.Handle missing values using mean/median/imputation.

=> The column ['Age'] contained 177 null values, handled them using np.mean method.
=> The column ['Embarked'] conatained only 2 null values, filled the null values using .mode() method.

3.Convert categorical features into numerical using encoding.
4.Normalize/standardize the numerical features.

Used ColumnTransfer pipeline to:-
=> Encoded the categorical features using OneHotEncoder with drop='first'.
=> Scaled the numeical features using StandardScaler.

5.Visualize outliers using boxplots and remove them.
=>Visualized the outliers using boxplots and saved the fig in plot folder.

#### Interview Questions

1.What are the different types of missing data?

=> Missing completely at random, MCAR: MCAR is a type of missign data mechanism in which the probability of a value being missing is unrelated to both the observed data and missing data. In other words, if the data is MCAR, the missing values are randomly distributed throughout the dataset, and there is no sytematic reason for why they are missing.

=> Missing at random MAR: Missing at Random (MAR) is a type of missing data mechanism in which the probability of a value being missing depends only on the observed data, but not on the missing data itself. In other words, if the data MAR, the missing values are systematically related to the observed data, but not to the missing data.

=>Missing data not at Random (MNAR): It is a type of missing data mechanism where the probability of missing values depends upon the value of the missing data itself. In other words, if the data is MNAR, the missingness is not random and is dependent on unobserved and unmeasured factors that are associated with the missing values.

2.How do you handle categorical variables?

=> We have to convert the categorical variables into numerical variables using encoding methods such as OneHotEncoder, LabelEncoder, OrdinalEncoder, etc...

3.What is the difference between normalization and standardization?

=> Normalization: Rescale the data to a fixed range, usually [0, 1].
=> Standardization: Center the data around mean = 0 and standard deviation = 1.

4.How do you detect outliers?

=> We can detect outliers using boxplots.

5.Why is preprocessing important in ML?

=> Preprocessing is a crucial step in machine learning because raw data is rarely clean or consistent.
=> It helps improve data quality by handling missing values, removing noise, and standardizing formats.
=> It also ensures all features are on a comparable scale, which is especially important for algorithms that rely on distance or gradient calculations.
=> Additionally, preprocessing converts categorical data into numerical form so that models can interpret it properly, and it removes irrelevant or redundant features that might harm model performance.
=> In short, preprocessing ensures the data is accurate, consistent, and meaningful — which leads to faster training, better model accuracy, and improved generalization on unseen data.

6.What is one-hot encoding vs label encoding?

=> One-hot encoding is used to encode categorical features which is not ordered, it's just categorized. It creates a vector of the size of the number of categories present in the feature and puts 1 where the category is present and 0 elsewhere.

=> Label encoding is used to encode categorical features which is ordered, it labels the categories a unique number.

7.How do you handle data imbalance?

=> We can handle the data imbalance using resampling methods like SMOTE().
=> Data Augmentation: We can use our domain specific knowledge to generate our synthetic data.
NOTE: Always split data before resampling to avoid data leakage.

8.Can preprocessing affect model accuracy?

=> Absolutely, Preprocessing quality of data is one of the main factors in models's predicting accuracy.


to regenerate the processed data and plots:
run: python preprocess_data.py