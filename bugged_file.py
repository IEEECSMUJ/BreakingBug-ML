# Import libraries

# 1. Data Handling
import pandas as pd
import numpy as np

# 2. Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

# 3. Data Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. Iterative Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 5. Machine Learning Models
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier as XGB
from lightgbm import LGBMClassifier as LGBM
from sklearn.naive_bayes import GaussianNB

# 6. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 7. Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# Load and explore the dataset
df = pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")

# Print the first 5 rows of the dataframe
print(df.head())

# Exploring the data type of each column
df.info()

# Checking the data shape
print(df.shape)

# Summary statistics of the 'age' column
print(df['age'].describe())

# Plot histogram of 'age' column with custom colors using seaborn
sns.histplot(df['age'], kde=True, color="#FF5733")
plt.axvline(df['age'].mean(), color='Red', label='Mean')
plt.axvline(df['age'].median(), color='Green', label='Median')
plt.axvline(df['age'].mode()[0], color='Blue', label='Mode')
plt.legend()
plt.show()

# Plot the histogram of 'age' column using Plotly
fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()

# Find the values of the 'sex' column
print(df['sex'].value_counts())

# Calculate percentage of male and female in the dataset
male_count = df['sex'].value_counts()[1]
female_count = df['sex'].value_counts()[0]

total_count = male_count + female_count

male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data: {female_percentage:.2f}%')

# Difference percentage
difference_percentage = ((male_count - female_count) / female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than females in the data.')

# Find the unique values in the 'dataset' column
print(df['dataset'].value_counts())

# Plot the count plot of 'dataset' column grouped by 'sex'
fig = px.bar(df, x='dataset', color='sex')
fig.show()

# Impute missing values using Iterative Imputer
imputer = IterativeImputer(max_iter=10, random_state=42)
df['trestbps'] = imputer.fit_transform(df[['trestbps']])

# Checking for missing values in the dataset
missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
print(f"Columns with missing data: {missing_data_cols}")

# Find categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(exclude='object').columns.tolist()

print(f'Categorical Columns: {cat_cols}')
print(f'Numerical Columns: {num_cols}')

# Define function for imputing categorical missing data
def impute_categorical_missing_data(df, col):
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df

# Define function for imputing continuous missing data
def impute_continuous_missing_data(df, col):
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[col] = imputer.fit_transform(df[[col]])
    return df

# Impute missing values
for col in missing_data_cols:
    if col in cat_cols:
        df = impute_categorical_missing_data(df, col)
    elif col in num_cols:
        df = impute_continuous_missing_data(df, col)

# Model Training and Evaluation
X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('KNeighbors Classifier', KNN()),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('XGBoost Classifier', XGB(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model: {name}, Accuracy: {accuracy:.2f}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f'Best Model: {best_model}')

