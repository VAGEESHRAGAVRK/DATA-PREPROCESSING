============================================================
                    DATA PRE-PROCESSING
============================================================

AIM
---

To perform data preprocessing on a dataset using Python and
Scikit-learn by handling missing values, encoding categorical data,
splitting the dataset, and applying feature scaling.


PROCEDURE
---------

1. Import the required Python libraries.

2. Mount Google Drive and load the dataset using Pandas.

3. Display the first few records of the dataset.

4. Inspect the dataset using df.info() and df.shape.

5. Separate the independent variables (X) and dependent variable (Y).

6. Convert the independent variables into an array.

7. Identify and handle missing values using SimpleImputer with the
   mean strategy.

8. Encode the categorical Country column using LabelEncoder.

9. Apply One-Hot Encoding to convert categorical country values into
   dummy variables.

10. Encode the dependent variable Purchased using LabelEncoder.

11. Split the dataset into training and testing sets using
    train_test_split.

12. Apply StandardScaler for feature scaling.

13. Display the preprocessed training and testing datasets.


PROGRAM
-------

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------------------

# If using Google Colab, uncomment the following lines:

# from google.colab import drive
# drive.mount('/content/drive')

# Example dataset path:
# dataset = pd.read_csv('/content/drive/MyDrive/Data.csv')

# For local execution:
dataset = pd.read_csv("Data.csv")


# ------------------------------------------------------------
# STEP 2: DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 60)
print("              DATA PREPROCESSING")
print("=" * 60)

print("\nFIRST FIVE RECORDS:")
print(dataset.head())


# ------------------------------------------------------------
# STEP 3: INSPECT DATASET
# ------------------------------------------------------------

print("\nDATASET INFORMATION:")
dataset.info()

print("\nDATASET SHAPE:")
print(dataset.shape)


# ------------------------------------------------------------
# STEP 4: SEPARATE X AND Y
# ------------------------------------------------------------

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


print("\nINDEPENDENT VARIABLES (X):")
print(X)

print("\nDEPENDENT VARIABLE (Y):")
print(Y)


# ------------------------------------------------------------
# STEP 5: HANDLE MISSING VALUES
# ------------------------------------------------------------

# Mean strategy is applied to numerical columns.
# The example assumes missing values are represented as NaN.

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean"
)

# Apply imputation to numerical columns
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])


print("\nDATA AFTER HANDLING MISSING VALUES:")
print(X)


# ------------------------------------------------------------
# STEP 6: LABEL ENCODING
# ------------------------------------------------------------

label_encoder_country = LabelEncoder()

X[:, 0] = label_encoder_country.fit_transform(X[:, 0])


print("\nCOUNTRY AFTER LABEL ENCODING:")
print(X)


# ------------------------------------------------------------
# STEP 7: ONE-HOT ENCODING
# ------------------------------------------------------------

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

country_encoded = one_hot_encoder.fit_transform(
    X[:, 0].reshape(-1, 1)
)


# Convert numerical columns into float
numerical_data = X[:, 1:].astype(float)


# Combine encoded country and numerical columns
X = np.concatenate(
    (country_encoded, numerical_data),
    axis=1
)


print("\nINDEPENDENT VARIABLES AFTER ONE-HOT ENCODING:")
print(X)


# ------------------------------------------------------------
# STEP 8: LABEL ENCODING OF DEPENDENT VARIABLE
# ------------------------------------------------------------

label_encoder_y = LabelEncoder()

Y = label_encoder_y.fit_transform(Y)


print("\nDEPENDENT VARIABLE AFTER LABEL ENCODING:")
print(Y)


# ------------------------------------------------------------
# STEP 9: TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


print("\nTRAINING DATA:")
print(X_train)

print("\nTESTING DATA:")
print(X_test)

print("\nTRAINING TARGET:")
print(Y_train)

print("\nTESTING TARGET:")
print(Y_test)


# ------------------------------------------------------------
# STEP 10: FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# STEP 11: DISPLAY FINAL DATA
# ------------------------------------------------------------

print("\nSCALED TRAINING DATA:")
print(X_train)

print("\nSCALED TESTING DATA:")
print(X_test)


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)


RESULT
------

The given dataset was successfully preprocessed using Python and
Scikit-learn. Missing values were handled using SimpleImputer,
categorical data was encoded using LabelEncoder and One-Hot Encoding,
and the dataset was divided into training and testing sets.

Feature scaling was successfully performed using StandardScaler.
The preprocessed training and testing datasets were obtained and
prepared for further machine learning applications.


CONCLUSION
----------

Data preprocessing is an important step in machine learning. The
dataset was successfully cleaned, encoded, split, and scaled using
Scikit-learn preprocessing techniques. The resulting dataset is now
suitable for applying machine learning algorithms.


REQUIREMENTS
------------

Python 3.x
Pandas
NumPy
Scikit-learn


INSTALLATION
------------

Install the required libraries using:

pip install pandas numpy scikit-learn


DATASET FORMAT
--------------

The CSV file should contain columns similar to:

Country
Age
Salary
Purchased

Example:

Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
Spain,38,61000,No
Germany,40,63777,Yes
France,35,58000,Yes
Spain,48,79000,No
France,50,83000,Yes
Germany,37,67000,Yes
France,45,56000,No


HOW TO RUN
----------

1. Install Python 3.x.

2. Install the required libraries.

3. Place the dataset file named Data.csv in the same folder as
   the Python program.

4. Save the program as:

   data_preprocessing.py

5. Run the program using:

   python data_preprocessing.py


============================================================
                         END
============================================================
