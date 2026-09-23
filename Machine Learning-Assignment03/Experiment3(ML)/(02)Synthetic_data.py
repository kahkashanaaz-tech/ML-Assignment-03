import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
data = {
    "Age": [22, 25, np.nan, 30, 28, 35],
    "Salary": [25000, 30000, 35000, np.nan, 45000, 50000],
    "Department": ["IT", "HR", "IT", np.nan, "Finance", "HR"],
    "Years of Experience": [1, 3, 5, 7, np.nan, 10]
}

df = pd.DataFrame(data)
print("--- Original Dataset ---")
print(df)
numeric_features = ["Age", "Salary", "Years of Experience"]
categorical_features = ["Department"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", SimpleImputer(strategy="most_frequent"), categorical_features)
    ]
)
X_processed = preprocessor.fit_transform(df)

print("\n--- Preprocessed Dataset ---")
print(X_processed)
