import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Create dataset
data = {
    "Age": [20, 21, None, 23, 24],
    "Income": [25000, 30000, 28000, None, 40000],
    "City": ["Kolkata", "Durgapur", "Kolkata", "Asansol", "Durgapur"],
    "Purchased": [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Separate Features and Target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Define features
numeric_features = ["Age", "Income"]
categorical_features = ["City"]

# Numerical pipeline using MinMaxScaler
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.2,
    random_state=42
)

# Display results
print("--- Original Dataset ---")
print(df)

print("\n--- Processed Feature Matrix Shape ---")
print(X_processed.shape)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])