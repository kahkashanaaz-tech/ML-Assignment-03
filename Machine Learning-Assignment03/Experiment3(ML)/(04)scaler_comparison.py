import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = np.array([
    [20, 25000],
    [21, 30000],
    [22, 28000],
    [23, 35000],
    [24, 40000]
])
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(data)
minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(data)
print("Original Data:")
print(data)
print("\n--- StandardScaler ---")
print(standard_data)
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())
print("\n--- MinMaxScaler ---")
print(minmax_data)
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())
