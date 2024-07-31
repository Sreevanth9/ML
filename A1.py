import pandas as pd
import numpy as np

# Use forward slashes in the file path
s = "C:/Users/Nikhil/Downloads/Lab Session Data.xlsx"

# Load the correct sheet name
data = pd.read_excel(s, sheet_name='Purchase data')

# Extract relevant columns for matrix A
A = data[['Candies (#)', 'Mangoes (Kg)']].dropna()

# Convert to NumPy array
A_matrix = A.to_numpy()

# Print the matrix
print(A_matrix)