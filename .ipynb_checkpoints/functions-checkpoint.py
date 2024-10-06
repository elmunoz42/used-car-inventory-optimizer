import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_dataframe_missing_values(dataframe, filepath):
    # Gets the mean of missing values in each column. Since True = 1 and False = 0 the mean tells us how much of the data is missing.
    missing_values = dataframe.isnull().mean().sort_values(ascending=False)
    
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    missing_plot = missing_values.plot(kind='bar')
    
    # Customize the plot
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Value Ratio per Column (Sorted)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    plt.savefig(filepath)

def calculate_percentage_difference(value1, value2):
    """
    Calculate the percentage difference between two values using NumPy.
    
    Args:
    value1 (float): First value
    value2 (float): Second value
    
    Returns:
    float: Percentage difference between the two values
    """
    return np.abs(value1 - value2) / np.mean([value1, value2]) * 100

def calculate_percentage_change(original_value, new_value):
    """
    Calculate the percentage change between two values using NumPy.
    
    Args:
    original_value (float): The original value
    new_value (float): The new value
    
    Returns:
    float: Percentage change from the original value to the new value
    """
    return (new_value - original_value) / original_value * 100

def identify_feature_types(df):
    # Identify numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Identify categorical features
    # This includes object dtype and any integer column with low cardinality
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Check for integer columns that might be categorical (e.g., ordinal data)
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].nunique() < 10:  # Adjust this threshold as needed
            categorical_features.append(col)
            numeric_features.remove(col)
    
    # Remove the target variable if it's in either list
    target_variable = 'price'  # Adjust this to your target variable name
    if target_variable in numeric_features:
        numeric_features.remove(target_variable)
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)
    
    return numeric_features, categorical_features