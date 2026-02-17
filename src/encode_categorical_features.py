"""
Feature Engineering Module 2: Encode Categorical Features
Converts categorical variables into numerical representations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical_features(df):
    """
    Encode categorical columns into numerical format
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    df_new = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_new.select_dtypes(include=['object']).columns.tolist()
    
    # Exclude datetime columns if any
    datetime_cols = []
    for col in categorical_cols:
        try:
            pd.to_datetime(df_new[col])
            datetime_cols.append(col)
        except:
            pass
    
    categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
    
    print(f"\n🔍 Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # 1. Label Encoding for ordinal features
    ordinal_features = ['education']
    
    for col in ordinal_features:
        if col in df_new.columns:
            # Define order
            if col == 'education':
                order = ['High School', 'Bachelor', 'Master', 'PhD']
                df_new[f'{col}_encoded'] = df_new[col].map({v: i for i, v in enumerate(order)})
                print(f"✅ Label encoded: {col} → {col}_encoded")
    
    # 2. One-Hot Encoding for nominal features
    nominal_features = ['gender', 'product_category']
    
    for col in nominal_features:
        if col in df_new.columns:
            # Get unique values
            unique_vals = df_new[col].unique()
            print(f"✅ One-hot encoding: {col} ({len(unique_vals)} categories)")
            
            # Create dummy variables
            dummies = pd.get_dummies(df_new[col], prefix=col, drop_first=False)
            df_new = pd.concat([df_new, dummies], axis=1)
            
            print(f"   Created columns: {', '.join(dummies.columns.tolist())}")
    
    # 3. Frequency Encoding (for high-cardinality features)
    # This encodes based on how frequent each category appears
    for col in categorical_cols:
        if col not in ordinal_features + nominal_features:
            if col in df_new.columns:
                freq = df_new[col].value_counts(normalize=True)
                df_new[f'{col}_freq'] = df_new[col].map(freq).round(4)
                print(f"✅ Frequency encoded: {col} → {col}_freq")
    
    return df_new


def process_csv(input_file, output_file=None):
    """
    Process CSV file and encode categorical features
    
    Args:
        input_file (str or pd.DataFrame): Path to input CSV file or DataFrame
        output_file (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print("\n" + "="*60)
    print("🏷️  MODULE 2: ENCODE CATEGORICAL FEATURES")
    print("="*60)
    
    # Load data
    if isinstance(input_file, str):
        df = pd.read_csv(input_file)
        print(f"\n📂 Loaded: {input_file}")
    else:
        df = input_file.copy()
        print(f"\n📂 Loaded DataFrame from previous step")
    
    print(f"📊 Original shape: {df.shape}")
    
    # Apply feature engineering
    df_processed = encode_categorical_features(df)
    
    print(f"\n📊 New shape: {df_processed.shape}")
    print(f"✨ Added {df_processed.shape[1] - df.shape[1]} new columns")
    
    # Save if output path provided
    if output_file:
        df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
    
    return df_processed


if __name__ == "__main__":
    # Test the module
    df = process_csv('data/processed/step1_computed_columns.csv',
                     'data/processed/step2_encoded_features.csv')
    print("\n🔍 Sample of encoded columns:")
    encoded_cols = [col for col in df.columns if '_encoded' in col or 'gender_' in col or 'product_category_' in col]
    print(df[encoded_cols[:5]].head())