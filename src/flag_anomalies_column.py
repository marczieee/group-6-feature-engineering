"""
Feature Engineering Module 5: Flag Anomalies Column
Detects and flags outliers/anomalies in numerical columns
"""

import pandas as pd
import numpy as np
from scipy import stats


def flag_anomalies_zscore(df, column, threshold=3):
    """
    Flag anomalies using Z-score method
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for anomalies
        threshold (float): Z-score threshold (default: 3)
        
    Returns:
        pd.Series: Binary flags (1 = anomaly, 0 = normal)
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    flags = pd.Series(0, index=df.index)
    flags.loc[df[column].notna()] = (z_scores > threshold).astype(int)
    return flags


def flag_anomalies_iqr(df, column, multiplier=1.5):
    """
    Flag anomalies using IQR (Interquartile Range) method
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for anomalies
        multiplier (float): IQR multiplier (default: 1.5)
        
    Returns:
        pd.Series: Binary flags (1 = anomaly, 0 = normal)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    flags = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
    return flags


def flag_anomalies_percentile(df, column, lower=1, upper=99):
    """
    Flag anomalies using percentile method
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for anomalies
        lower (float): Lower percentile threshold
        upper (float): Upper percentile threshold
        
    Returns:
        pd.Series: Binary flags (1 = anomaly, 0 = normal)
    """
    lower_bound = df[column].quantile(lower / 100)
    upper_bound = df[column].quantile(upper / 100)
    
    flags = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
    return flags


def flag_anomalies_column(df):
    """
    Flag anomalies in multiple numerical columns using various methods
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with anomaly flag columns
    """
    df_new = df.copy()
    
    # Select numeric columns to check for anomalies
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID columns and already flagged columns
    exclude = ['customer_id', 'id']
    numeric_cols = [col for col in numeric_cols if not any(ex in col.lower() for ex in exclude)]
    numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded')]
    numeric_cols = [col for col in numeric_cols if not col.startswith('gender_')]
    numeric_cols = [col for col in numeric_cols if not col.startswith('product_category_')]
    
    print(f"\n🔍 Checking {len(numeric_cols)} numeric columns for anomalies")
    
    # Key columns to flag
    priority_cols = ['income', 'purchase_amount', 'final_price', 'age']
    cols_to_check = [col for col in priority_cols if col in numeric_cols]
    
    # Add other numeric columns (limit to first 10 for performance)
    other_cols = [col for col in numeric_cols if col not in priority_cols][:6]
    cols_to_check.extend(other_cols)
    
    print(f"📌 Flagging anomalies in: {cols_to_check}\n")
    
    anomaly_count = {}
    
    for col in cols_to_check:
        if col in df_new.columns:
            # Method 1: Z-score (for normally distributed data)
            df_new[f'{col}_anomaly_zscore'] = flag_anomalies_zscore(df_new, col, threshold=3)
            z_count = df_new[f'{col}_anomaly_zscore'].sum()
            
            # Method 2: IQR (robust to outliers)
            df_new[f'{col}_anomaly_iqr'] = flag_anomalies_iqr(df_new, col, multiplier=1.5)
            iqr_count = df_new[f'{col}_anomaly_iqr'].sum()
            
            # Method 3: Combined flag (anomaly in either method)
            df_new[f'{col}_is_anomaly'] = (
                (df_new[f'{col}_anomaly_zscore'] == 1) | 
                (df_new[f'{col}_anomaly_iqr'] == 1)
            ).astype(int)
            combined_count = df_new[f'{col}_is_anomaly'].sum()
            
            anomaly_count[col] = combined_count
            
            print(f"✅ {col}:")
            print(f"   Z-score anomalies: {z_count}")
            print(f"   IQR anomalies: {iqr_count}")
            print(f"   Combined anomalies: {combined_count}")
    
    # Create an overall anomaly score (how many columns have anomalies)
    anomaly_cols = [col for col in df_new.columns if col.endswith('_is_anomaly')]
    df_new['anomaly_score'] = df_new[anomaly_cols].sum(axis=1)
    df_new['has_any_anomaly'] = (df_new['anomaly_score'] > 0).astype(int)
    
    print(f"\n✨ Created overall anomaly indicators:")
    print(f"   - anomaly_score: Total anomalies per row")
    print(f"   - has_any_anomaly: Binary flag for any anomaly")
    
    total_with_anomalies = df_new['has_any_anomaly'].sum()
    print(f"\n📊 Total rows with at least one anomaly: {total_with_anomalies} ({total_with_anomalies/len(df_new)*100:.1f}%)")
    
    return df_new


def process_csv(input_file, output_file=None):
    """
    Process CSV file and flag anomalies
    
    Args:
        input_file (str or pd.DataFrame): Path to input CSV file or DataFrame
        output_file (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print("\n" + "="*60)
    print("🚩 MODULE 5: FLAG ANOMALIES COLUMN")
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
    df_processed = flag_anomalies_column(df)
    
    print(f"\n📊 New shape: {df_processed.shape}")
    print(f"✨ Added {df_processed.shape[1] - df.shape[1]} new columns")
    
    # Save if output path provided
    if output_file:
        df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
    
    return df_processed


if __name__ == "__main__":
    # Test the module
    df = process_csv('data/processed/step4_time_features.csv',
                     'data/processed/step5_anomaly_flags.csv')
    print("\n🔍 Sample of anomaly flags:")
    anomaly_cols = ['income', 'income_is_anomaly', 'purchase_amount', 'purchase_amount_is_anomaly', 'anomaly_score']
    available_cols = [col for col in anomaly_cols if col in df.columns]
    print(df[available_cols].head(10))