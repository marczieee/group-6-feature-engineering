"""
Feature Engineering Module 4: Time-Based Feature Extraction
Extracts various temporal features from datetime columns
"""

import pandas as pd
import numpy as np
from datetime import datetime


def time_based_feature_extraction(df):
    """
    Extract time-based features from datetime columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with extracted time features
    """
    df_new = df.copy()
    
    # Identify datetime columns
    datetime_cols = []
    for col in df_new.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_new[col] = pd.to_datetime(df_new[col])
                datetime_cols.append(col)
            except:
                pass
    
    print(f"\n🕐 Found {len(datetime_cols)} datetime columns: {datetime_cols}")
    
    for col in datetime_cols:
        print(f"\n📅 Extracting features from: {col}")
        
        # 1. Year
        df_new[f'{col}_year'] = df_new[col].dt.year
        print(f"   ✅ {col}_year")
        
        # 2. Month
        df_new[f'{col}_month'] = df_new[col].dt.month
        print(f"   ✅ {col}_month")
        
        # 3. Month Name
        df_new[f'{col}_month_name'] = df_new[col].dt.month_name()
        print(f"   ✅ {col}_month_name")
        
        # 4. Day of Month
        df_new[f'{col}_day'] = df_new[col].dt.day
        print(f"   ✅ {col}_day")
        
        # 5. Day of Week (0=Monday, 6=Sunday)
        df_new[f'{col}_day_of_week'] = df_new[col].dt.dayofweek
        print(f"   ✅ {col}_day_of_week")
        
        # 6. Day Name
        df_new[f'{col}_day_name'] = df_new[col].dt.day_name()
        print(f"   ✅ {col}_day_name")
        
        # 7. Quarter
        df_new[f'{col}_quarter'] = df_new[col].dt.quarter
        print(f"   ✅ {col}_quarter")
        
        # 8. Week of Year
        df_new[f'{col}_week_of_year'] = df_new[col].dt.isocalendar().week
        print(f"   ✅ {col}_week_of_year")
        
        # 9. Is Weekend (Boolean)
        df_new[f'{col}_is_weekend'] = df_new[col].dt.dayofweek.isin([5, 6]).astype(int)
        print(f"   ✅ {col}_is_weekend")
        
        # 10. Is Month Start
        df_new[f'{col}_is_month_start'] = df_new[col].dt.is_month_start.astype(int)
        print(f"   ✅ {col}_is_month_start")
        
        # 11. Is Month End
        df_new[f'{col}_is_month_end'] = df_new[col].dt.is_month_end.astype(int)
        print(f"   ✅ {col}_is_month_end")
        
        # 12. Days Since Epoch (numerical representation)
        epoch = pd.Timestamp('1970-01-01')
        df_new[f'{col}_days_since_epoch'] = (df_new[col] - epoch).dt.days
        print(f"   ✅ {col}_days_since_epoch")
        
        # 13. Season (Northern Hemisphere)
        month = df_new[col].dt.month
        df_new[f'{col}_season'] = month.apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else
            'Fall'
        )
        print(f"   ✅ {col}_season")
        
        # 14. Days Until Today (if past) or Days From Today (if future)
        today = pd.Timestamp.now()
        df_new[f'{col}_days_from_today'] = (today - df_new[col]).dt.days
        print(f"   ✅ {col}_days_from_today")
        
        # 15. Is Recent (within last 30 days)
        df_new[f'{col}_is_recent'] = (df_new[f'{col}_days_from_today'] <= 30).astype(int)
        print(f"   ✅ {col}_is_recent")
    
    return df_new


def process_csv(input_file, output_file=None):
    """
    Process CSV file and extract time-based features
    
    Args:
        input_file (str or pd.DataFrame): Path to input CSV file or DataFrame
        output_file (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print("\n" + "="*60)
    print("⏰ MODULE 4: TIME-BASED FEATURE EXTRACTION")
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
    df_processed = time_based_feature_extraction(df)
    
    print(f"\n📊 New shape: {df_processed.shape}")
    print(f"✨ Added {df_processed.shape[1] - df.shape[1]} new columns")
    
    # Save if output path provided
    if output_file:
        df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
    
    return df_processed


if __name__ == "__main__":
    # Test the module
    df = process_csv('data/processed/step3_binned_features.csv',
                     'data/processed/step4_time_features.csv')
    print("\n🔍 Sample of time-based columns:")
    time_cols = [col for col in df.columns if 'purchase_date' in col][:8]
    if time_cols:
        print(df[time_cols].head())