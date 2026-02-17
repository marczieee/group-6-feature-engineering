"""
Feature Engineering Module 1: Derive Computed Columns
Creates new columns based on mathematical operations on existing columns
"""

import pandas as pd
import numpy as np
from datetime import datetime


def derive_computed_columns(df):
    """
    Derive new columns through computations on existing columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new computed columns
    """
    df_new = df.copy()
    
    # 1. Calculate total cost (purchase + shipping)
    if 'purchase_amount' in df_new.columns and 'shipping_cost' in df_new.columns:
        df_new['total_cost'] = df_new['purchase_amount'] + df_new['shipping_cost']
        print("✅ Created: total_cost")
    
    # 2. Calculate discount amount
    if 'purchase_amount' in df_new.columns and 'discount_percent' in df_new.columns:
        df_new['discount_amount'] = (df_new['purchase_amount'] * df_new['discount_percent'] / 100).round(2)
        print("✅ Created: discount_amount")
    
    # 3. Calculate final price after discount
    if 'total_cost' in df_new.columns and 'discount_amount' in df_new.columns:
        df_new['final_price'] = (df_new['total_cost'] - df_new['discount_amount']).round(2)
        print("✅ Created: final_price")
    
    # 4. Calculate price per rating point
    if 'final_price' in df_new.columns and 'rating' in df_new.columns:
        df_new['price_per_rating'] = (df_new['final_price'] / df_new['rating']).round(2)
        print("✅ Created: price_per_rating")
    
    # 5. Calculate income to purchase ratio
    if 'income' in df_new.columns and 'purchase_amount' in df_new.columns:
        df_new['income_purchase_ratio'] = (df_new['purchase_amount'] / df_new['income'] * 100).round(2)
        print("✅ Created: income_purchase_ratio")
    
    # 6. Calculate age groups
    if 'age' in df_new.columns:
        df_new['age_squared'] = df_new['age'] ** 2
        print("✅ Created: age_squared")
    
    # 7. Calculate spending power index (normalized)
    if 'income' in df_new.columns and 'age' in df_new.columns:
        df_new['spending_power_index'] = ((df_new['income'] / 1000) / df_new['age']).round(2)
        print("✅ Created: spending_power_index")
    
    return df_new


def process_csv(input_file, output_file=None):
    """
    Process CSV file and derive computed columns
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print("\n" + "="*60)
    print("🔧 MODULE 1: DERIVE COMPUTED COLUMNS")
    print("="*60)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"\n📂 Loaded: {input_file}")
    print(f"📊 Original shape: {df.shape}")
    
    # Apply feature engineering
    df_processed = derive_computed_columns(df)
    
    print(f"\n📊 New shape: {df_processed.shape}")
    print(f"✨ Added {df_processed.shape[1] - df.shape[1]} new columns")
    
    # Save if output path provided
    if output_file:
        df_processed.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
    
    return df_processed


if __name__ == "__main__":
    # Test the module
    df = process_csv('data/raw/sample_data.csv', 
                     'data/processed/step1_computed_columns.csv')
    print("\n🔍 Sample of new columns:")
    print(df[['purchase_amount', 'shipping_cost', 'total_cost', 
              'discount_amount', 'final_price']].head())