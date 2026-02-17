"""
Main Pipeline: Feature Engineering - Group 6
Runs all 5 feature engineering modules in sequence
"""

import pandas as pd
import sys
from datetime import datetime

# Import all modules
from src.derive_computed_columns import process_csv as derive_columns
from src.encode_categorical_features import process_csv as encode_features
from src.bin_numeric_ranges import process_csv as bin_features
from src.time_based_feature_extraction import process_csv as extract_time_features
from src.flag_anomalies_column import process_csv as flag_anomalies


def print_header():
    """Print pipeline header"""
    print("\n" + "="*70)
    print("🚀 FEATURE ENGINEERING PIPELINE - GROUP 6")
    print("="*70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_summary(original_df, final_df, start_time):
    """Print pipeline summary"""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Original columns: {original_df.shape[1]}")
    print(f"   Final columns: {final_df.shape[1]}")
    print(f"   New features created: {final_df.shape[1] - original_df.shape[1]}")
    print(f"   Total rows: {final_df.shape[0]}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"\n⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def run_pipeline(input_csv='data/raw/sample_data.csv', 
                 output_csv='data/processed/final_engineered_features.csv'):
    """
    Run the complete feature engineering pipeline
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to save final output
        
    Returns:
        pd.DataFrame: Final processed dataframe
    """
    start_time = datetime.now()
    print_header()
    
    try:
        # Load original data for comparison
        original_df = pd.read_csv(input_csv)
        print(f"📂 Input file: {input_csv}")
        print(f"📊 Original shape: {original_df.shape}\n")
        
        # Step 1: Derive Computed Columns
        print("STEP 1/5: Running derive_computed_columns...")
        df = derive_columns(input_csv)
        
        # Step 2: Encode Categorical Features
        print("\nSTEP 2/5: Running encode_categorical_features...")
        df = encode_features(df)
        
        # Step 3: Bin Numeric Ranges
        print("\nSTEP 3/5: Running bin_numeric_ranges...")
        df = bin_features(df)
        
        # Step 4: Extract Time-Based Features
        print("\nSTEP 4/5: Running time_based_feature_extraction...")
        df = extract_time_features(df)
        
        # Step 5: Flag Anomalies
        print("\nSTEP 5/5: Running flag_anomalies_column...")
        df = flag_anomalies(df)
        
        # Save final output
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Final output saved: {output_csv}")
        
        # Print summary
        print_summary(original_df, df, start_time)
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def show_sample_output(df, n_rows=5):
    """Display sample of the output"""
    print(f"\n🔍 Sample Output (first {n_rows} rows):")
    print("="*70)
    
    # Show a few key columns
    key_columns = [
        'customer_id', 'age', 'age_group', 'income', 'income_bracket',
        'purchase_amount', 'final_price', 'rating_category',
        'education_encoded', 'has_any_anomaly'
    ]
    
    # Filter to existing columns
    available_cols = [col for col in key_columns if col in df.columns]
    
    print(df[available_cols].head(n_rows).to_string())
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run the pipeline
    final_df = run_pipeline()
    
    # Show sample output
    show_sample_output(final_df)
    
    print("\n✨ All feature engineering steps completed!")
    print("📁 Check the 'data/processed/' folder for all intermediate and final outputs.")