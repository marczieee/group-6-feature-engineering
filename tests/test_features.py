import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.derive_computed_columns import derive_computed_columns
from src.encode_categorical_features import encode_categorical_features
from src.bin_numeric_ranges import bin_numeric_ranges
from src.time_based_feature_extraction import time_based_feature_extraction
from src.flag_anomalies_column import flag_anomalies_column


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'customer_id':      [1, 2, 3, 4, 5],
        'age':              [22, 35, 45, 55, 70],
        'gender':           ['Male', 'Female', 'Male', 'Other', 'Female'],
        'education':        ['Bachelor', 'Master', 'PhD', 'High School', 'Bachelor'],
        'income':           [30000, 50000, 75000, 100000, 400000],
        'purchase_amount':  [100.0, 500.0, 1000.0, 2000.0, 12000.0],
        'purchase_date':    pd.to_datetime(['2024-01-15', '2024-06-20',
                                            '2024-03-10', '2023-12-01', '2024-11-25']),
        'product_category': ['Electronics', 'Clothing', 'Food', 'Books', 'Home'],
        'rating':           [4.5, 3.0, 2.0, 5.0, 1.5],
        'discount_percent': [10.0, 25.0, 5.0, 40.0, 0.0],
        'shipping_cost':    [5.0, 10.0, 15.0, 20.0, 25.0],
    })


class TestDeriveComputedColumns:
    def test_creates_total_cost(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert 'total_cost' in result.columns

    def test_total_cost_math_correct(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert result['total_cost'].iloc[0] == round(100.0 + 5.0, 2)

    def test_creates_discount_amount(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert 'discount_amount' in result.columns

    def test_creates_final_price(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert 'final_price' in result.columns

    def test_creates_income_purchase_ratio(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert 'income_purchase_ratio' in result.columns

    def test_creates_spending_power_index(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert 'spending_power_index' in result.columns

    def test_no_rows_lost(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert len(result) == len(sample_df)

    def test_adds_new_columns(self, sample_df):
        result = derive_computed_columns(sample_df)
        assert result.shape[1] > sample_df.shape[1]


class TestEncodeCategoricalFeatures:
    def test_creates_education_encoded(self, sample_df):
        result = encode_categorical_features(sample_df)
        assert 'education_encoded' in result.columns

    def test_education_ordinal_order_correct(self, sample_df):
        result = encode_categorical_features(sample_df)
        edu_map = dict(zip(result['education'], result['education_encoded']))
        assert edu_map['PhD'] > edu_map['Master']
        assert edu_map['Master'] > edu_map['Bachelor']
        assert edu_map['Bachelor'] > edu_map['High School']

    def test_creates_gender_onehot(self, sample_df):
        result = encode_categorical_features(sample_df)
        assert 'gender_Male' in result.columns
        assert 'gender_Female' in result.columns

    def test_creates_product_category_onehot(self, sample_df):
        result = encode_categorical_features(sample_df)
        assert 'product_category_Electronics' in result.columns

    def test_onehot_values_are_binary(self, sample_df):
        result = encode_categorical_features(sample_df)
        assert set(result['gender_Male'].unique()).issubset({0, 1})

    def test_no_rows_lost(self, sample_df):
        result = encode_categorical_features(sample_df)
        assert len(result) == len(sample_df)


class TestBinNumericRanges:
    def test_creates_age_group(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        assert 'age_group' in result.columns

    def test_age_group_values_valid(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        valid = ['18-25', '26-35', '36-50', '51-65', '65+']
        for val in result['age_group'].dropna():
            assert str(val) in valid

    def test_creates_income_bracket(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        assert 'income_bracket' in result.columns

    def test_creates_purchase_category(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        assert 'purchase_category' in result.columns

    def test_creates_rating_category(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        assert 'rating_category' in result.columns

    def test_creates_discount_tier(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        assert 'discount_tier' in result.columns

    def test_no_rows_lost(self, sample_df):
        result = bin_numeric_ranges(sample_df)
        assert len(result) == len(sample_df)


class TestTimeBasedFeatureExtraction:
    def test_creates_year_column(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert 'purchase_date_year' in result.columns

    def test_creates_month_column(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert 'purchase_date_month' in result.columns

    def test_creates_day_of_week(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert 'purchase_date_day_of_week' in result.columns

    def test_creates_is_weekend(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert 'purchase_date_is_weekend' in result.columns

    def test_is_weekend_binary(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert set(result['purchase_date_is_weekend'].unique()).issubset({0, 1})

    def test_creates_season(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert 'purchase_date_season' in result.columns

    def test_season_valid_values(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        valid_seasons = {'Winter', 'Spring', 'Summer', 'Fall'}
        assert set(result['purchase_date_season'].unique()).issubset(valid_seasons)

    def test_creates_quarter(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert 'purchase_date_quarter' in result.columns

    def test_year_values_reasonable(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert all(result['purchase_date_year'].between(2020, 2030))

    def test_no_rows_lost(self, sample_df):
        result = time_based_feature_extraction(sample_df)
        assert len(result) == len(sample_df)


class TestFlagAnomaliesColumn:
    def test_creates_has_any_anomaly(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert 'has_any_anomaly' in result.columns

    def test_creates_anomaly_score(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert 'anomaly_score' in result.columns

    def test_has_any_anomaly_is_binary(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert set(result['has_any_anomaly'].unique()).issubset({0, 1})

    def test_high_income_flagged(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert result['income_is_anomaly'].iloc[4] == 1

    def test_high_purchase_flagged(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert result['purchase_amount_is_anomaly'].iloc[4] == 1

    def test_anomaly_score_non_negative(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert (result['anomaly_score'] >= 0).all()

    def test_no_rows_lost(self, sample_df):
        result = flag_anomalies_column(sample_df)
        assert len(result) == len(sample_df)


class TestFullPipeline:
    def test_pipeline_increases_columns(self, sample_df):
        df = derive_computed_columns(sample_df)
        df = encode_categorical_features(df)
        df = bin_numeric_ranges(df)
        df = time_based_feature_extraction(df)
        df = flag_anomalies_column(df)
        assert df.shape[1] > sample_df.shape[1]

    def test_pipeline_preserves_rows(self, sample_df):
        df = derive_computed_columns(sample_df)
        df = encode_categorical_features(df)
        df = bin_numeric_ranges(df)
        df = time_based_feature_extraction(df)
        df = flag_anomalies_column(df)
        assert len(df) == len(sample_df)

    def test_pipeline_output_has_no_extra_nulls(self, sample_df):
        df = derive_computed_columns(sample_df)
        df = encode_categorical_features(df)
        assert df['customer_id'].isnull().sum() == 0

    def test_input_csv_detected(self):
        os.makedirs("input", exist_ok=True)
        tiny_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        tiny_df.to_csv("input/test_detect.csv", index=False)
        files = [f for f in os.listdir("input") if f.endswith('.csv')]
        assert len(files) >= 1
        os.remove("input/test_detect.csv")