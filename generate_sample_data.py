import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'purchase_amount': np.random.uniform(10, 5000, n_samples).round(2),
    'purchase_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) 
                      for _ in range(n_samples)],
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], n_samples),
    'rating': np.random.uniform(1, 5, n_samples).round(1),
    'discount_percent': np.random.uniform(0, 50, n_samples).round(2),
    'shipping_cost': np.random.uniform(0, 50, n_samples).round(2),
}

# Add some anomalies (outliers)
anomaly_indices = np.random.choice(n_samples, 50, replace=False)
data['income'][anomaly_indices] = np.random.randint(200000, 500000, 50)
data['purchase_amount'][anomaly_indices] = np.random.uniform(8000, 15000, 50)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('input/sample_data.csv', index=False)
print("Sample data generated: input/sample_data.csv")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())