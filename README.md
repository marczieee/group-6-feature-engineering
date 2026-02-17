# Feature Engineering Pipeline - Group 6

## 📋 Project Overview

This project implements a comprehensive **Feature Engineering Pipeline** that transforms raw CSV data into enriched datasets with engineered features. The pipeline consists of 5 core modules that create new columns through various data transformation techniques.

## 👥 Team Members - Group 6

- [Gelera, Marc Eldrian L]
- [Aganan, Akira Yuki]
- [Bolina, Renz Andrae]
- [Pacheco, Haroliyen]


## 🎯 Features

### Module 1: Derive Computed Columns
Creates new features through mathematical operations on existing columns:
- Total cost calculations
- Discount amounts
- Price ratios
- Spending power indices

### Module 2: Encode Categorical Features
Converts categorical variables into numerical representations:
- **Label Encoding**: For ordinal features (education levels)
- **One-Hot Encoding**: For nominal features (gender, product categories)
- **Frequency Encoding**: For high-cardinality features

### Module 3: Bin Numeric Ranges
Groups continuous values into discrete categories:
- Age groups (18-25, 26-35, etc.)
- Income brackets (Low, Middle, High, etc.)
- Purchase categories
- Rating tiers

### Module 4: Time-Based Feature Extraction
Extracts temporal features from datetime columns:
- Year, month, day, quarter
- Day of week, weekend flags
- Season extraction
- Days since/until dates

### Module 5: Flag Anomalies
Detects and flags outliers using statistical methods:
- **Z-score method**: For normally distributed data
- **IQR method**: Robust to outliers
- **Combined flags**: Comprehensive anomaly detection

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/feature-engineering.git
cd feature-engineering
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate sample data** (optional)
```bash
python generate_sample_data.py
```

4. **Run the pipeline**
```bash
python main.py
```

## 📁 Project Structure
```
feature-engineering/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── main.py                           # Main pipeline orchestrator
├── generate_sample_data.py           # Sample data generator
│
├── data/
│   ├── raw/                          # Input CSV files
│   │   └── sample_data.csv
│   └── processed/                    # Output files from each step
│       ├── step1_computed_columns.csv
│       ├── step2_encoded_features.csv
│       ├── step3_binned_features.csv
│       ├── step4_time_features.csv
│       ├── step5_anomaly_flags.csv
│       └── final_engineered_features.csv
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── derive_computed_columns.py
│   ├── encode_categorical_features.py
│   ├── bin_numeric_ranges.py
│   ├── time_based_feature_extraction.py
│   └── flag_anomalies_column.py
│
├── tests/                            # Unit tests
│   └── test_features.py
│
└── notebooks/                        # Jupyter notebooks
    └── demo.ipynb
```

## 💻 Usage

### Running Individual Modules

Each module can be run independently:
```bash
# Module 1: Derive Computed Columns
python src/derive_computed_columns.py

# Module 2: Encode Categorical Features
python src/encode_categorical_features.py

# Module 3: Bin Numeric Ranges
python src/bin_numeric_ranges.py

# Module 4: Time-Based Feature Extraction
python src/time_based_feature_extraction.py

# Module 5: Flag Anomalies
python src/flag_anomalies_column.py
```

### Running the Complete Pipeline
```bash
python main.py
```

### Using as a Python Module
```python
from src.derive_computed_columns import derive_computed_columns
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Apply feature engineering
df_engineered = derive_computed_columns(df)

# Save results
df_engineered.to_csv('output.csv', index=False)
```

## 📊 Sample Output

The pipeline transforms data like this:

**Before (Original):**
```
customer_id | age | income | purchase_amount | gender
1           | 25  | 45000  | 150.00         | Male
```

**After (Engineered):**
```
customer_id | age | age_group | income | income_bracket | purchase_amount | 
1           | 25  | 18-25     | 45000  | Lower-Middle   | 150.00         |

final_price | gender_Male | has_any_anomaly | ... (50+ more features)
145.50      | 1           | 0               | ...
```

## 🔧 Configuration

Modify parameters in each module:
```python
# In flag_anomalies_column.py
threshold = 3  # Z-score threshold
multiplier = 1.5  # IQR multiplier

# In bin_numeric_ranges.py
age_bins = [0, 25, 35, 50, 65, 100]
income_bins = [0, 30000, 50000, 75000, 100000, np.inf]
```

## 🧪 Testing

Run unit tests:
```bash
python -m pytest tests/
```

## 📈 Performance

- **Processing Speed**: ~1000 rows/second
- **Memory Usage**: Optimized for datasets up to 1M rows
- **Output Size**: Typically 3-5x original columns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Known Issues

- Large datetime ranges may cause memory issues
- Some categorical encodings may need manual adjustment for your specific dataset

## 📧 Contact

**Group 6 Team**
- Email: group6@example.com
- Project Link: https://github.com/yourusername/feature-engineering

## 🙏 Acknowledgments

- Pandas Documentation
- Scikit-learn Documentation
- Feature Engineering Best Practices

---

**Made with ❤️ by Group 6**