# E-Gen Examples

This directory contains example scripts to test and demonstrate E-Gen functionality.

## Data Loading Test

To run the data loading test:

1. Ensure your data files are in place:
   ```
   data/
   └── bio_database/
       ├── table_dict.pkl
       └── data_desc.pkl
   ```

2. From the root directory (C:\Users\difen\Egen), run:
   ```bash
   python examples/test_data_loading.py
   ```

The test will:
- Load data from bio_database
- Display information about available tables
- Test the falsification setup
- Report success or failure 