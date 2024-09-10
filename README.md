# ASSIGNMENT
Tasks:

1. Data Ingestion:
• Develop a script that can ingest OHLC data feeds from various sources and formats.
• Validate the data integrity (e.g., check for missing values, outliers, data type consistency).
• Standardize the data format to a common structure (e.g., pandas DataFrame).
2. Data Cleaning:
• Identify and handle missing values (e.g., imputation, removal).
• Detect and correct outliers using statistical methods or domain knowledge.
• Address any inconsistencies in timestamps or date formats.
3. Data Transformation:
• Calculate technical indicators based on OHLC data (e.g., moving averages, Bollinger Bands, Relative Strength Index).
• Apply feature engineering techniques to create new features relevant for your trading strategy (e.g., volatility measures, price patterns).
• Resample the data based on desired frequencies (e.g., daily to hourly).
4. Data Validation:
• Implement unit tests to ensure the pipeline's functionality and data integrity.
• Monitor the pipeline for errors and data quality issues.
5. Data Storage:
• Use a simple DB to store this (such as Sqlite, Mysql etc)
• Partition the data by year, month, or another relevant category for efficient querying.
• Optimize the data format for fast retrieval and analysis (e.g., columnar format).
