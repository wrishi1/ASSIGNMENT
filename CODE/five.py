import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
import unittest

# Data Ingestion
def ingest_data(file_path):
    """Reads CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        raise

# Handling Missing Values
def handle_missing_values(df, method='drop'):
    """Handles missing values by dropping or filling them."""
    if method == 'drop':
        df = df.dropna()
    elif method == 'ffill':
        df = df.fillna(method='ffill') 
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())
    return df

# Outlier Detection and Handling
def detect_and_handle_outliers(df, method='remove'):
    """Detects and handles outliers in Open, High, Low, Close columns."""
    numeric_cols = ['Open', 'High', 'Low', 'Close']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'remove':
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'cap':
            df[col] = np.clip(df[col], lower_bound, upper_bound)  # Limit outliers to bounds

    df['Close'] = df[['Close', 'Low', 'High']].apply(lambda x: min(max(x['Close'], x['Low']), x['High']), axis=1)
    
    return df

# Standardize Data and Handle Date Format
def standardize_data(df):
    """Ensures the data has required columns and correct types."""
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Data Quality Check
def check_data_quality(df):
    """Checks the data for missing values, outliers, and inconsistencies."""
    issues = []
    
# Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        issues.append(f"Missing values:\n{missing_values[missing_values > 0]}")
    
# Check for outliers
    for col in ['Open', 'High', 'Low', 'Close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outliers.any():
            issues.append(f"Outliers in {col}: {df.loc[outliers, ['Date', col]]}")
    
# Check for date inconsistencies
    if df['Date'].isnull().any():
        issues.append("Null values in 'Date'.")
    
    if issues:
        raise ValueError("\n".join(issues))
    else:
        print("Data passed quality check.")

# SQLite Connection Setup
def create_connection(db_file):
    """Creates a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite database: {db_file}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
    return conn

# Create Table in SQLite
def create_table(conn):
"""Creates an OHLC table in the SQLite database."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS ohlc_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
    except Exception as e:
        print(f"Error creating table: {e}")

# Insert Data into SQLite
def insert_data(conn, df):
    """Inserts OHLC data into the SQLite database."""
    insert_sql = """
    INSERT INTO ohlc_data (date, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    try:
        cursor = conn.cursor()
        data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].values.tolist()
        cursor.executemany(insert_sql, data)
        conn.commit()
        print(f"Inserted {len(data)} records.")
    except Exception as e:
        print(f"Error inserting data: {e}")

# Query Data by Year and Month
def query_data(conn, year, month):
    """Queries data by year and month from the database."""
    query_sql = """
    SELECT * FROM ohlc_data
    WHERE strftime('%Y', date) = ? AND strftime('%m', date) = ?;
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query_sql, (str(year), str(month).zfill(2)))
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=['id', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    except Exception as e:
        print(f"Error querying data: {e}")

# Complete Pipeline: Ingest, Process, Store, and Query Data
def run_pipeline(file_path, db_file, year, month):
    """Runs the full data pipeline: Ingest, Process, Store, Query."""
    # Ingest and clean data
    df = ingest_data(file_path)
    df = handle_missing_values(df)
    df = detect_and_handle_outliers(df)
    df = standardize_data(df)
    
# Check data quality
    check_data_quality(df)
    
# Connect to SQLite and create table
    conn = create_connection(db_file)
    create_table(conn)
    
# Insert data
    insert_data(conn, df)
    
# Query data for a specific year and month
    queried_df = query_data(conn, year, month)
    print(queried_df)
    
# Close the connection
    if conn:
        conn.close()

# Plot Interactive Chart (Plotly)
def plot_interactive_chart(df):
    """Plots an interactive line chart for 'Close' price."""
    fig = px.line(df, x='Date', y='Close', title='Close Price Chart')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Close Price')
    fig.show()

# Unit Tests
class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=5),
            'Open': [100, 105, 110, 120, 130],
            'High': [110, 115, 120, 125, 135],
            'Low': [90, 100, 105, 115, 125],
            'Close': [105, 110, 115, 125, 130],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
#5tests..
    def test_ingest_data(self):
        # Test data ingestion
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_handle_missing_values(self):
        # Test handling missing values
        df_copy = self.df.copy()
        df_copy.loc[1, 'Close'] = np.nan  # Introduce missing value
        df_filled = handle_missing_values(df_copy, method='mean')
        self.assertFalse(df_filled.isnull().any().any())  # No missing values remain

    def test_detect_and_handle_outliers(self):
        # Test outlier detection and handling
        df_copy = self.df.copy()
        df_copy.loc[0, 'Close'] = 1000000  # Add an outlier
        df_cleaned = detect_and_handle_outliers(df_copy, method='remove')
        self.assertTrue((df_cleaned['Close'] <= df_cleaned['High']).all())  # Ensure Close is within the range

    def test_standardize_data(self):
        # Test data standardization
        df_standardized = standardize_data(self.df)
        self.assertTrue(pd.to_datetime(df_standardized['Date'], errors='coerce').notna().all())  # Valid dates
        for col in ['Open', 'High', 'Low', 'Close']:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_standardized[col]))

    def test_check_data_quality(self):
        # Test data quality check
        try:
            check_data_quality(self.df)
        except ValueError as e:
            self.fail(f"Data quality check failed: {e}")
#printing the output..
if __name__ == "__main__":
    unittest.main()
