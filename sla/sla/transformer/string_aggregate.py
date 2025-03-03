import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Optional
from datetime import datetime, timedelta


class DataFrameStringAggregator:
    """
    A class to perform aggregations over DataFrames with string columns.
    
    This class focuses on aggregating data based on string columns and time periods,
    allowing for various aggregation operations like count, sum, mean, etc.
    
    Attributes:
        df (pd.DataFrame): The DataFrame to be aggregated.
        time_column (str): The column containing time/date information.
        string_columns (List[str]): Columns containing string data to aggregate by.
    """

    def __init__(self, df: pd.DataFrame, time_column: str, string_columns: Optional[List[str]] = None):
        """
        Initialize the DataFrameStringAggregator with a DataFrame.
        
        Args:
            df: A pandas DataFrame to perform aggregations on.
            time_column: Column name containing datetime information.
            string_columns: List of column names containing string data to aggregate by.
                If None, automatically detect string columns.
        """
        self.df = df.copy()
        self.time_column = time_column
        
        # Validate time column
        if self.time_column not in self.df.columns:
            raise ValueError(f"Time column '{time_column}' not found in DataFrame")
        
        # Convert time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[time_column]):
            try:
                self.df[time_column] = pd.to_datetime(self.df[time_column])
            except Exception as e:
                raise ValueError(f"Could not convert '{time_column}' to datetime: {str(e)}")
        
        # Automatically detect string columns if not provided
        if string_columns is None:
            self.string_columns = [
                col for col in self.df.columns 
                if pd.api.types.is_string_dtype(self.df[col]) and col != time_column
            ]
        else:
            # Validate provided string columns
            for col in string_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
            self.string_columns = string_columns

    def count_by_time_period(
        self, 
        string_column: str, 
        time_period: str = 'D', 
        normalize: bool = False,
        fill_missing: bool = True,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Count occurrences of each string value in specified column grouped by time period.
        
        Args:
            string_column: The string column to aggregate by.
            time_period: Time period to group by ('D' for day, 'M' for month, etc.).
                         See pandas time offset aliases for more options.
            normalize: If True, calculate proportions instead of counts.
            fill_missing: If True, fill missing time periods with zeros.
            top_n: If specified, return only the top N most frequent values.
                   
        Returns:
            A DataFrame with time periods as index and string values as columns,
            containing counts or proportions of each string value.
        """
        if string_column not in self.string_columns:
            raise ValueError(f"'{string_column}' is not a valid string column")
        
        # Group by time period and string column, then count
        df_grouped = (
            self.df
            .groupby([pd.Grouper(key=self.time_column, freq=time_period), string_column])
            .size()
            .reset_index(name='count')
            .pivot(index=self.time_column, columns=string_column, values='count')
            .fillna(0)
        )
        
        # Normalize if requested
        if normalize:
            df_grouped = df_grouped.div(df_grouped.sum(axis=1), axis=0)
        
        # Fill missing time periods if requested
        if fill_missing:
            idx = pd.date_range(
                start=self.df[self.time_column].min(),
                end=self.df[self.time_column].max(),
                freq=time_period
            )
            df_grouped = df_grouped.reindex(idx, fill_value=0)
        
        # Filter for top N values if requested
        if top_n is not None and top_n > 0:
            top_columns = df_grouped.sum().nlargest(top_n).index
            df_grouped = df_grouped[top_columns]
        
        return df_grouped
    
    def custom_aggregation(
        self,
        string_column: str,
        value_column: str,
        agg_func: Union[str, List[str]],
        time_period: str = 'D',
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        Perform custom aggregation on a value column grouped by time and string values.
        
        Args:
            string_column: The string column to group by.
            value_column: The column containing values to aggregate.
            agg_func: Aggregation function(s) to apply. Either a string ('sum', 'mean', etc.)
                      or a list of strings ['sum', 'mean', 'max'].
            time_period: Time period to group by.
            fill_missing: If True, fill missing time periods.
            
        Returns:
            A DataFrame with time periods as index and string values as columns,
            containing the aggregated values.
        """
        if string_column not in self.string_columns:
            raise ValueError(f"'{string_column}' is not a valid string column")
        
        if value_column not in self.df.columns:
            raise ValueError(f"Value column '{value_column}' not found in DataFrame")
        
        # Group by time period and string column, then aggregate
        df_grouped = (
            self.df
            .groupby([pd.Grouper(key=self.time_column, freq=time_period), string_column])
            [value_column]
            .agg(agg_func)
        )
        
        # Handle multi-level columns resulting from multiple aggregation functions
        df_grouped = df_grouped.unstack(level=string_column)
            
        # Fill missing values
        df_grouped = df_grouped.fillna(0)
        
        # Fill missing time periods if requested
        if fill_missing:
            idx = pd.date_range(
                start=self.df[self.time_column].min(),
                end=self.df[self.time_column].max(),
                freq=time_period
            )
            df_grouped = df_grouped.reindex(idx, fill_value=0)
        
        return df_grouped
    
    def get_unique_values(self, string_column: str) -> List[str]:
        """
        Get all unique values in a string column.
        
        Args:
            string_column: The string column to get unique values from.
            
        Returns:
            A list of unique values in the column.
        """
        if string_column not in self.string_columns:
            raise ValueError(f"'{string_column}' is not a valid string column")
            
        return self.df[string_column].unique().tolist()
    

# Example Usage
if __name__ == "__main__":
    # Generate sample data with a date range, categories, and values
    np.random.seed(42)  # For reproducibility
    
    # Create date range for the past 30 days
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create categories and products
    categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Books']
    products = ['Laptop', 'Phone', 'T-shirt', 'Jeans', 'Lamp', 'Chair', 'Bread', 'Fruit', 'Novel', 'Textbook']
    
    # Create a mapping from products to categories
    product_to_category = {
        'Laptop': 'Electronics', 'Phone': 'Electronics',
        'T-shirt': 'Clothing', 'Jeans': 'Clothing',
        'Lamp': 'Home', 'Chair': 'Home',
        'Bread': 'Food', 'Fruit': 'Food',
        'Novel': 'Books', 'Textbook': 'Books'
    }
    
    # Generate 1000 random records
    n_records = 1000
    random_dates = [dates[np.random.randint(0, len(dates))] for _ in range(n_records)]
    random_products = [products[np.random.randint(0, len(products))] for _ in range(n_records)]
    random_values = np.random.randint(10, 200, size=n_records)
    
    # Create DataFrame
    data = {
        'timestamp': random_dates,
        'product': random_products,
        'value': random_values
    }
    df = pd.DataFrame(data)
    
    # Add category based on product
    df['category'] = df['product'].map(product_to_category)
    
    print("Sample of the generated dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    
    # Create an instance of the aggregator
    aggregator = DataFrameStringAggregator(df, time_column='timestamp', string_columns=['category', 'product'])
    
    # Example 1: Daily count of products
    print("\n--- Example 1: Daily count of products ---")
    daily_product_counts = aggregator.count_by_time_period('product', time_period='D', top_n=5)
    print("Top 5 products by daily count:")
    print(daily_product_counts.head())

    
    # Example 2: Weekly category counts with normalization (proportions)
    print("\n--- Example 2: Weekly category proportions ---")
    weekly_category_props = aggregator.count_by_time_period(
        'category', time_period='W', normalize=True
    )
    print("Weekly category proportions:")
    print(weekly_category_props)
    
    
    # Example 3: Custom aggregation - Daily average value by category
    print("\n--- Example 3: Daily average value by category ---")
    daily_avg_value = aggregator.custom_aggregation(
        'category', 'value', 'mean', time_period='D'
    )
    print("Daily average value by category:")
    print(daily_avg_value.head())
    
    
    # Example 4: Multiple aggregations
    print("\n--- Example 4: Multiple aggregations ---")
    # Using a list of aggregation functions instead of a dictionary
    agg_funcs = ['sum', 'mean', 'max']
    
    multi_agg = aggregator.custom_aggregation(
        'category', 'value', agg_funcs, time_period='W'
    )
    
    print("Multi-aggregation result structure:")
    print(multi_agg.columns)
    
    # Extract just the max aggregation for each category for demonstration
    weekly_max_by_category = multi_agg.xs('max', axis=1, level=0)
    print("Weekly maximum value by category:")
    print(weekly_max_by_category)
    
    
    # Example 5: Getting unique values
    print("\n--- Example 5: Unique values in category ---")
    unique_categories = aggregator.get_unique_values('category')
    print(f"Unique categories: {unique_categories}")
    
    print("\nExample usage completed successfully!")