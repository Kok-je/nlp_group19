import pandas as pd

def merge_reasoning_to_data(original_data_file_path: str, reasoning_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads a JSONL file with original data, then merges in the 'reasoning' column
    from the provided dataframe based on the 'id' column.

    Args:
        original_data_file_path (str): Path to a JSONL file containing the original data.
        reasoning_df (pd.DataFrame): A DataFrame with two columns: 'id' (string) and 'reasoning' (string).

    Returns:
        pd.DataFrame: A DataFrame resulting from merging the original data with the reasoning information.
    """
    # Read the JSONL file into a DataFrame
    original_data_df = pd.read_json(original_data_file_path, lines=True)
    
    # Check that the number of rows in the original data matches the reasoning dataframe
    if len(original_data_df) != len(reasoning_df):
        raise ValueError(f"Row count mismatch: original data has {len(original_data_df)} rows, but reasoning data has {len(reasoning_df)} rows.")
    
    # Merge on the 'id' column. Using a left join to keep all original data rows.
    merged_df = original_data_df.merge(reasoning_df, on='id', how='left')
    
    return merged_df

if __name__ == "__main__":
    # Read the original data first to know the number of rows and the corresponding ids
    original_data_df = pd.read_json("data/test.jsonl", lines=True)
    
    # Create fake reasoning DataFrame using the 'id' values from the original data
    fake_reasoning_data = {
        'id': original_data_df['id'].tolist(),
        'reasoning': [f"This is reasoning for id {id}" for id in original_data_df['id']]
    }
    reasoning_df = pd.DataFrame(fake_reasoning_data)
    
    # Merge reasoning data with the original data
    merged_df = merge_reasoning_to_data("data/test.jsonl", reasoning_df)
    
    # Print out the resulting merged dataframe
    print(merged_df.head())


