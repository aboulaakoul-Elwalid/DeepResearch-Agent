import pandas as pd

# Path to your Parquet file
file_path = "/home/elwalid/projects/parallax_project/0000.parquet"

# Read the Parquet file
df = pd.read_parquet(file_path)

# Display the first 5 rows (books)
print(df.head(5))
