import sqlite3
import pandas as pd

conn = sqlite3.connect('data/blood_group.db')
df = pd.read_sql_query('SELECT * FROM predictions', conn)
print("Total records:", len(df))
print("Last 15 records:")
print(df[['id', 'timestamp', 'input_type', 'image_path', 'predicted_group', 'confidence', 'finger_used']].tail(15))
