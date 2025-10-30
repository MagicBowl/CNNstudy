import pyarrow.parquet as pq
import pandas as pd
# 读取Parquet文件
table = pq.read_table('b.parquet')
df = table.to_pandas()
print(df)