import pandas as pd

# 载入 CSV 文件
df = pd.read_csv('train.csv')

# 随机选择 100 行
sampled_df = df.sample(n=100)

# 保存到新的 CSV 文件
sampled_df.to_csv('test.csv', index=False)

# 从原始 DataFrame 中删除这些行
df = df.drop(sampled_df.index)

# 保存更新后的原始 CSV 文件
df.to_csv('train.csv', index=False)