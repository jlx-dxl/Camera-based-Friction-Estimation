import pandas as pd
import os

# 请替换下面的路径为你的CSV文件和图片文件夹的路径
csv_path = 'overall.csv'
image_folder_path = 'Images'
output_csv_path = 'overall_new.csv'

# 读取CSV文件
df = pd.read_csv(csv_path)

# 检查文件是否存在，并过滤数据
df['exists'] = df.iloc[:, 0].apply(lambda x: os.path.isfile(os.path.join(image_folder_path, x)))
filtered_df = df[df['exists']]

# 删除辅助列
filtered_df = filtered_df.drop(columns=['exists'])

# 保存过滤后的CSV
filtered_df.to_csv(output_csv_path, index=False)

print("过滤后的CSV已保存至:", output_csv_path)