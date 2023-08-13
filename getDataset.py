#PPI_features是收集到的20204个蛋白质特征，Training set.xlsx是已证实的药物靶点数据
#制作label数据：已证实的为1，剩余写成0
import pandas as pd
import random

# 读取数据
PPI_features = pd.read_excel("data/PPI_features.xlsx")
training_set = pd.read_excel("data/Training set.xlsx")

# 创建空的out数据表
out = pd.DataFrame(columns=PPI_features.columns)

# 得到Training_Set的第一列的元素
training_set_proteins = training_set.iloc[:, 0]

# 记录已选中行
selected_rows = set()

# 遍历Training_Set的每一个元素
for i in training_set_proteins:
    position = -1
    # 检查元素i是否出现在PPI_features的第一列
    for index, row in PPI_features.iterrows():
        if row["Protein"] == i:
            position = index
            break

    # 如果元素出现在PPI_features的第一列，则将相应行添加到out.xlsx
    if position != -1:
        out = out.append(PPI_features.iloc[position, :])
        selected_rows.add(position)


remaining_rows = set(range(len(PPI_features))) - selected_rows

# 在剩余行
for row_index in remaining_rows:
    out = out.append(PPI_features.iloc[row_index, :])

# 将out输出到out.xlsx文件中
out.to_excel("data/out_all.xlsx", index=False)
