    # -*- coding: utf-8 -*-
# @Time    : 2026/5/7 11:57
# @Author  : gaolei
# @FileName: mysql_demo.py
# @Software: PyCharm

import pandas as pd
from sqlalchemy import create_engine, types

# ---------- 1. 配置数据库连接 ----------
engine = create_engine('mysql+pymysql://root:mysql@localhost:3306/security?charset=utf8mb4')

# ---------- 2. 读取 CSV ----------
df = pd.read_csv('predict_tes_data/881281.csv', encoding='utf-8')
print(df.head())  # 预览数据

# ---------- 3. 写入 MySQL ----------
df.to_sql(
    name='employees',
    con=engine,
    if_exists='replace',   # 第一次写入用 replace，后续可改 append
    index=False,
    dtype={
        'name': types.NVARCHAR(length=50),
        'department': types.NVARCHAR(length=50),
        'salary': types.DECIMAL(10, 2),
        'hire_date': types.DATE
    }
)

print("数据写入 MySQL 成功！")
# from sqlalchemy import create_engine
#
# # 替换以下占位符为你的真实信息
# user = 'root'      # 用户名
# password = 'mysql'  # 密码
# host = 'localhost'          # 主机地址，本地为 localhost 或 127.0.0.1
# port = 3306                 # MySQL 默认端口
# database = 'security'  # 数据库名
#
# # 创建引擎
# engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4')