# coding=utf-8
# 通过横截面找因子

import pandas as pd
from alpha_auto.function.technique import MultiIndexMethod
from alpha_auto.alpha.alpha_rank_auto_produce import AlphaRankAutoProduce

file_path = "day.csv"
df = pd.read_csv(file_path, index_col=[0, 2], skipinitialspace=True)

df = MultiIndexMethod.get_filter_symbols(df, 365*2)
a = AlphaRankAutoProduce(df, canshu_nums=0)
a.run()

