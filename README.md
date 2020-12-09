# alpha_auto
    这是一段通过调用deap库，用于自动化寻找alpha因子的脚本。

    通过遗传算法暴力生成因子，然后快速验证求解，部分模块还有待完善。

---------------------------------------------------------------------------------
```python
pip3 install -r requirements.txt
```

---------------------------------------------------------------------------------

```python
import pandas as pd
from alpha_auto.function.technique import MultiIndexMethod
from alpha_auto.alpha.alpha_rank_auto_produce import AlphaRankAutoProduce

file_path = "day.csv"
df = pd.read_csv(file_path, index_col=[0, 2], skipinitialspace=True)

df = MultiIndexMethod.get_filter_symbols(df, 365*2)
a = AlphaRankAutoProduce(df, canshu_nums=0)
a.run()
```
