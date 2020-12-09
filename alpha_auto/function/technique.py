# encoding: UTF-8

import pandas as pd
import numpy as np
from collections import defaultdict


class DFMethod(object):
    @staticmethod
    def count(series, n):
        '''
        统计最近n 行中 , > 0 的出现次数
        '''
        if n <= 0:
            return pd.Series([])
        ret = []
        num = 0
        for i in range(len(series)):
            if i >= n:
                num -= series[i - n] > 0
            num += series[i] > 0
            ret.append(num)
        return pd.Series(ret)

    @staticmethod
    def prod(series, n):
        '''
        序列最近n个数的累乘
        '''
        if n <= 0:
            return pd.Series([])
        ret = []
        num = 1
        for i in range(len(series)):
            if i >= n:
                num = num / series[i - n]
            num *= series[i]
            ret.append(num)
        return pd.Series(ret)

    @staticmethod
    def rank(series, n):
        '''
        滚动求出，每个数字在最近最近n个数中排名第几
        '''
        if n <= 0:
            return pd.Series([])
        nums = []
        ret = []
        for i in range(len(series)):
            if i >= n:
                nums.remove(series[i - n])
            val = series[i]
            index = len(nums)
            nums.append(val)
            j = index - 1
            while j >= 0 and nums[j] > nums[j + 1]:
                k = nums[j]
                nums[j] = nums[j + 1]
                nums[j + 1] = k
                j = j - 1
                index = j + 1
            ret.append(index + 1)
        return pd.Series(ret)

    @staticmethod
    def question_mark(condition_series, true_series, false_series):
        ret = []
        for i in range(len(condition_series)):
            if condition_series[i]:
                ret.append(true_series[i])
            else:
                ret.append(false_series[i])
        return pd.Series(ret)

    @staticmethod
    def sum_if(series, n, condition_series):
        '''
        滚动求出，最近n个数字中，满足条件的累计
        '''
        if n <= 0:
            return pd.Series([])
        ret = []
        num = 0
        for i in range(len(condition_series)):
            if i >= n:
                if condition_series[i - n]:
                    num -= series[i - n]
            if condition_series[i]:
                num += series[i]
            ret.append(num)
        return pd.Series(ret)

    @staticmethod
    def corr(series_a, series_b, n):
        if n <= 0:
            return pd.Series([])
        ret = []
        va = []
        vb = []
        for i in range(len(series_a)):
            if i >= n:
                va.pop(0)
                vb.pop(0)
            va.append(series_a[i])
            vb.append(series_b[i])
            val = np.mean(np.multiply((va - np.mean(va)), (vb - np.mean(vb)))) / (np.std(vb) * np.std(va))
            ret.append(val)
        return pd.Series(ret)

    @staticmethod
    def coviance(series_a, series_b, n):
        '''
        序列A,B 过去n天的协方差
        '''
        if n <= 0:
            return pd.Series([])
        ret = []
        va = []
        vb = []
        for i in range(len(series_a)):
            if i >= n:
                va.pop(0)
                vb.pop(0)
            va.append(series_a[i])
            vb.append(series_b[i])
            val = np.cov(va, vb)[0, 1]
            ret.append(val)
        return pd.Series(ret)


class MultiIndexMethod(object):
    @staticmethod
    def get_filter_symbols(df, min_num_rows=365):
        '''
        过滤掉df 中, 行数少于 365 的票
        '''
        coin_dict = defaultdict(list)
        rows = df.index
        for row in rows:
            coin = row[0]
            coin_dict[coin].append(row)

        to_delete_arr = []
        for coin, values in coin_dict.items():
            if len(values) < min_num_rows:
                to_delete_arr.extend(values)
        return df.drop(to_delete_arr)

    @staticmethod
    def get_multi_index_drop_nums(df, drop_first_nums=1):
        '''
        删去多元索引中，按照第一个索引的前几行
        '''
        rows = df.index
        ret = []
        start_num = 0
        pre_key = ""
        for row in rows:
            key = row[0]
            if key != pre_key:
                start_num = 0
            if start_num < drop_first_nums:
                ret.append(row)
                start_num += 1
            pre_key = key
        return df.drop(ret)

    @staticmethod
    def get_multi_index_rank_by_key2(df, colume_name, rank_colume_name="", reverse=True):
        '''
        计算多元索引中的，按照 第二列 groupby 的rank排序
        '''
        if not rank_colume_name:
            rank_colume_name = "rank_{}".format(colume_name)
        k_dic = defaultdict(list)
        rows = df.index
        values = df[colume_name]
        for i in range(len(rows)):
            _, key = rows[i]
            val = values[i]
            k_dic[key].append((val, i))

        ret = []
        for _, arr in k_dic.items():
            arr.sort(reverse=reverse)
            for i in range(len(arr)):
                _, index = arr[i]
                ret.append((index, i + 1))

        ret.sort()
        ret = [x[1] for x in ret]
        df[rank_colume_name] = np.array(ret)
        return df

    @staticmethod
    def get_rank_ave_score(df, rank_colume_name="", rank_rate_name="", denominator=3):
        '''
        rank_colume_name, 排名的那列
        rank_rate_name, 未来一天的涨跌幅
        对于 denominator = 3
        通过排名， 按照 排名前1/3 放第一组， 中间的放第二组， 剩余的放第三组
        '''
        all_stocks = set([])
        k_dic = defaultdict(list)
        rows = df.index
        rank_colume_values = df[rank_colume_name]
        rank_rate_values = df[rank_rate_name]
        for i in range(len(rows)):
            stock, key = rows[i]
            rank = rank_colume_values[i]
            rate = rank_rate_values[i]

            k_dic[key].append((stock, int(rank), rate))
            all_stocks.add(stock)

        len_stocks = len(all_stocks)
        index_arr = []
        result = {}
        for i in range(denominator):
            result[i] = []

        keys = list(k_dic.keys())
        keys.sort()
        for key in keys:
            arr = k_dic[key]
            ll = len(arr)
            if ll < len_stocks:
                continue

            if ll % denominator == 0:
                group_num = int(ll / denominator)
            else:
                group_num = int(ll / denominator) + 1

            res = [(0, 0)] * denominator
            for stock, rank, rate in arr:
                ind = int((rank - 1) / group_num)
                pre_val, pre_num = res[int((rank - 1) / group_num)]
                pre_val += rate
                pre_num += 1
                res[ind] = (pre_val, pre_num)

            for i in range(denominator):
                val, num = res[i]
                if num > 0:
                    result[i].append(val * 1.0 / num)
                else:
                    result[i].append(0)

            index_arr.append(key)

        return pd.DataFrame(result, index=index_arr)


class PD_Technique(object):
    @staticmethod
    def rate(df, n, field="close", name=None):
        if not name:
            name = "rate_{}".format(n)
        df[name] = df[field].shift(-n) / df[field] - 1
        return df

    @staticmethod
    def quick_income_compute(df, sllippage, rate, size=1, name="income", pos_name="pos"):
        '''
        :param df: 传入的矩阵df
        :param sllippage: 滑点
        :param rate: 交易的手续费 0.1 表示千一
        :return:
        '''
        close_arr = list(df["close"])
        open_arr = list(df["open"])
        close_arr.append(close_arr[-1])  # 多加一个close, 用于后面补充计算
        open_arr.append(open_arr[-1])  # 多加一个open, 用于后面补充计算
        pos_arr = list(df[pos_name])
        for i in range(len(pos_arr)):
            if str(pos_arr[i]) == "nan":
                pos_arr[i] = 0

        income_ret = []
        ll = len(pos_arr)
        income = 0
        pre_pos = 0
        last_entry_price = 0
        new_entry_price = 0
        for i in range(ll):
            fee = size * abs(pos_arr[i] - pre_pos) * (close_arr[i] * rate / 100.0 + sllippage)
            pc_pos = 0
            exit_price = 0
            if pre_pos > 0:
                if pos_arr[i] < pre_pos:
                    pc_pos = min(pre_pos, pre_pos - pos_arr[i])
                    exit_price = open_arr[i + 1]

                    if pos_arr[i] < 0:
                        new_entry_price = open_arr[i + 1]

                elif pos_arr[i] > pre_pos:
                    pc_pos = 0
                    exit_price = 0
                    new_sz = pos_arr[i] - pre_pos
                    new_entry_price = (last_entry_price * abs(pre_pos) + open_arr[i + 1] * new_sz) / abs(pos_arr[i])

            elif pre_pos < 0:
                if pos_arr[i] < pre_pos:
                    pc_pos = 0
                    exit_price = 0
                    new_sz = abs(pos_arr[i] - pre_pos)
                    new_entry_price = (last_entry_price * abs(pre_pos) + open_arr[i + 1] * new_sz) / abs(pos_arr[i])

                elif pos_arr[i] > pre_pos:
                    pc_pos = min(pos_arr[i] - pre_pos, abs(pre_pos))
                    exit_price = open_arr[i + 1]
                    if pos_arr[i] > 0:
                        new_entry_price = open_arr[i + 1]
            else:
                if pos_arr[i] > 0:
                    new_entry_price = open_arr[i + 1]
                elif pos_arr[i] < 0:
                    new_entry_price = open_arr[i + 1]

            if pre_pos > 0:
                direction = 1
            elif pre_pos < 0:
                direction = -1
            else:
                direction = 0
            pnl = size * pc_pos * (exit_price - last_entry_price) * direction
            income += pnl - fee

            income_ret.append(income)
            last_entry_price = new_entry_price
            pre_pos = pos_arr[i]
        df[name] = np.array(income_ret)
        return df

    @staticmethod
    def quick_compute_current_drawdown(df, name_cur_down="cur_drawdown", name_max_drawdown="max_drawdown"):
        '''
        通过income计算 当前一次离最近的高点的最大回撤
        '''
        income_arr = list(df["income"])
        cur_drawdown_arr = []
        max_drawdown_arr = []
        n = len(income_arr)
        now_max_income = 0
        pre_max_drawdown = 0
        for i in range(0, n):
            now_max_income = max(income_arr[i], now_max_income)
            cur_down = now_max_income - income_arr[i]
            pre_max_drawdown = max(pre_max_drawdown, cur_down)
            cur_drawdown_arr.append(cur_down)
            max_drawdown_arr.append(pre_max_drawdown)

        df[name_cur_down] = np.array(cur_drawdown_arr)
        df[name_max_drawdown] = np.array(max_drawdown_arr)
        return df

    @staticmethod
    def assume_strategy(df, trade_days=365):
        '''
        通过income计算策略
        '''
        entry_price = 0
        pre_pos = 0
        now_day = ""
        daily_result = []
        close_arr = list(df["close"])
        income_arr = list(df["income"])
        datetime_arr = list(df["datetime"])
        pos_arr = list(df["pos"])
        n = len(income_arr)
        for i in range(n):
            pos = pos_arr[i]
            close = close_arr[i]
            if pos != pre_pos:
                if pre_pos == 0:
                    entry_price = close
                elif pre_pos > 0:
                    if pos < 0:
                        entry_price = close
                    elif pos > pre_pos:
                        entry_price = (entry_price * pre_pos + (pos - pre_pos) * close) * 1.0 / pos
                else:
                    if pos > 0:
                        entry_price = close
                    elif pos < pre_pos:
                        entry_price = (entry_price * pre_pos + (pos - pre_pos) * close) * 1.0 / pos

            dt = datetime_arr[i]
            d, t = dt.split(' ')
            income = income_arr[i]
            if (now_day and now_day != d) or (i == n - 1 and now_day == d):
                daily_result.append(income + (close - entry_price) * pos)
                # daily_result.append(income)

            now_day = d

        df = pd.DataFrame(np.array(daily_result), columns=['balance'])
        df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
        daily_return = df["return"].mean() * 100
        return_std = df["return"].std() * 100

        sharpe_ratio = daily_return / return_std * np.sqrt(trade_days)

        return sharpe_ratio

