# coding=utf-8


class AlphaTemplate(object):
    def __init__(self, df):
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']
        self.returns = df['close'] / df['close'].shift(1)

        self.vwap = df['volume']
        #self.vwap = (df['S_DQ_AMOUNT'] * 1000) / (df['S_DQ_VOLUME'] * 100 + 1)

        self.config_funcs = []
        self.init_config()

    def init_config(self):
        arr = dir(self)
        for func_name in arr:
            if func_name.startswith("alpha"):
                self.config_funcs.append(func_name)

    def call_func(self, func_name):
        return getattr(self, func_name)()

