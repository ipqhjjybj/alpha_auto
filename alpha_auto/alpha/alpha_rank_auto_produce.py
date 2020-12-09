# coding=utf-8

import operator
import time
import random
from copy import copy

from deap import gp, creator, base, tools, algorithms

from alpha_auto.function.technique import PD_Technique, MultiIndexMethod
from alpha_auto.function import parse_maxint_from_str, is_arr_sorted

from .basic_func import *
from .alpha_template import AlphaTemplate


class AlphaRankAutoProduce(AlphaTemplate):
    """
    通过横截面方式找 最好的排名前多少的股票这样
    """
    def __init__(self, df, population_num=300, canshu_nums=0, name="auto"):
        super(AlphaRankAutoProduce, self).__init__(df)

        self._population_num = population_num
        self._name = name
        self._rank_name = "rank_{}".format(name)
        self._rate_name = "rate_{}".format(1)

        self._df = df
        self._canshu_nums = canshu_nums

        self._df = PD_Technique.rate(self._df, 1, field="close", name=self._rate_name)

        self.pset = self.init_primitive_set()
        self.toolbox = self.init_toolbox()

    def evalfunc(self, individual):
        code = str(individual)
        tdf = copy(self._df)
        tdf[self._name] = self.toolbox.compile(expr=individual)
        max_int = parse_maxint_from_str(code)
        max_int = max(max_int, 30)

        tdf = MultiIndexMethod.get_multi_index_drop_nums(tdf, max_int)
        # 排序
        df = MultiIndexMethod.get_multi_index_rank_by_key2(tdf, self._name, reverse=False)
        res = MultiIndexMethod.get_rank_ave_score(df, rank_colume_name=self._rank_name,
                                                  rank_rate_name=self._rate_name, denominator=4)
        # 删除最后一行
        res.drop(res.index[-1], inplace=True)

        ans = [res[col].mean() for col in res.columns]
        print(code, ans, max(ans[0], ans[-1]), is_arr_sorted(ans))
        return max(ans[0], ans[-1]), is_arr_sorted(ans),

    def init_primitive_set(self):
        pset = gp.PrimitiveSet("MAIN", self._canshu_nums)

        #下面这个函数要修改
        #pset.addPrimitive(if_then_else, 3)
        pset.addPrimitive(rank, 1)
        for window in range(5, 80, 5):
            pset.addPrimitive(make_partial_func(ts_sum, window=window), 1)
            pset.addPrimitive(make_partial_func(sma, window=window), 1)
            pset.addPrimitive(make_partial_func(stddev, window=window), 1)
            pset.addPrimitive(make_partial_func(correlation, window=window), 2)
            pset.addPrimitive(make_partial_func(covariance, window=window), 2)
            pset.addPrimitive(make_partial_func(ts_rank, window=window), 1)
            pset.addPrimitive(make_partial_func(product, window=window), 1)
            pset.addPrimitive(make_partial_func(ts_min, window=window), 1)
            pset.addPrimitive(make_partial_func(ts_max, window=window), 1)
            pset.addPrimitive(make_partial_func(delta, window=window), 1)
            pset.addPrimitive(make_partial_func(delay, window=window), 1)
            pset.addPrimitive(make_partial_func(ts_argmax, window=window), 1)
            pset.addPrimitive(make_partial_func(ts_argmin, window=window), 1)

            # 下面这个函数要修改下
            #pset.addPrimitive(make_partial_func(decay_linear, window=window), 1)

        pset.addTerminal(self.open, "open")
        pset.addTerminal(self.high, "high")
        pset.addTerminal(self.low, "low")
        pset.addTerminal(self.close, "close")
        pset.addTerminal(self.volume, "volume")
        pset.addTerminal(self.returns, "returns")

        # pset.renameArguments(ARG0='x')

        # pset.addTerminal(self.vwap, "vwap")
        # pset.addPrimitive(make_partial_func(scale, constant=0.3), 1)
        return pset

    def init_toolbox(self):
        creator.create("FitnessAssume", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessAssume)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        toolbox.register("evaluate", self.evalfunc)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        return toolbox

    def run(self, cxpb=0.5, mutpb=0.1, ngen=40):
        random.seed(int(time.time()))

        pop = self.toolbox.population(n=self._population_num)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, ngen, stats=mstats,
                                       halloffame=hof, verbose=True)
        # print log
        return pop, log, hof

