import array
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

target_1 = 0.3
target_2 = 0.1

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
NDIM = 3  # 次元数
BOUND_LOW = [0, 0, 0]
BOUND_UP = [0.5, 1, 0.5,]

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

# 目的関数
def evaluate(individual):
    x1, x2, x3 = individual
    inputs = np.array([[x1, x2, x3]]).reshape(1, -1)
    predicted = forward_model(inputs) # 学習済みモデルにインプット
    obj_1 = np.sum((predicted[0][0].numpy() - target_1) ** 2)
    obj_2 = np.sum((predicted[0][1].numpy() - target_2) ** 2)
    return obj_1, obj_2

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main():
    random.seed(64)
    MU = 100 # 個体数
    NGEN = 100 # 繰り返し世代数
    CXPB = 0.9 # 交叉率

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # 第一世代の生成
    pop = toolbox.population(n=MU)
    pop_init = pop[:]
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # 最適計算の実行
    for gen in range(1, NGEN):
        # 子母集団生成
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 交叉と突然変異
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # 交叉させる個体を選択
            if random.random() <= CXPB:
                # 交叉
                toolbox.mate(ind1, ind2)

            # 突然変異
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)

            # 交叉と突然変異させた個体は適応度を削除する
            del ind1.fitness.values, ind2.fitness.values

        # 適応度を削除した個体について適応度の再評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 次世代を選択
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    # 最終世代のハイパーボリュームを出力
    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, pop_init, logbook

if __name__ == "__main__":
    pop, pop_init, stats = main()

# ポジションを出力
for i in range(len(pop)):
  print("{} → x1：{:.3f}, x2：{:.3f}, x3：{:.3f}, （{}, {}）".format(i, pop[i][0], pop[i][1], pop[i][2], fitnesses[i][0], fitnesses[i][1]))
  print("")

# パレート解プロット
fitnesses_init = np.array([list(pop_init[i].fitness.values) for i in range(len(pop_init))])
fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])
plt.plot(fitnesses_init[:,0], fitnesses_init[:,1], "b.", label="Initial")
plt.plot(fitnesses[:,0], fitnesses[:,1], "r.", label="Optimized" )
plt.legend(loc="upper right")
plt.title("optimization", fontsize=16)
plt.xlabel("obj1", fontsize=14, color="red")
plt.ylabel("obj2", fontsize=14, color="red")
plt.ylim(-0.00005, 0.0025)
plt.xlim(-0.00005, 0.0006)
plt.grid(True)
