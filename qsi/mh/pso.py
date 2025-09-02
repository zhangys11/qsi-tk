import numpy as np
import random
import matplotlib.pyplot as plt
import time 
import copy
from ..cla import run_multiclass_clfs

class PSO(object):
    '''
    粒子群算法（Particle Swarm Optimization，PSO）

    Parameters
    ----------
    dim : 粒子维度  
    fitness : 适应度函数
    num_particle ： 粒子群规模，初始化种群个数
    max_iter ： 最大迭代次数  50
    [p_low, p_up] ： 粒子位置的约束范围
    [v_low, v_high] ： 粒子速度的约束范围
    w  : 惯性权值
    c1 : 个体学习因子
    c2 : 群体学习因子

    Example
    -------
    low = [1, 1, 1, 1, 1, 1]
    up = [25, 25, 25, 25, 25, 25]

    pso = PSO(fitness = lambda x:  - (x[0] * np.exp(x[1]) + x[2] * np.sin(x[1]) + x[3] * x[4] * x[5]),
            dim=6, num_particle=1000, max_iter=50, p_low=low, p_up=up, 
            v_low=-1, v_high=1, w = 0.9, c1 = 2., c2 = 2.)

    pso.interate(draw = 1)  # 迭代开始， draw = 1表示绘图
    '''

    def __init__(self, dim, fitness, num_particle, max_iter, p_low, p_up, v_low, v_high, w = 1., c1 = 2., c2 = 2.):
        self.w = w  # 惯性权值
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 群体学习因子
        self.D = dim  # 粒子维度  
        self.N = num_particle  # 粒子群规模，初始化种群个数  100
        self.max_iter = max_iter  # 最大迭代次数  50
        self.p_range = [p_low, p_up]  # 粒子位置的约束范围
        self.v_range = [v_low, v_high]  # 粒子速度的约束范围
        self.x = np.zeros((self.N, self.D))  # 粒子群的位置
        self.v = np.zeros((self.N, self.D))  # 粒子群的速度
        self.p_best = np.zeros((self.N, self.D))  # 每个粒子的最优位置
        self.g_best = np.zeros((1, self.D))[0]  # 种群（全局）的最优位置
        self.p_bestFit = np.zeros(self.N)  # 每个粒子的最优适应值
        self.g_bestFit = -float('Inf')  # 初始的全局最优适应度
        if not 'fitness' in self.__dir__(): # if not predefine within the class,
            self.fitness = fitness # set fitness function
        else:
            print('This class has already defined a fitness function. Skip override.')

    def init_swarm(self): 
        '''
        初始化所有个体和全局信息
        '''        
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(self.p_range[0][j], self.p_range[1][j])
                self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
            self.p_best[i] = self.x[i]  # 保存个体历史最优位置，初始默认第0代为最优
            fit = self.fitness(self.p_best[i])
            self.p_bestFit[i] = fit  # 保存个体历史最优适应值
            if fit > self.g_bestFit:  # 寻找并保存全局最优位置和适应值
                self.g_best = self.p_best[i]
                self.g_bestFit = fit
    
    def interate(self, draw = 1):  
        '''
        The main loop of PSO
        '''
        self.init_swarm()
        
        # 开始迭代
        best_fit = []  # 记录每轮迭代的最佳适应度，用于绘图
        w_range = None
        if isinstance(self.w, tuple):
            w_range = self.w[1] - self.w[0]
            self.w = self.w[1]
        time_start = time.time()  # 记录迭代寻优开始时间
        for i in range(self.max_iter):
            self.update()  # 更新主要参数和信息
            if w_range:
                self.w -= w_range / self.max_iter  # 惯性权重线性递减
            print("\rIter: {:d}/{:d} fitness: {:.4f} ".format(i, self.max_iter, self.g_bestFit, end = '\n'))
            best_fit.append(self.g_bestFit.copy())
        time_end = time.time()  # 记录迭代寻优结束时间
        print(f'Algorithm takes {time_end - time_start} seconds')  # 打印算法总运行时间，单位为秒/s
        if draw:
            plt.figure()
            plt.plot([i for i in range(self.max_iter)], best_fit)
            plt.xlabel("iter")
            plt.ylabel("fitness")
            plt.title("Iter process")
            plt.show()
    
    def update(self):
        for i in range(self.N):
 
            # 更新速度
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            
            # 速度限制
            for j in range(self.D):
                if self.v[i][j] < self.v_range[0]:
                    self.v[i][j] = self.v_range[0]
                if self.v[i][j] > self.v_range[1]:
                    self.v[i][j] = self.v_range[1]
 
            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
 
            # 位置限制
            for j in range(self.D):
                if self.x[i][j] < self.p_range[0][j]:
                    self.x[i][j] = self.p_range[0][j]
                if self.x[i][j] > self.p_range[1][j]:
                    self.x[i][j] = self.p_range[1][j]
 
            # 更新个体和全局历史最优位置及适应值
            _fit = self.fitness(self.x[i])
            if _fit > self.p_bestFit[i]:
                self.p_best[i] = copy.copy(self.x[i])
                self.p_bestFit[i] = _fit
            if _fit > self.g_bestFit:
                self.g_best = copy.copy(self.x[i])
                self.g_bestFit = _fit

class MKS_DiscretePSO(PSO):
    '''
    Discrete PSO for multi-kernel selection

    Example
    -------
    # 核函数列表及数据初始化
    KEYS = list(KX.keys())  # KX 是 31 个核函数转换的特征矩阵，dict类型
    n_kernels = len(KEYS)
    clf = 'LinearDiscriminantAnalysis()'
    PENALTY_FACTOR = 0.02  # 惩罚系数
    STOP_THRESHOLD = 0.99
    max_kernels_allowed = 10  # 限制核函数最大选择数量

    # 定义粒子的位置和速度范围
    low = [-10] * n_kernels
    up = [10] * n_kernels

    # 创建 DiscretePSO 对象
    pso = MKS_DiscretePSO(
        KEYS=KEYS, KX=KX, y=y, clf=clf,
        PENALTY_FACTOR=PENALTY_FACTOR, STOP_THRESHOLD=STOP_THRESHOLD, max_kernels_allowed=max_kernels_allowed,
        dim=n_kernels, fitness='dummy', num_particle=30, max_iter=100,
        p_low=low, p_up=up, v_low=-4, v_high=4, w=0.9, c1=1.5, c2=1.7
    )

    # 运行 PSO 算法
    pso.iterate(draw=True)

    # 输出选择的核函数
    selected_kernels = [KEYS[i] for i in range(n_kernels) if pso.g_best[i] == 1]
    print("Selected kernels:", selected_kernels)
    print(f"Best Adjusted Accuracy: {pso.g_bestFit:.4f}")
    '''

    # 将布尔向量转换为核函数选择，并限制选择数量
    def boolean_to_selection(self, position_array):
        # 通过 Sigmoid 映射为 [0, 1] 概率
        sigmoid_probs = 1 / (1 + np.exp(-position_array))
        binary_selection = (sigmoid_probs > np.random.rand(self.D)).astype(int)
        
        # 随机选择核函数数量（至少选择 2 个）
        num_kernels_to_select = np.random.randint(2, self.max_kernels_allowed + 1)
        
        if np.sum(binary_selection) > num_kernels_to_select:
            selected_indices = np.where(binary_selection == 1)[0]
            np.random.shuffle(selected_indices)
            binary_selection[:] = 0  # 重置
            binary_selection[selected_indices[:num_kernels_to_select]] = 1
            
        return binary_selection
    
    # 适应度函数（带惩罚项）
    def fitness(self, kernel_combination):  
        
        KEYS, KX, y, clf, PENALTY_FACTOR, max_kernels_allowed = self.KEYS, self.KX, self.y, self.clf, self.PENALTY_FACTOR, self.max_kernels_allowed
        
        combined = np.zeros((len(y), 0))
        selected_kernels = [KEYS[i] for i in range(len(kernel_combination)) if kernel_combination[i] == 1]
        num_selected_kernels = len(selected_kernels)

        # print(selected_kernels)
        
        if not selected_kernels:
            return 0.1  # 如果没有选择任何核函数，则返回很低的适应度
        
        # 核函数组合数据
        for kernel in selected_kernels:
            combined = np.hstack((combined, KX[kernel]))
        
        dics, _ = run_multiclass_clfs(combined, y, clfs=[clf], show=False)
        accuracy = dics[1][clf][0]
    
        # 惩罚项：超过 max_kernels_allowed 进行更强的惩罚
        penalty = PENALTY_FACTOR * max(0, num_selected_kernels - max_kernels_allowed)
        adjusted_accuracy = accuracy - penalty
        return adjusted_accuracy
        
    def __init__(self, KEYS, KX, y, clf, PENALTY_FACTOR, STOP_THRESHOLD, max_kernels_allowed, dim, fitness, num_particle, max_iter, p_low, p_up, v_low, v_high, w=1., c1=2., c2=2.):
        super().__init__(dim, fitness, num_particle, max_iter, p_low, p_up, v_low, v_high, w, c1, c2)
        self.KEYS = KEYS
        self.KX = KX
        self.y = y
        self.clf = clf
        self.PENALTY_FACTOR = PENALTY_FACTOR
        self.STOP_THRESHOLD = STOP_THRESHOLD
        self.max_kernels_allowed = max_kernels_allowed

    def init_swarm(self):
        # 初始化粒子的适应度
        for i in range(self.N):
            kernel_selection = self.boolean_to_selection(self.x[i])
            self.p_bestFit[i] = self.fitness(kernel_selection)
            if self.p_bestFit[i] > self.g_bestFit:
                self.g_bestFit = self.p_bestFit[i]
                self.g_best = self.x[i]
    
    def iterate(self, draw=1):

        self.fitness_history = []
        self.init_swarm()
        
        for iter_num in range(self.max_iter):
            print(f"\nIteration {iter_num + 1}/{self.max_iter}")
            for i in range(self.N):
                # 更新粒子的速度
                r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
                self.v[i] = (self.w * self.v[i] 
                            + self.c1 * r1 * (self.p_best[i] - self.x[i]) 
                            + self.c2 * r2 * (self.g_best - self.x[i]))
                
                # 更新粒子的位置
                self.x[i] = self.boolean_to_selection(self.v[i])

                # 计算新的适应度
                current_fitness = self.fitness(self.x[i])
                
                # 更新个体最优
                if current_fitness > self.p_bestFit[i]:
                    self.p_bestFit[i] = current_fitness
                    self.p_best[i] = self.x[i]
                
                # 更新全局最优
                if current_fitness > self.g_bestFit:
                    self.g_bestFit = current_fitness
                    self.g_best = self.x[i]

                # 输出当前粒子的核函数组合和准确率
                selected_kernels = [self.KEYS[j] for j in range(self.D) if self.x[i][j] == 1]
                print(f"Particle {i + 1}: Kernels: {selected_kernels}, Adjusted Accuracy: {current_fitness:.4f}")

            # 记录当前迭代的全局最优
            print(f"\nBest Fitness So Far: {self.g_bestFit:.4f}")
            self.fitness_history.append(self.g_bestFit)

            # 如果达到停止阈值，则提前结束
            if self.g_bestFit >= self.STOP_THRESHOLD:
                print("达到预期阈值，结束搜索")
                break

        # 绘制适应度变化图
        if draw:

            from matplotlib.ticker import MaxNLocator

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(range(1, len(self.fitness_history)+1), self.fitness_history)
            plt.xlabel("Iteration")
            plt.ylabel("Fitness")
            plt.title("fitness change over iterations")
            plt.show()