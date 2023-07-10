import numpy as np
from loguru import logger
from loss_function import classic_loss_function , regularization_loss_function
import os , sys
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir)
sys.path.append(path)
log = os.path.join(current_dir,'log/error.log')

from performance_estimate import performance
logger.add(log,level='INFO', rotation='10 MB', format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}')

from scipy.optimize import linprog

if __name__ == '__main__':
    print('..')

    # 定义目标函数的系数
    c = np.array([-1, -2])

    # 定义不等式约束条件的系数矩阵和右侧向量
    A = np.array([[1, 1], [1, -1]])
    b = np.array([5, 1])

    # 定义变量的上下界
    x_bounds = (0, None)
    y_bounds = (0, None)

    # 构建线性规划模型
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds])

    # 提取结果
    optimal_solution = res.x
    optimal_value = res.fun

    print("最优解：", optimal_solution)
    print("目标函数值：", optimal_value)