tasks.

    - 随机梯度

    - 共轭梯度法，这个理论熵比梯度法快一些, CONJUGATE GRADIENT METHODS

    - petsc https://petsc.org/release/ 美国能源开源

    - 估计波动率(25 delta etc)

    - vba 与 丁萍姐任务
    
opt self learning.

- gradient descent.

    X 这个变数的定义?
    如何呈现资料视觉化?
    怎么改这个例子?
    [done.]是否要做一个总测试?各种的loss函数和模型.
        - loss [done.]
        - function [done.]
    [done.]建构各式各样的损失函数.
    了解rosen brock function.[loading.]
        - 怎么绘画出可视化的等高线.
    和优化方法针对每个函数做笔记整理成markdown.
    对于每个方法一个class例如gradient-descent一个,newton一个.    

- loss function

    梯度下降法可以应用于各种优化问题，并且可以使用不同的损失函数来衡量模型的拟合程度。以下是几种常见的损失函数及其对应的优缺点：

        均方误差（Mean Squared Error, MSE）：
        损失函数公式：MSE = (1/n) * Σ(y_pred - y)^2
        优点：
            对异常值不敏感，因为平方操作会放大较大的误差。
            可微分，方便使用梯度下降法进行优化。
            缺点：
            由于平方操作，MSE 对离群点比较敏感，可能导致模型过度拟合。
            可能存在多个局部最小值。

        平均绝对误差（Mean Absolute Error, MAE）：
        损失函数公式：MAE = (1/n) * Σ|y_pred - y|
        优点：
            相对于MSE，对异常值更加鲁棒。
            可微分，方便使用梯度下降法进行优化。
            缺点：
            对异常值仍然有一定的敏感性。
            不连续可导，可能导致优化问题。

        对数损失（Log Loss，也称为交叉熵损失）：
        适用于二分类或多分类问题。
        损失函数公式（二分类）：Log Loss = -(1/n) * Σ(y * log(y_pred) + (1-y) * log(1-y_pred))
        优点：
            对于分类问题，是一种常用的损失函数。
            梯度下降法对交叉熵损失函数有良好的效果。
            缺点：
            对于回归问题不适用。
            容易受到类别不平衡的影响。

        Hinge Loss（用于支持向量机）：
        适用于二分类问题，特别用于支持向量机（SVM）的线性分类器。
        损失函数公式：Hinge Loss = Σ(max(0, 1 - y * y_pred))
        优点：
            在支持向量机中具有几何解释。
            鼓励模型产生更大的间隔。
            缺点：
            不可微分，在优化过程中可能需要使用次梯度等方法。

    优点：

        梯度下降法是一种通用的优化算法，适用于各种优化问题。
        可以应用于大规模数据集和高维参数空间。
        在凸函数情况下，能够收敛到全局最优解。

    缺点：

        对于非凸函数，可能收敛到局部最优解。
        学习率的选择对算法的性能和收敛速度有很大影响，选择不当可能导致收敛缓慢甚至无法收敛。
        对于特征值病态条件较差的问题，可能收敛缓慢。

    请注意，损失函数的选择应根据具体问题和模型类型进行调整。不同的损失函数会导致不同的优化效果和模型性能。

    这个损失函数是均方误差（Mean Squared Error, MSE）加上L2正则化项。它在回归问题中常被使用。

    损失函数公式：
    loss = np.mean((y_pred - y) ** 2) + (regularization_param / (2 * num_samples)) * np.sum(theta ** 2)

    其中，(y_pred - y) ** 2 是均方误差部分，计算预测值与真实值之间的差异的平方，并求平均值。均方误差衡量模型预测值与真实值之间的差异，通过求平方可以放大较大的误差。

    (regularization_param / (2 * num_samples)) * np.sum(theta ** 2) 是L2正则化项，用于控制模型的复杂度。L2正则化项将模型参数theta的平方和乘以正则化参数，并除以2*num_samples进行归一化。正则化惩罚项有助于防止模型过度拟合，通过惩罚较大的参数值，促使模型选择较小的参数值。

    将均方误差和L2正则化项相加，即可得到最终的损失函数。优化这个损失函数可以通过梯度下降法来更新模型参数theta，以最小化损失函数，从而提高模型的拟合性能和泛化能力。

- 绘画等高线.

    ```
    import numpy as np
    import matplotlib.pyplot as plt

    # 定义目标函数（Rosenbrock 函数作为示例）
    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    # 定义目标函数的梯度
    def gradient_rosenbrock(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
        dy = 200 * (y - x ** 2)
        return dx, dy

    # 梯度下降优化算法
    def gradient_descent(x_start, y_start, learning_rate, num_iterations):
        x = x_start
        y = y_start
        trajectory = [(x, y)]
        
        for _ in range(num_iterations):
            dx, dy = gradient_rosenbrock(x, y)
            x -= learning_rate * dx
            y -= learning_rate * dy
            trajectory.append((x, y))
        
        return x, y, trajectory

    # 定义 x 和 y 的范围
    x_range = (-2, 2)
    y_range = (-1, 3)

    # 创建 x 和 y 的网格
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)

    # 计算目标函数在每个网格点上的值
    Z = rosenbrock(X, Y)

    # 进行梯度下降优化
    x_start = -1.5  # 设置初始点的 x 坐标
    y_start = 2.5  # 设置初始点的 y 坐标
    learning_rate = 0.001  # 设置学习率
    num_iterations = 100  # 设置迭代次数
    opt_x, opt_y, trajectory = gradient_descent(x_start, y_start, learning_rate, num_iterations)

    # 绘制等高线图
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 10))
    plt.colorbar()

    # 绘制优化结果的轨迹
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', color='red', label='Optimization Trajectory')

    # 标记最优点
    plt.scatter(opt_x, opt_y, color='red', label='Optimal Point')

    # 添加轮廓线标签和标题
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Result on Rosenbrock Function')

    # 显示图形
    plt.legend()
    plt.show()
    ```

-  接下去.