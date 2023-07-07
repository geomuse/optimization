import numpy as np
import matplotlib.pyplot as pt

class generate_loss_function:
    def __init__(self) -> None:
        ... 
    def mean_squared_error(self,y,y_pred,num_samples,theta,regularization_param):
        return np.mean(np.square(y_pred - y)) + (regularization_param / (2 * num_samples)) * np.sum(np.square(theta))


class gradient_descent:
    gls = generate_loss_function()

    def batch_gradient_descent(self,X, y, learning_rate, num_iterations, regularization_param):
        num_samples, num_features = X.shape
        theta = np.random.randn(num_features) * 0.01
        prev_loss = float('inf')  # 上一次迭代的损失函数值
        count = 0
        for i in range(num_iterations):
            # 计算模型预测值
            y_pred = np.dot(X, theta)
            # 计算误差
            error = y_pred - y
            # 计算梯度
            gradient = np.dot(X.T, error) / num_samples
            # 添加正则化项
            gradient += (regularization_param / num_samples) * theta
            # 更新参数
            theta -= learning_rate * gradient
            # 计算当前损失函数值
            loss = self.gls.mean_squared_error(y,y_pred,num_samples,theta,regularization_param)
            # 判断收敛条件
            if np.abs(loss - prev_loss) < 1e-6:
                break
            prev_loss = loss
            count +=1
        return theta , count

def gradient_check():
    ...


'''
X 这个变数的定义?
如何呈现资料视觉化?
怎么改这个例子?
[loading]是否要做一个总测试?各种的loss函数和模型.
'''

if __name__ == '__main__':
    opt = gradient_descent()
    # 示例数据
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    z = np.array([2, 3, 4, 5])
    # 设置学习率、迭代次数和正则化参数
    learning_rate = 0.1
    num_iterations = 1000
    regularization_param = 0.1

    # 调用梯度下降法函数进行模型训练
    theta , count = opt.batch_gradient_descent(X, z, learning_rate, num_iterations, regularization_param)

    print(f"模型参数 theta: {theta} 共用了 {count} 次.")
    