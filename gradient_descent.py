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

class gradient_descent:
    clf = classic_loss_function()
    rlf = regularization_loss_function()
    tol = 1e-6
    # def parameters_init()
    def batch_gradient_descent(self,X, z, learning_rate, num_iterations, regularization_param=None):
        num_samples, num_features = X.shape
        theta = np.random.randn(num_features) * 0.01
        prev_loss = float('inf')  # 上一次迭代的损失函数值
        for epoch in range(num_iterations):
            #计算模型预测值
            z_pred = np.dot(X, theta)
            # 计算误差
            error = z_pred - z
            # 计算梯度
            gradient = np.dot(X.T, error) / num_samples
            # 更新参数
            theta -= learning_rate * gradient
            self.clf.parameters_init(z,z_pred)
            loss = self.clf.mean_squared_error()
            if np.abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
        return theta , epoch
    
    def batch_gradient_descent_regularization(self,X, z, learning_rate, num_iterations, regularization_param):
        num_samples, num_features = X.shape
        theta = np.random.randn(num_features) * 0.01
        prev_loss = float('inf')  # 上一次迭代的损失函数值
        
        for epoch in range(num_iterations):
            # 计算模型预测值
            z_pred = np.dot(X, theta)
            # 计算误差
            error = z_pred - z
            # 计算梯度
            gradient = np.dot(X.T, error) / num_samples
            # 添加正则化项
            gradient += (regularization_param / num_samples) * theta
            # 更新参数
            theta -= learning_rate * gradient
            # 计算当前损失函数值
            self.rlf.parameters_init(z,z_pred,num_samples,theta,regularization_param)
            loss = self.rlf.mean_squared_error()
            # 判断收敛条件
            if np.abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
        return theta , epoch

    def stochastic_gradient_descent(self,X, z, learning_rate, num_iterations, regularization_param):
        num_samples, num_features = X.shape
        theta = np.random.randn(num_features) * 0.01
        prev_loss = float('inf')  # 上一次迭代的损失函数值
        for epoch in range(num_iterations):
            for _ in range(num_samples):
                random_index = np.random.randint(num_samples)  # 随机选择一个样本
                xi = X[random_index]
                zi = z[random_index]
                zi_pred = np.dot(xi,theta)
                gradient = 2 * xi * (xi.dot(theta) - zi)  # 计算梯度
                theta -= learning_rate * gradient  # 更新参数
                self.rlf.parameters_init(zi,zi_pred,num_samples,theta,regularization_param)
                loss = self.rlf.mean_squared_error()
            # 判断收敛条件
            if np.abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
        return theta , epoch

    def mini_batch_gradient_descent(self,X, z, learning_rate, num_iterations, regularization_param,batch_size=2):
        num_samples, num_features = X.shape
        theta = np.random.randn(num_features) * 0.01
        prev_loss = float('inf')  # 上一次迭代的损失函数值
        for epoch in range(num_iterations):
            permutation = np.random.permutation(num_samples)  # 随机选择一个样本
            X_shuffled = X[permutation]
            z_shuffled = z[permutation]
            for _ in range(0, num_samples, batch_size):
                # 获取小批量样本
                X_batch = X_shuffled[_:_+batch_size]
                z_batch = z_shuffled[_:_+batch_size]
                z_batch_pred = np.dot(X_batch,theta)
                # 计算梯度
                gradient = np.mean(2 * X_batch * (X_batch.dot(theta) - z_batch), axis=0)
                # 计算正则化项
                regularization_term = regularization_param * theta
                # 更新参数
                theta -= learning_rate * (gradient + regularization_term)
                self.rlf.parameters_init(z_batch,z_batch_pred,num_samples,theta,regularization_param)
                loss = self.rlf.mean_squared_error()
            # 判断收敛条件
            if np.abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
        return theta , epoch

    def gradient_check(self):
        ...

if __name__ == '__main__':
    perf = performance()
    opt = gradient_descent()
    perf.performance_for_one_method(4,opt)