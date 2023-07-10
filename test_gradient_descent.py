import numpy as np
import matplotlib.pyplot as pt
from loguru import logger
from abc import ABC , abstractmethod
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
log = os.path.join(current_dir,'log/error.log')
logger.add(log,level='INFO', rotation='10 MB', format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}')

class generate_loss_function(ABC):
    def parameters_init(self,y,y_pred):
        self.y,self.y_pred = y,y_pred

    @abstractmethod
    def mean_squared_error(self):
        ...

    @abstractmethod
    def mean_absolute_error(self):
        ...

    @abstractmethod
    def log_loss(self):
        ...
        
    @abstractmethod    
    def hinge_loss(self):
        ...

class classic_loss_function(generate_loss_function):
    def __init__(self) -> None:
        super().__init__()
        # print('loading classic.')

    def mean_squared_error(self):
        return np.mean(np.square(self.y_pred - self.y))

    def mean_absolute_error(self):
        return np.mean(np.abs(self.y_pred - self.y))

    def log_loss(self):
        return np.mean(self.y * np.log(self.y_pred) + (1-self.y) * np.log(1-self.y_pred))
    
    def hinge_loss(self):
        return np.sum(np.max(0,1-self.y*self.y_pred))    

class regularization_loss_function(classic_loss_function):
    def __init__(self) -> None:
        super().__init__()
        # print('loading regularization.')

    def parameters_init(self,y,y_pred,num_samples,theta,regularization_param):
        super().parameters_init(y,y_pred)
        self.num_samples,self.theta,self.regularization_param = num_samples,theta,regularization_param
    
    def regularization(self):
        return (self.regularization_param / self.num_samples) * np.sum(np.square(self.theta))

    def mean_squared_error(self):
        return super().mean_squared_error() +\
            self.regularization()

    def mean_absolute_error(self):
        return super().mean_absolute_error() +\
            self.regularization()

    def log_loss(self):
        return super().log_loss() +\
            self.regularization()

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

class generate_sample:
    def array(self):
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        z = np.array([2, 3, 4, 5])
        return X , z
    
    def rosenbrock(self,size):
        X = np.array([np.linspace(-2, 2, size),np.linspace(-1, 3, size)]).T
        z = lambda x,y : np.square(1-x) + 100*np.square(y-np.square(x))
        return X , z
    
    def non_linear_func_example(self,size):
        X = np.array([np.linspace(-2, 2, size),np.linspace(-1, 3, size)]).T
        z = lambda x,y : np.sin(0.5*np.square(x)-0.25*np.square(y)+3) * np.cos(2*x+1-np.exp(y))
        return X , z

class performance :
    def performance_at_all(self):
        opt = gradient_descent()
        learning_rate = 0.1
        num_iterations = 10000
        regularization_param = 0.1
        X , z = generate_sample().array()
        theta , count = opt.batch_gradient_descent(X, z, learning_rate, num_iterations, regularization_param)
        print(f"模型参数 theta: {theta} 共用了 {count} 次.")
        theta , count = opt.batch_gradient_descent_regularization(X, z, learning_rate, num_iterations, regularization_param)
        print(f"模型参数 theta: {theta} 共用了 {count} 次.")
        theta , count = opt.stochastic_gradient_descent(X, z, learning_rate, num_iterations, regularization_param)
        print(f"模型参数 theta: {theta} 共用了 {count} 次.")
        theta , count = opt.mini_batch_gradient_descent(X, z, learning_rate, num_iterations, regularization_param,batch_size=1)
        print(f"模型参数 theta: {theta} 共用了 {count} 次.")

    def performance_for_one_method(self,met):
        r'''
        met -> int
            1 : batch_gradient_descent
            2 : batch_gradient_descent_regularization
            3 : stochastic_gradient_descent
            4 : mini_batch_gradient_descent
        '''
        opt = gradient_descent()
        learning_rate = 0.1
        num_iterations = 10000
        regularization_param = 0.1
        X , z = generate_sample().array()
        function = {
            1 : opt.batch_gradient_descent(X, z, learning_rate, num_iterations, regularization_param) ,
            2 : opt.batch_gradient_descent_regularization(X, z, learning_rate, num_iterations, regularization_param) ,
            3 : opt.stochastic_gradient_descent(X, z, learning_rate, num_iterations, regularization_param) ,
            4 : opt.mini_batch_gradient_descent(X, z, learning_rate, num_iterations, regularization_param,batch_size=1)
        }
        theta , count = function.get(met)
        print(f"模型参数 theta: {theta} 共用了 {count} 次.")
        return theta , count 

def plot():
    X , z = generate_sample().rosenbrock(50)
    x , y  = np.meshgrid(X[:,0],X[:,1])
    z = z(x,y)
    pt.contour(x,y,z,levels=np.logspace(-1, 3, 10))
    pt.xlabel('x')
    pt.ylabel('y')
    pt.title('Rosenbrock Function.')
    pt.colorbar()
    pt.show()

if __name__ == '__main__':
    perf = performance()
    perf.performance_for_one_method(2)
    # plot()
    ...