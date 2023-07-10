import numpy as np
import matplotlib.pyplot as pt

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