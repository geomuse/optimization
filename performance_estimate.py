from generate_example import generate_sample

class performance :
    def performance_at_all(self,opt):
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

    def performance_for_one_method(self,met,opt):
        r'''
        met -> int
            1 : batch_gradient_descent
            2 : batch_gradient_descent_regularization
            3 : stochastic_gradient_descent
            4 : mini_batch_gradient_descent
        '''
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