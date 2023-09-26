import numpy as np


class ELM():
    # 定义输入数据集X、标签Y、神经元个数m、控制参数L、训练函数TRAIN_beta
    def __init__(self, X, Y, m, L):
        self.X = X
        self.Y = Y
        self.m = m
        self.L = L
        self.TRAIN_beta()

    def sigmoid(self, x): #使用S函数做特征映射，将输入层数据由原来空间映射到ELM的特征空间
        return 1.0 / (1 + np.exp(-x))

    # 定义训练函数，随机w,b 计算输出矩阵H、输出权重beta、F1输出函数
    def TRAIN_beta(self):
        n, d = self.X.shape#X.shape表示返回维数，返回的是一个元组； 例如y.shape[0]代表行数，y.shape[1]代表列数。
        self.w = np.random.rand(d, self.m)#随机初始化
        self.b = np.random.rand(1, self.m)
        H = self.sigmoid(np.dot(self.X, self.w) + self.b)#使用特征映射求解输出矩阵
        self.beta = np.dot(np.linalg.inv(np.identity(self.m) / self.L + np.dot(H.T, H)),#计算输出权重
                           np.dot(H.T, self.Y))  # 加入正则化且 n >> m
        #F1 = H.dot(beta) #计算输出函数
        # self.beta = np.dot(np.linalg.pinv(H), self.Y) # 不加入正则化


    def TEST(self, x):
        H = self.sigmoid(np.dot(x, self.w) + self.b)  # 使用测试集计算H，其中w、b是之前随机得到的
        result = np.dot(H, self.beta) #得到测试函数
        return result
