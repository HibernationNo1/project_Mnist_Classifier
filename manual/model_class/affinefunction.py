import numpy as np

class Flatten:
    def __init__(self, x):
        self._re = x.shape[1]*x.shape[2]*x.shape[3]
        self._x = x.reshape([-1, self._re])
        return x

class AffineFunction:
    def __init__(self, feature_dim, units):
        self._feature_dim = feature_dim
        self.units = units

        self._z1_list = [None]*(self._feature_dim + 1)
        self._z2_list = self._z1_list.copy()

        

        self.node_imp()
        self.random_theta_initialization()

    def random_theta_initialization(self):
        r_feature_dim = 1/np.power(self._feature_dim, 0.5)
        self._Th = np.random.uniform(low = -1*r_feature_dim, 
                                    high = r_feature_dim, 
                                    size =(self._feature_dim + 1, 1))
        # 초기 ramdom theta의 범위 = 1/np.power(self._feature_mid, 0.5)

    def node_imp(self):
        self._node1 = [None] + [self.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [self.plus_node() for _ in range(self._feature_dim)]

    def plus_node(self, x, y):
        self._x, self._y = x, y
        self._z = self._x + self._y
        return self._z

    def mul_node(self, x, y):
        self._x, self._y = x, y
        self._z = self._x*self._y
        return self._z 

    def perseptron(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._z1_list[node_idx] = self._node1[node_idx](self._Th[node_idx], 
                                                                    X[node_idx]) 
        # 곱셈                                                              
        self._z2_list[1] = self._node2[1](self._Th[0], self._z1_list[1]) # bias
        for node_idx in range(2, self._feature_dim + 1): 
            self._z2_list[node_idx] = self._node2[node_idx](self._z2_list[node_idx-1], 
                                                                    self._z1_list[node_idx])
        # 덧셈
        return self._z2_list[-1]     


    def layer(self, X):
        for persep_idx in range(self.units):
            self.persep_list[persep_idx] = self.perseptron(X)
        return self.persep_list
            