import numpy as np

class SingleLayer:
    def __init__(self, learning_rate=0.1, l1=0, l2=0):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses =[]
        self.lr = learning_rate
        self.w_history = []
        self.l1 = l1
        self.l2 = l2

    def forpass(self, x):
        z = np.dot(x, self.w) +self.b
        return z

    def activation(self, z):
        z = np.clip(z, a_min=-100, a_max = None)   
        a = 1/(1+np.exp(-z))
        return a

    def backprop(self, x, err):
        m = len(x)
        w_grad = np.dot(x.T,err)/m
        b_grad = np.sum(err)/m
        return w_grad, b_grad

    def fit(self, x, y, epochs = 100, x_val = None, y_val = None):
        self.w = np.ones((x.shape[1],1))
        self.b = 0
        y=y.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        m = len(x)
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        
        for epochs in range(epochs):
            z = self.forpass(x)
            a = self.activation(z)
            err = -(y-a)
            w_grad, b_grad = self.backprop(x, err)
            w_grad += (self.l1*np.sign(self.w)+self.l2*self. w)/m
            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad
            self.w_history.append(self.w.copy())
            a = np.clip(a, a_min=1e-10, a_max = 1-1e-10)
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss+self.reg_loss())/m) 
            self.update_val_loss(x_val, y_val)

    def reg_loss(self):
        return self.l1*np.sum(np.abs(self.w))+self.l2/2*np.sum(self.w**2)


    def update_val_loss(self, x_val, y_val):
        val_loss = 0
        z = self.forpass(x_val)
        a = self.activation(z)
        a = np.clip(a, 1e-10, 1-1e-10)
        val_loss += np.sum(-(y_val*np.log(a)+(1-y_val)*np.log(1-a)))
        self.val_losses.append((val_loss + self.reg_loss())/len(y_val))

    def predict(self, x):
        z = self.forpass(x)
        return z>0

    def score(self, x, y):
        return np.mean(self.predict(x)==y.reshape(-1,1))


class TripleLayer(SingleLayer):
     def __init__(self, units_1 = 10, units_2 = 5, learning_rate=0.1, l1=0, l2=0):
         super().__init__(learning_rate, l1, l2)
         self.units_1 = units_1
         self.units_2 = units_2
         self.w1 = None
         self.w2 = None
         self.w3 = None
         self.b1 = None
         self.b2 = None
         self.b3 = None
         self.a1 = None
         self.a2 = None
         self.losses = []
         self.val_losses = []
         self.l1 = l1
         self.l2 = l2

     def forpass(self, x):
         z1 = np.dot(x, self.w1) +self.b1
         self.a1 = self.activation(z1)
         z2 = np.dot(self.a1, self.w2) +self.b2
         self.a2 = self.activation(z2)
         z3 = np.dot(self.a2, self.w3) +self.b3
         return z3

     def backprop(self, x, err):
         m = len(x)
         w3_grad = np.dot(self.a2.T, err)/m
         b3_grad = np.sum(err)/m
         err_to_hidden_2 = np.dot(err, self.w3.T)*self.a2*(1-self.a2)
         w2_grad = np.dot(self.a1.T, err_to_hidden_2)/m
         b2_grad = np.sum(err_to_hidden_2)/m
         err_to_hidden_1 = np.dot(err_to_hidden_2, self.w2.T)*self.a1*(1-self.a1)
         w1_grad = np.dot(x.T, err_to_hidden_1)/m
         b1_grad = np.sum(err_to_hidden_1)/m
         return w3_grad, b3_grad, w2_grad, b2_grad, w1_grad, b1_grad

     def init_weights(self, n_features):
         self.w1 = np.ones((n_features, self.units_1))
         self.b1 = np.zeros(self.units_1)
         self.w2 = np.ones((self.units_1, self.units_2))
         self.b2 = np.zeros(self.units_2)
         self.w3 = np.ones((self.units_2,1))
         self.b3 = 0

     def fit(self, x, y, epochs=100, x_val=None, y_val=None):
         y = y.reshape(-1,1)
         y_val = y_val.reshape(-1,1)
         m = len(x)
         self.init_weights(x.shape[1])
         for i in range(epochs):
             a = self.training(x, y, m)
             a = np.clip(a, 1e-10, 1-1e-10)
             loss = np.sum(-(y*np.log(a)+(1-y)*np.log(1-a)))
             self.losses.append((loss + self.reg_loss())/m)
             self.update_val_loss(x_val, y_val)

     def training(self, x, y, m):
         z3 = self.forpass(x)
         a3 = self.activation(z3)
         err = -(y-a3)
         w3_grad, b3_grad, w2_grad, b2_grad, w1_grad, b1_grad = self.backprop(x, err)
         w1_grad += (self.l1*np.sign(self.w1) + self.l2*self.w1)/m
         w2_grad += (self.l1*np.sign(self.w2) + self.l2*self.w2)/m
         w3_grad += (self.l1*np.sign(self.w3) + self.l2*self.w3)/m
         self.w1 -= w1_grad
         self.w2 -= w2_grad
         self.w3 -= w3_grad
         self.b1 -= b1_grad
         self.b2 -= b2_grad
         self.b3 -= b3_grad

         return a3


     def reg_loss(self):
         return (self.l1*(np.sum(np.abs(self.w1))+np.sum(np.abs(self.w2))+np.sum(np.abs(self.w3)))
                 + self.l2/2*(np.sum(self.w1**2) + np.sum(self.w2**2)+ np.sum(self.w3**2)))



class RandomInitNetwork(TripleLayer):
     def init_weights(self, n_features):
         np.random.seed(42)
         self.w1 = np.random.normal(0, 1, (n_features, self.units_1))
         self.b1 = np.zeros(self.units_1)
         self.w2 = np.random.normal(0, 1, (self.units_1, self.units_2))
         self.b2 = np.zeros(self.units_2)
         self.w3 = np.random.normal(0, 1, (self.units_2, 1))
         self.b3 = 0

