import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, eta0=0.001, max_iterations=10000):
        self.theta_ = np.zeros(shape=(1,))
        self.eta = eta0
        self.max_iterations = max_iterations
    
    def fit(self, X, y):
        X = np.array(X)
        X = X.reshape(X.shape[0], 1)
        y = np.array(y)
        m = X.shape[0]
        X_b = np.concatenate((np.ones((m, 1)), X), axis=1)
        self.theta_ = np.zeros(shape=X_b.shape[1])
        for i in range(self.max_iterations):
            gradients = 2/m * X_b.T.dot(X_b.dot(self.theta_) - y)
            self.theta_ = self.theta_ - self.eta * gradients
        return self
    
    def predict(self, X):
        X = np.array(X)
        X = X.reshape(X.shape[0], 1)
        m = X.shape[0]
        X_b = np.concatenate((np.ones((m, 1)), X), axis=1)
        return X_b.dot(self.theta_)
    
    def get_params(self, deep=True):
        result = {}
        for i in range(self.theta_.shape[0]):
            name = 'theta_'+str(i)
            result[name] = self.theta_[i]
        return result
        
fig1 = plt.figure(1)
x_points = [1,1,2,3,4,5,6,7,8,9,10,11]
y_points = [1,2,3,1,4,5,6,4,7,10,15,9]

model = LinearRegression()
model.fit(x_points, y_points)

plt.scatter(x_points, y_points)
plt.scatter(x_points, model.predict(x_points))
plt.show()
print(model.get_params())

fig2 = plt.figure(2)
x = np.linspace(-50, 50, 100) 
y = np.random.normal(loc=0.0, scale=10.0, size=x.shape)
y = y + x
model2 = LinearRegression(eta0=0.001, max_iterations=10000)
model2.fit(x, y)

plt.scatter(x, y)
plt.scatter(x, model.predict(x))
plt.show()
print(model2.get_params())



