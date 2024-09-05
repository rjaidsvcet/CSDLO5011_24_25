import numpy as np
from sklearn.metrics import r2_score

class LinearRegression:
    def __init__ (self):
        self.a = 0
        self.b = 0

    def fit (self, X, y):
        X_mean = np.mean (X)
        y_mean = np.mean (y)
        ssxy, ssx = 0, 0
        for _ in range (len (X)):
            ssxy += (X[_]-X_mean)*(y[_]-y_mean)
            ssx += (X[_]-X_mean)**2

        self.b = ssxy / ssx
        self.a = y_mean - (self.b * X_mean)
        return self.a, self.b
        
    def predict (self, X):
        y_hat = self.a + (self.b * X)
        return y_hat
    
if __name__ == '__main__':
    X = np.array ([
        [42], [34], [25], [35], [37],
        [38], [31], [33], [19], [29],
        [38], [28], [29], [36], [18]
    ])

    y = np.array ([
        18, 6, 0, -1, 13,
        14, 7, 7, -9, 8,
        8, 5, 3, 14, -7
    ])

    model = LinearRegression ()
    a, b = model.fit (X, y)
    print (f'Value for a : {a} and value for {b}')

    y_pred = model.predict (X)
    r_sqaured = r2_score (y, y_pred)
    print (f'The Goodness of Fit for Regression is {r_sqaured}')

    # y_pred = model.predict ([[30]])
    # print (f'The predicted value is {y_pred}')