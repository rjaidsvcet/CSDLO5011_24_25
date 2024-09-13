import numpy as np

class LinearRegression:
    def __init__ (self):
        self.params = np.zeros(int(np.random.random()), float)[:, np.newaxis]

    def fit (self, X_train, y_train):
        bias = np.ones (len (X_train))
        X_b = np.c_[bias, X_train]
        # print (X_b)
        inner_part = np.transpose (X_b) @ X_b
        # print (inner_part)
        inverse = np.linalg.inv (inner_part)
        # print (inverse)
        X_part = inverse @ np.transpose (X_b)
        # print (X_part)
        lse = X_part @ y_train
        self.params = lse
        return self.params

    def predict (self, Xi):
        bias_test = np.ones (len (Xi))
        X_test = np.c_[bias_test, Xi]
        y_hat = X_test @ self.params
        return y_hat
        

if __name__ == '__main__':
    model = LinearRegression ()
    X = np.array ([
        [1, 4],
        [2, 5],
        [3, 8],
        [4, 2] 
    ])

    y = np.array ([1, 6, 8, 12])

    b_hat = model.fit (X, y)
    # print (b_hat)

    y_pred = model.predict ([[5, 3]])
    print (f'The prediction is : {y_pred}')