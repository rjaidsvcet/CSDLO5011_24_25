import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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
    
    def sumOfSquaredErrors (self, y_true, y_pred, error=0):
        for _ in range (len (y_true)):
            error += (y_true[_] - y_pred[_])**2
        return error
    
    def sumOfSquaredTotals (self, y_true, total=0):
        y_mean = np.mean (y_true)
        for _ in  range (len (y_true)):
            total += (y_true[_] - y_mean)**2
        return total
    
    def standardError (self, X, y_true, y_pred, error=0):
        for _ in range (len (y_true)):
            error += (y_true[_] - y_pred[_])**2
        stdErr = np.sqrt (error / (X.shape[0]-X.shape[1]-1))
        return stdErr

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

    y_pred = model.predict (X)
    print (f'The prediction is : {y_pred}')

    sse = model.sumOfSquaredErrors (y_true=y, y_pred=y_pred)
    print (f'The sum of squared errors is : {sse}')

    sst = model.sumOfSquaredTotals (y_true=y)
    print (f'The sum of squared total is : {sst}')

    r_squared = 1 - (sse/sst)
    print (f'Calculated R-Squared : {r_squared}')

    r2 = r2_score (y_true=y, y_pred=y_pred)
    print (f'Sklearn R2 : {r2}')

    adjR2 = 1 - ((sse/(X.shape[0]-X.shape[1]-1))/(sst/(X.shape[0]-1)))
    print (f'The adjusted R2 : {adjR2}')

    mse = sse / (X.shape[0]-X.shape[1]-1)
    print (f'The calculated MSE : {mse}')

    ssr = sst - sse
    print (f'SSR : {ssr}')
    msr = ssr
    f_c = msr / mse
    print (f'The F-Ratio Value : {f_c}')

    standardError = model.standardError (X, y, y_pred)
    print (f'Standard Error : {standardError}')

    # skmse = mean_squared_error (y_true=y, y_pred=y_pred)
    # print (f'Sklearn MSE : {skmse}')
