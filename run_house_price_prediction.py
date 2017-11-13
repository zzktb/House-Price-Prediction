from process_data import process_data
from visualize_results import visualize_results
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np


# process data
X_train, y_train, X_test, y_test, y_test_norm, label = process_data('train.csv', 1000)


# fit linear regression object
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


# test use all the 10 features
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test * y_test_norm, y_pred * y_test_norm)
print("RMSE of using all the features: \t%.4f" % np.sqrt(mse))
print("y_test_norm: \n\t%.4f" % y_test_norm)


# test use only 9 features
mse0 = np.zeros((1, X_train.shape[1]))
for i in range(X_train.shape[1]):
    idx = np.arange(X_train.shape[1])
    idx0 = np.delete(idx, i)
    X_train0 = X_train[:, idx0]
    X_test0 = X_test[:, idx0]
    # fit linear regression object
    regr0 = linear_model.LinearRegression()
    regr0.fit(X_train0, y_train)
    y_pred0 = regr0.predict(X_test0)
    mse0[0, i] = mean_squared_error(y_test, y_pred0)
    print("RMSE without feature %d: %.4f" % (i, np.sqrt(mse0[0, i])))


# find the best result
f_id = np.argmin(mse0)
print("Minimum RMSE: %.4f" % np.sqrt(mse0[0, f_id]), " when Feature \"", label[f_id], "\" is removed.")


# Visualize results
idx = np.arange(X_train.shape[1])
index = np.delete(idx, f_id)
X_train1 = X_train[:, index]
X_test1 = X_test[:, index]
best_regr = linear_model.LinearRegression()
best_regr.fit(X_train1, y_train)
visualize_results(1, False, y_train, best_regr.predict(X_train1), y_test, best_regr.predict(X_test1))
visualize_results(2, False, y_train[:50], (best_regr.predict(X_train1))[:50], y_test[:50], (best_regr.predict(X_test1))[:50])

