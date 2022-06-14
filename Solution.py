from CustomModel import CustomModel, mapping_fun
from ReadingData import read_data, dataset
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, \
    r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np


X, Y = read_data(dataset)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
y_range = len(Y_test)


def custom_model_imp():
    params, _ = curve_fit(mapping_fun, xdata=X_train, ydata=Y_train.ravel(),
                          p0=np.ones((len(X_train.columns) + 1)))
    model = CustomModel(params)
    Y_predicted = model.run(X_test)

    ms_err = mean_squared_error(Y_test, Y_predicted)
    map_err = mean_absolute_percentage_error(Y_test, Y_predicted)
    r2sc = r2_score(Y_test, Y_predicted)

    output = Solution(Y_predicted, ms_err, map_err, r2sc)
    return output


def svr_model_imp():
    model = svm.SVR()
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)

    ms_err = mean_squared_error(Y_test, Y_predicted)
    map_err = mean_absolute_percentage_error(Y_test, Y_predicted)
    r2sc = r2_score(Y_test, Y_predicted)

    output = Solution(Y_predicted, ms_err, map_err, r2sc)
    return output


def linear_model_imp():
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)

    ms_err = mean_squared_error(Y_test, Y_predicted)
    map_err = mean_absolute_percentage_error(Y_test, Y_predicted)
    r2sc = r2_score(Y_test, Y_predicted)

    output = Solution(Y_predicted, ms_err, map_err, r2sc)
    return output


class Solution:
    def __init__(self, *args):
        self.y_predicted = args[0]
        self.mean_sq_err = args[1]
        self.mean_ap_per_err = args[2]
        self.r2score = args[3]
