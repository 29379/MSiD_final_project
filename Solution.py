from CustomModel import CustomModel, mapping_fun
from ReadingData import read_data, dataset
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,\
    r2_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np


X, Y = read_data(dataset)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=42)
y_range = len(Y_test)


"""
OptimizeWarning: Covariance of the parameters could not be estimated
"""
def custom_model_imp():
    params, _ = curve_fit(mapping_fun, xdata=X_train, ydata=Y_train,
                          p0=np.ones((len(X_train.columns)+1)))
    model = CustomModel(params)
    Y_predicted = model.run(X_test)

    ms_err = mean_squared_error(Y_test, Y_predicted)
    map_err = mean_absolute_percentage_error(Y_test, Y_predicted)
    r2sc = r2_score(Y_test, Y_predicted)
    """accuracy = accuracy_score(Y_test, Y_predicted)
    precision = precision_score(Y_test, Y_predicted)
    recall = recall_score(Y_test, Y_predicted)
    f1score = f1_score(Y_test, Y_predicted)
    matt_corr_coef = matthews_corrcoef(Y_test, Y_predicted)"""

    """output = Solution(Y_predicted, ms_err, map_err, r2sc,
                      accuracy, precision, recall, f1score,
                      matt_corr_coef)"""
    output = Solution(Y_predicted, ms_err, map_err, r2sc)
    return output


def svr_model_imp():
    model = svm.SVR()
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)

    ms_err = mean_squared_error(Y_test, Y_predicted)
    map_err = mean_absolute_percentage_error(Y_test, Y_predicted)
    r2sc = r2_score(Y_test, Y_predicted)
    """accuracy = accuracy_score(Y_test, Y_predicted)
    precision = precision_score(Y_test, Y_predicted)
    recall = recall_score(Y_test, Y_predicted)
    f1score = f1_score(Y_test, Y_predicted)
    matt_corr_coef = matthews_corrcoef(Y_test, Y_predicted)"""

    """output = Solution(Y_predicted, ms_err, map_err, r2sc,
                      accuracy, precision, recall, f1score,
                      matt_corr_coef)"""
    output = Solution(Y_predicted, ms_err, map_err, r2sc)
    return output


def linear_model_imp():
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)

    ms_err = mean_squared_error(Y_test, Y_predicted)
    map_err = mean_absolute_percentage_error(Y_test, Y_predicted)
    r2sc = r2_score(Y_test, Y_predicted)
    """accuracy = accuracy_score(Y_test, Y_predicted)
    precision = precision_score(Y_test, Y_predicted)
    recall = recall_score(Y_test, Y_predicted)
    f1score = f1_score(Y_test, Y_predicted)
    matt_corr_coef = matthews_corrcoef(Y_test, Y_predicted)"""

    """output = Solution(Y_predicted, ms_err, map_err, r2sc,
                      accuracy, precision, recall, f1score,
                      matt_corr_coef)"""
    output = Solution(Y_predicted, ms_err, map_err, r2sc)
    return output


class Solution:
    def __init__(self, *args):
        self.y_predicted = args[0]
        self.mean_sq_err = args[1]
        self.mean_ap_per_err = args[2]
        self.r2score = args[3]
        """self.accuracy = args[4]
        self.precision = args[5]
        self.recall = args[6]
        self.f1score = args[7]
        self.matt_corr_coef = args[8]"""
        """
        When I apply accuracy, ... etc to my model:
            ValueError: Can't handle mix of multiclass and continuous
        """
