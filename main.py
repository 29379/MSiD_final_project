from Solution import Solution, custom_model_imp,\
    linear_model_imp, svr_model_imp, y_range, Y_test, Y_test
import seaborn
from matplotlib import pyplot
import time


def plot():
    linear_solution = linear_model_imp()
    svr_solution = svr_model_imp()
    custom_solution = custom_model_imp()
    print("\nCustom model solution:\n\t*\tMean squared error - "
          + str(custom_solution.mean_sq_err) + "\n\t*\tMean absolute percentage error - "
          + str(custom_solution.mean_ap_per_err) + "\n\t*\tR2 score - "
          + str(custom_solution.r2score) + "\n")
    """+ "\n\t*\tAccuracy - "
          + str(custom_solution.accuracy) + "\n\t*\tPrecision - "
          + str(custom_solution.precision) + "\n\t*\tRecall - "
          + str(custom_solution.recall) + "\n\t*\tF1 Score - "
          + str(custom_solution.f1score) + "\n\t*\tMatthews correlation coefficient - "
          + str(custom_solution.matt_corr_coef) + "\n")"""

    print("\nLinear solution:\n\t*\tMean squared error - "
          + str(linear_solution.mean_sq_err) + "\n\t*\tMean absolute percentage error - "
          + str(linear_solution.mean_ap_per_err) + "\n\t*\tR2 score - "
          + str(linear_solution.r2score) + "\n")
    """\n\t*\tAccuracy - "
          + str(linear_solution.accuracy) + "\n\t*\tPrecision - "
          + str(linear_solution.precision) + "\n\t*\tRecall - "
          + str(linear_solution.recall) + "\n\t*\tF1 Score - "
          + str(linear_solution.f1score) + "\n\t*\tMatthews correlation coefficient - "
          + str(linear_solution.matt_corr_coef) + "\n")"""

    print("\nSVR solution:\n\t*\tMean squared error - "
          + str(svr_solution.mean_sq_err) + "\n\t*\tMean absolute percentage error - "
          + str(svr_solution.mean_ap_per_err) + "\n\t*\tR2 score - "
          + str(svr_solution.r2score) + "\n")
    """\n\t*\tAccuracy - "
          + str(svr_solution.accuracy) + "\n\t*\tPrecision - "
          + str(svr_solution.precision) + "\n\t*\tRecall - "
          + str(svr_solution.recall) + "\n\t*\tF1 Score - "
          + str(svr_solution.f1score) + "\n\t*\tMatthews correlation coefficient - "
          + str(svr_solution.matt_corr_coef) + "\n")"""

    seaborn.scatterplot(x=range(0, y_range), y=Y_test, color='purple')
    seaborn.scatterplot(x=range(0, y_range), y=custom_solution.y_predicted, color='blue')
    seaborn.scatterplot(x=range(0, y_range), y=linear_solution.y_predicted, color='red')
    seaborn.scatterplot(x=range(0, y_range), y=svr_solution.y_predicted, color='green')
    pyplot.show()


if __name__ == '__main__':
    start = time.time()
    print(start)
    plot()
    print("\n--- %s seconds ---" % (time.time() - start))
