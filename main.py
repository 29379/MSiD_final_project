from Solution import Solution, custom_model_imp, \
    linear_model_imp, svr_model_imp, y_range, Y_test, X_test
import seaborn
from matplotlib import pyplot


def show_error_values(custom, linear, svr):
    print("\nCustom model solution:\n\t*\tMean squared error - "
          + str(custom.mean_sq_err) + "\n\t*\tMean absolute percentage error - "
          + str(custom.mean_ap_per_err) + "\n\t*\tR2 score - "
          + str(custom.r2score) + "\n")
    """+ "\n\t*\tAccuracy - "
          + str(custom_solution.accuracy) + "\n\t*\tPrecision - "
          + str(custom_solution.precision) + "\n\t*\tRecall - "
          + str(custom_solution.recall) + "\n\t*\tF1 Score - "
          + str(custom_solution.f1score) + "\n\t*\tMatthews correlation coefficient - "
          + str(custom_solution.matt_corr_coef) + "\n")"""

    print("\nLinear solution:\n\t*\tMean squared error - "
          + str(linear.mean_sq_err) + "\n\t*\tMean absolute percentage error - "
          + str(linear.mean_ap_per_err) + "\n\t*\tR2 score - "
          + str(linear.r2score) + "\n")
    """\n\t*\tAccuracy - "
          + str(linear_solution.accuracy) + "\n\t*\tPrecision - "
          + str(linear_solution.precision) + "\n\t*\tRecall - "
          + str(linear_solution.recall) + "\n\t*\tF1 Score - "
          + str(linear_solution.f1score) + "\n\t*\tMatthews correlation coefficient - "
          + str(linear_solution.matt_corr_coef) + "\n")"""

    print("\nSVR solution:\n\t*\tMean squared error - "
          + str(svr.mean_sq_err) + "\n\t*\tMean absolute percentage error - "
          + str(svr.mean_ap_per_err) + "\n\t*\tR2 score - "
          + str(svr.r2score) + "\n")
    """\n\t*\tAccuracy - "
          + str(svr_solution.accuracy) + "\n\t*\tPrecision - "
          + str(svr_solution.precision) + "\n\t*\tRecall - "
          + str(svr_solution.recall) + "\n\t*\tF1 Score - "
          + str(svr_solution.f1score) + "\n\t*\tMatthews correlation coefficient - "
          + str(svr_solution.matt_corr_coef) + "\n")"""


def plot(custom, linear, svr):
    fig, axes = pyplot.subplots(2, 2)
    fig.set_figheight(16)
    fig.set_figwidth(16)
    axes[0, 0].set_title('Real data')
    axes[0, 0].set_ylabel('Time difference')
    axes[0, 1].set_title('Custom model')
    axes[0, 1].set_ylabel('Time difference')
    axes[1, 0].set_title('Linear model')
    axes[1, 0].set_ylabel('Time difference')
    axes[1, 1].set_title('SVR model')
    axes[1, 1].set_ylabel('Time difference')

    seaborn.scatterplot(x=range(0, y_range), y=Y_test, color='purple', ax=axes[0, 0])
    seaborn.scatterplot(x=range(0, y_range), y=custom.y_predicted, color='blue', ax=axes[0, 1])
    seaborn.scatterplot(x=range(0, y_range), y=linear.y_predicted, color='red', ax=axes[1, 0])
    seaborn.scatterplot(x=range(0, y_range), y=svr.y_predicted, color='green', ax=axes[1, 1])

    pyplot.show()


def plot_district_details(custom, linear, svr):
    fig, axes = pyplot.subplots(3, 4)
    fig.set_figheight(16)
    fig.set_figwidth(16)
    axes[0, 0].set_title("Real data")
    axes[0, 0].set_xlabel("Police district - 'Bayview'")
    axes[0, 0].set_ylabel('Time difference')
    axes[0, 1].set_title("Custom model")
    axes[0, 1].set_xlabel("Police district - 'Bayview'")
    axes[0, 1].set_ylabel('Time difference')
    axes[0, 2].set_title("Linear model")
    axes[0, 2].set_xlabel("Police district - 'Bayview'")
    axes[0, 2].set_ylabel('Time difference')
    axes[0, 3].set_title("SVR model")
    axes[0, 3].set_xlabel("Police district - 'Bayview'")
    axes[0, 3].set_ylabel('Time difference')

    axes[1, 0].set_xlabel("Police district - 'Out of SF'")
    axes[1, 0].set_ylabel('Time difference')
    axes[1, 1].set_xlabel("Police district - 'Out of SF'")
    axes[1, 1].set_ylabel('Time difference')
    axes[1, 2].set_xlabel("Police district - 'Out of SF'")
    axes[1, 2].set_ylabel('Time difference')
    axes[1, 3].set_xlabel("Police district - 'Out of SF'")
    axes[1, 3].set_ylabel('Time difference')

    axes[2, 0].set_xlabel("Police district - 'Southern'")
    axes[2, 0].set_ylabel('Time difference')
    axes[2, 1].set_xlabel("Police district - 'Southern'")
    axes[2, 1].set_ylabel('Time difference')
    axes[2, 2].set_xlabel("Police district - 'Southern'")
    axes[2, 2].set_ylabel('Time difference')
    axes[2, 3].set_xlabel("Police district - 'Southern'")
    axes[2, 3].set_ylabel('Time difference')

    seaborn.set(rc={'figure.figsize': (3, 3)})

    seaborn.boxplot(x=X_test['PoliceDistrict_Bayview'], y=Y_test, color='purple', ax=axes[0, 0])
    seaborn.boxplot(x=X_test['PoliceDistrict_Bayview'], y=custom.y_predicted, color='blue', ax=axes[0, 1])
    seaborn.boxplot(x=X_test['PoliceDistrict_Bayview'], y=linear.y_predicted, color='red', ax=axes[0, 2])
    seaborn.boxplot(x=X_test['PoliceDistrict_Bayview'], y=svr.y_predicted, color='green', ax=axes[0, 3])

    seaborn.boxplot(x=X_test['PoliceDistrict_Out of SF'], y=Y_test, color='purple', ax=axes[1, 0])
    seaborn.boxplot(x=X_test['PoliceDistrict_Out of SF'], y=custom.y_predicted, color='blue', ax=axes[1, 1])
    seaborn.boxplot(x=X_test['PoliceDistrict_Out of SF'], y=linear.y_predicted, color='red', ax=axes[1, 2])
    seaborn.boxplot(x=X_test['PoliceDistrict_Out of SF'], y=svr.y_predicted, color='green', ax=axes[1, 3])

    seaborn.boxplot(x=X_test['PoliceDistrict_Southern'], y=Y_test, color='purple', ax=axes[2, 0])
    seaborn.boxplot(x=X_test['PoliceDistrict_Southern'], y=custom.y_predicted, color='blue', ax=axes[2, 1])
    seaborn.boxplot(x=X_test['PoliceDistrict_Southern'], y=linear.y_predicted, color='red', ax=axes[2, 2])
    seaborn.boxplot(x=X_test['PoliceDistrict_Southern'], y=svr.y_predicted, color='green', ax=axes[2, 3])

    pyplot.show()


def plot_custom_model(custom):
    seaborn.set(rc={'figure.figsize': (16, 16)})
    seaborn.scatterplot(x=range(0, y_range), y=Y_test, color='purple').set(title="Custom model")
    seaborn.scatterplot(x=range(0, y_range), y=custom.y_predicted, color='blue')
    pyplot.show()


def plot_linear_model(linear):
    seaborn.set(rc={'figure.figsize': (16, 16)})
    seaborn.scatterplot(x=range(0, y_range), y=Y_test, color='purple').set(title="Linear model")
    seaborn.scatterplot(x=range(0, y_range), y=linear.y_predicted, color='blue')
    pyplot.show()


def plot_svr_model(svr):
    seaborn.set(rc={'figure.figsize': (16, 16)})
    seaborn.scatterplot(x=range(0, y_range), y=Y_test, color='purple').set(title="SVR model")
    seaborn.scatterplot(x=range(0, y_range), y=svr.y_predicted, color='blue')
    pyplot.show()


def run():
    custom_solution = custom_model_imp()
    linear_solution = linear_model_imp()
    svr_solution = svr_model_imp()
    show_error_values(custom_solution, linear_solution, svr_solution)
    plot(custom_solution, linear_solution, svr_solution)
    plot_district_details(custom_solution, linear_solution, svr_solution)
    plot_custom_model(custom_solution)
    plot_linear_model(linear_solution)
    plot_svr_model(svr_solution)


if __name__ == '__main__':
    run()
