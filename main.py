from Solution import Solution, custom_model_imp,\
    linear_model_imp, svr_model_imp, y_axis, Y_test, X_test
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

    seaborn.scatterplot(x=range(0, y_axis), y=Y_test, color='purple', ax=axes[0, 0])
    seaborn.scatterplot(x=range(0, y_axis), y=custom.y_predicted, color='blue', ax=axes[0, 1])
    seaborn.scatterplot(x=range(0, y_axis), y=linear.y_predicted, color='red', ax=axes[1, 0])
    seaborn.scatterplot(x=range(0, y_axis), y=svr.y_predicted, color='green', ax=axes[1, 1])

    pyplot.show()


def plot_details(custom, linear, svr):
    names = list(X_test.keys())
    districts = []
    for name in names:
        if "PoliceDistrict" in name:
            districts.append(name)

    #   seaborn.barplot(x=districts[0], y=Y_test, color='purple')
    #   pyplot.show()
    """fig, axes = pyplot.subplots(4, 3)
    fig.set_figheight(16)
    fig.set_figwidth(16)
    axes[0, 0].set_title('Real data')
    axes[0, 0].set_ylabel('Time difference')
    axes[0, 1].set_title('Custom model')
    axes[0, 1].set_ylabel('Time difference')
    axes[1, 0].set_title('Linear model')
    axes[1, 0].set_ylabel('Time difference')
    axes[1, 1].set_title('SVR model')
    axes[1, 1].set_ylabel('Time difference')"""




def run():
    custom_solution = custom_model_imp()
    linear_solution = linear_model_imp()
    svr_solution = svr_model_imp()
    show_error_values(custom_solution, linear_solution, svr_solution)
    #plot(custom_solution, linear_solution, svr_solution)
    plot_details(custom_solution, linear_solution, svr_solution)


if __name__ == '__main__':
    run()
