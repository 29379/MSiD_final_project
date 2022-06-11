from Solution import Solution, custom_model_imp,\
    linear_model_imp, svr_model_imp, y_range, Y_test, Y_test
import seaborn
from matplotlib import pyplot
import time


def plot():
    lin_wrapper = linear_model_imp()
    svr_wrapper = svr_model_imp()
    #   cus_wrapper = custom_model_imp()
    linear_solution = Solution(lin_wrapper[0], lin_wrapper[1], lin_wrapper[2], lin_wrapper[3])
    svr_solution = Solution(svr_wrapper[0], svr_wrapper[1], svr_wrapper[2], svr_wrapper[3])
    #   custom_solution = Solution(cus_wrapper[0], cus_wrapper[1], cus_wrapper[2], cus_wrapper[3])

    seaborn.scatterplot(x=range(0, y_range), y=Y_test, color='purple')
    #   seaborn.scatterplot(x=range(0, y_range), y=custom_solution.y_predicted)
    seaborn.scatterplot(x=range(0, y_range), y=linear_solution.y_predicted, color='red')
    seaborn.scatterplot(x=range(0, y_range), y=svr_solution.y_predicted, color='green')
    pyplot.show()


if __name__ == '__main__':
    start = time.time()
    print(start)
    plot()
    print("\n--- %s seconds ---" % (time.time() - start))
