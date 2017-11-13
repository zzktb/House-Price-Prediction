import numpy as np
import matplotlib.pyplot as plt

def visualize_results(fig_number, show_grid, y_train, y_train_pred, y_test, y_test_pred):
    plt.figure(fig_number)
    plt.subplot(1, 2, 1)
    plt.title('Train Dataset')
    plt.plot(np.arange(y_train.size) + 1, y_train, 'b-', label='True Value')
    plt.plot(np.arange(y_train_pred.size) + 1, y_train_pred, 'r-', label='Predicted Value')
    plt.legend(loc='upper right')
    plt.xlabel('ID')
    plt.ylabel('House Price/$')
    if show_grid:
        plt.grid()

    plt.subplot(1, 2, 2)
    plt.title('Test Dataset')
    plt.plot(np.arange(y_test.size) + 1 + 1000, y_test, 'b-', label='True Value')
    plt.plot(np.arange(y_test_pred.size) + 1 + 1000, y_test_pred, 'r-', label='Predicted Value')
    plt.legend(loc='upper right')
    plt.xlabel('ID')
    plt.ylabel('House Price/$')
    if show_grid:
        plt.grid()
    plt.show()






