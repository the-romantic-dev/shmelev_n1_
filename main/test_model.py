import json
import numpy as np
from matplotlib import pyplot as plt

import util
from model.neural_network import TDNN
from input import hidden_layer_size, delay, window, prediction_steps, activation
def show_difference_plot(expected, predicted):
    if len(expected) != len(predicted):
        raise ValueError(f"Не совпадают размерности expected и predicted:\nexpected: {len(expected)}\npredicted: {len(predicted)}")
    t_values = [i for i in range(len(expected))]
    plt.plot(t_values, expected, label="Expected")
    plt.title(f"Ns = {hidden_layer_size}, D = {delay}, M={window}")
    plt.plot(t_values, predicted, label="Predicted")
    mse_arr = [(expected[t] - predicted[t])**2 for t in t_values]
    plt.plot(t_values, mse_arr, label=f"Difference. MSE = {round(float(np.mean(mse_arr)), 4)}")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    with open(f'data/makkey_glass_data.json', 'r') as file:
        data = json.load(file)

    mackey_glass_data = data['mackey_glass']
    nn_model = TDNN(
        hidden_layer_size=hidden_layer_size,
        delay=delay,
        activation=activation,
        window=window
    )

    nn_model.load()
    test_x, test_y = util.get_mackey_glass_intervals(
        start=1000, end=1500,
        delay=delay, window=window, prediction_steps=prediction_steps
    )
    predicted_y = nn_model.predict(test_x)
    show_difference_plot(test_y, predicted_y)
