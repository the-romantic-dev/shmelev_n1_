from input import hidden_layer_size, delay, window, prediction_steps, activation
from util import get_mackey_glass_intervals
from model.neural_network import TDNN


if __name__ == '__main__':
    nn_model = TDNN(
        hidden_layer_size=hidden_layer_size,
        delay=delay,
        activation=activation,
        window=window
    )
    train_x, train_y = (
        get_mackey_glass_intervals(start=500, end=1000, delay=delay, window=window, prediction_steps=prediction_steps))
    nn_model.fit(
        train_x=train_x,
        train_y=train_y,
        epochs=100
    )

    nn_model.save()
