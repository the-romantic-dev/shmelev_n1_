from keras import Input, Model, Sequential
from keras.src.layers import Dense, Flatten


def reshape_input(input_data: list[float], delay: int, window: int):
    if len(input_data) < delay + window:
        raise ValueError("Недостаточный размер последовательности")
    result = []
    for i in range(delay, len(input_data) - window):
        result.append([])
        for j in range(window):
            result[-1].append(input_data[i + j - delay:i + j])
    return result


class TDNN:
    def __init__(self, hidden_layer_size: int, delay: int, window: int, activation: str):
        self.delay = delay
        self.window = window
        self.model = Sequential()

        # Добавляем скрытый слой размера hidden_layer_size
        # (входной слой генерируется автоматически исходя из аргумента input_shape, задающего форму входных данных)
        self.model.add(Dense(units=hidden_layer_size, input_shape=(window, delay), activation=activation))
        # Т.к. с каждого нейрона выходит вектор размера delay, а нам на выходе нужно одно число -
        # мы превращаем выход из предыдущего слоя в одномерный вектор с помощью Flatten
        self.model.add(Flatten())
        # На выходе один нейрон, т.к. нам нужно предсказать одно число
        self.model.add(Dense(units=1))

        # Компилируем модель
        self.model.compile(optimizer='adam', loss='mse')

        # self.model = model

    def fit(self, train_x: list[float], train_y: list[float], epochs: int):
        reshaped_x = reshape_input(input_data=train_x, delay=self.delay, window=self.window)
        self.model.fit(reshaped_x, train_y, epochs=epochs)

    def predict(self, predictable_data: list[float]):
        reshaped_x = reshape_input(input_data=predictable_data, delay=self.delay,
                                   window=self.window)
        return self.model.predict(reshaped_x)

    def save(self):
        filename = "data/network.h5"
        self.model.save_weights(filename)

    def load(self):
        filename = "data/network.h5"
        self.model.load_weights(filename)
