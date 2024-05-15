class MackeyGlassProcess:
    def __init__(self, a: float, b: float, tau: int):
        self.tau = tau
        self.first_mg_part = lambda x, i, tau: (a * x[i - tau]) / (1 + x[i - tau] ** 10)
        self.last_mg_part = lambda x, i: -b * x[i]

    def solve(self, x0: float | int, steps: int):
        """
        Решение последовательности Маккея-Гласса с помощью метода Эйлера

        :param x0: начальное значение для метода Эйлера
        :param steps: Количество искомых шагов последовательности
        """
        dt = 1  # Шаг метода Эйлера
        n = steps  # Количество искомых шагов последовательности
        x = [0 for _ in range(n)]  # Массив нулей для заполнения значениями последовательности
        x[0] = x0

        # Количество шагов метода относительно задержки tau
        tau_steps = round(self.tau / dt)

        # Генерация первых tau значений
        for i in range(tau_steps):
            x[i + 1] = x[i] + dt * self.last_mg_part(x, i)

        # Генерация последующих значений
        for i in range(tau_steps, n - 1):
            x[i + 1] = x[i] + dt * (self.first_mg_part(x, i, tau_steps) + self.last_mg_part(x, i))

        return x
