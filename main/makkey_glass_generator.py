import json

from matplotlib import pyplot as plt

from model.makkey_glass import MackeyGlassProcess


def show_mg_plot(mg_solution):
    t_values = [i for i in range(len(mg_solution))]
    plt.plot(t_values, mg_solution)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title(f'Последовательность Маккея-Гласса при a={a}, b={b}, tau={tau}')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    a = 0.3
    b = 0.1
    tau = 30
    steps = 2000
    x0 = 0.1
    mackey_glass_process = MackeyGlassProcess(a=a, b=b, tau=tau)
    solution = mackey_glass_process.solve(x0=x0, steps=steps)

    # Показать график значений последовательности Маккея-Гласса
    show_mg_plot(solution)

    json_data = {"mackey_glass": solution}

    with open(f'data/makkey_glass_data.json', 'w') as file:
        json.dump(json_data, file)
