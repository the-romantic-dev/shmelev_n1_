import json


def get_mackey_glass_intervals(start, end, delay, window, prediction_steps) -> tuple[list[float], list[float]]:
    with open(f'data/makkey_glass_data.json', 'r') as file:
        data = json.load(file)

    mackey_glass_data = data['mackey_glass']
    x = mackey_glass_data[start - delay:end + window]
    y = mackey_glass_data[start + prediction_steps:end + prediction_steps]

    return x, y
