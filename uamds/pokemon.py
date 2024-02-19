import numpy as np
import pandas as pd
from importlib import resources


def get_raw_data() -> pd.DataFrame:
    res = resources.files("res").joinpath("pokemon_stats.csv")
    return pd.read_csv(res, delimiter=";")


def get_normal_distribs() -> list[dict]:
    df = get_raw_data()
    distribs = []
    for i in range(df.shape[0]):
        low = df.iloc[i, 9:9+6].to_numpy(dtype=np.float64) / 100
        high = df.iloc[i, 15:15+6].to_numpy(dtype=np.float64) / 100
        mean = (low+high)/2
        deviation = (low-high)/2
        diag_variance = (deviation**2)/12
        cov = np.diag(diag_variance)
        distribs.append({'mean': mean, 'cov': cov})
    return distribs


def get_type1() -> list[str]:
    df = get_raw_data()
    types = df["types"]

    def typ1(name: str) -> str:
        return name.split(" ")[0]

    type1s = [typ1(types.iloc[i]) for i in range(types.shape[0])]
    return type1s

def get_names() -> list[str]:
    df = get_raw_data()
    names = df["name"]
    return [names[i] for i in range(names.size)]


def get_type_colors() -> dict[str, str]:
    colors = {
        "Normal": "#AAAA99",
        "Fire": "#FF4422",
        "Water": "#3399FF",
        "Electric": "#FFCC33",
        "Grass": "#77CC55",
        "Ice": "#66CCFF",
        "Fighting": "#BB5544",
        "Poison": "#AA5599",
        "Ground": "#DDBB55",
        "Flying": "#8899FF",
        "Psychic": "#FF5599",
        "Bug": "#AABB22",
        "Rock": "#BBAA66",
        "Ghost": "#6666BB",
        "Dragon": "#7766EE",
        "Dark": "#775544",
        "Steel": "#AAAABB",
        "Fairy": "#EE99EE"
    }
    return colors
