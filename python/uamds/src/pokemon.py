import numpy as np
import pandas as pd
import distrib_estimation as dstrbest


def get_raw_data() -> pd.DataFrame:
    return pd.read_csv("res/pokemon_stats.csv", delimiter=";")


def get_normal_distribs() -> dict[dstrbest.MVN]:
    df = get_raw_data()
    distribs = {}
    for i in range(df.shape[0]):
        low = df.iloc[i, 9:9+6].to_numpy(dtype=np.float64) / 100
        high = df.iloc[i, 15:15+6].to_numpy(dtype=np.float64) / 100
        mean = (low+high)/2
        deviation = (low-high)/2
        diag_variance = (deviation**2)/12
        cov = np.diag(diag_variance)
        distribs[i] = dstrbest.MVN(mean, cov)
    return distribs

def get_type1() -> list[str]:
    df = get_raw_data()
    types = df["types"]

    def typ1(name: str) -> str:
        return name.split(" ")[0]

    type1s = [typ1(types.iloc[i]) for i in range(types.shape[0])]
    return type1s

def get_type_colors() -> dict[str, str]:
    colors = {}
    colors["Normal"] = "#AAAA99"
    colors["Fire"] = "#FF4422"
    colors["Water"] = "#3399FF"
    colors["Electric"] = "#FFCC33"
    colors["Grass"] = "#77CC55"
    colors["Ice"] = "#66CCFF"
    colors["Fighting"] = "#BB5544"
    colors["Poison"] = "#AA5599"
    colors["Ground"] = "#DDBB55"
    colors["Flying"] = "#8899FF"
    colors["Psychic"] = "#FF5599"
    colors["Bug"] = "#AABB22"
    colors["Rock"] = "#BBAA66"
    colors["Ghost"] = "#6666BB"
    colors["Dragon"] = "#7766EE"
    colors["Dark"] = "#775544"
    colors["Steel"] = "#AAAABB"
    colors["Fairy"] = "#EE99EE"
    return colors
