import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(X0, Y0, X1):
    colors = ["#ffffff", "#ff6361", "#003f5c", "#63005c", "#006355", "#5c3f00"]
    sns.set_palette(sns.color_palette(colors))
    X = list(X0) + list(X1)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df.loc[:, "label"] = [str(Y0[i]) if i < len(Y0) else "unknown" for i, y in enumerate(X)]
    df = df.sort_values(by=["label"], ascending=True)

    fig, ax = plt.subplots()
    dplot = df[df["label"] == "unknown"]
    ax.scatter(dplot["x1"], dplot["x2"], label="unknown", s=15)

    for label in df[df["label"] != "unknown"].label.unique():
        dplot = df[df["label"] == label]
        ax.scatter(dplot["x1"], dplot["x2"], label=label, s=15)

    ax.legend()
    plt.show()
