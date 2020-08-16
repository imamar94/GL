import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(X0, Y0, X1, return_ax=False):
    colors = ["#ffffff", "#ff6361", "#003f5c", "#63005c", "#006355", "#5c3f00"]
    sns.set_palette(sns.color_palette(colors))
    X = list(X0) + list(X1)
    xx = [x[0] for x in X]
    xy = [x[1] for x in X]
    max_lim = max(max(xx), max(xy)) + 0.1
    min_lim = min(min(xx), min(xy)) + 0.1
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df.loc[:, "label"] = [str(Y0[i]) if i < len(Y0) else "unknown" for i, y in enumerate(X)]
    df = df.sort_values(by=["label"], ascending=True)

    fig, ax = plt.subplots()
    dplot = df[df["label"] == "unknown"]
    ax.scatter(dplot["x1"], dplot["x2"], label="unknown", s=15)

    for label in df[df["label"] != "unknown"].label.unique():
        dplot = df[df["label"] == label]
        ax.scatter(dplot["x1"], dplot["x2"], label=label, s=15)

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.legend()
    if return_ax:
        return ax
    plt.show()


def plot2d(GL, return_ax=False):
    colors = ["#ed705c", "#79b1ed", "#ffb3ff", "#006355", "#5c3f00"]
    sns.set_palette(sns.color_palette(colors))
    # X = list(X0) + list(X1)
    X0 = GL._X0
    X1 = GL._X1
    Y1 = GL.Y[len(X0):]
    Y = GL.Y


    xx = [x[0] for x in X1]
    xy = [x[1] for x in X1]
    max_lim = max(max(xx), max(xy)) + 0.1
    min_lim = min(min(xx), min(xy)) + 0.1
    df = pd.DataFrame(X1, columns=["x1", "x2"])
    df.loc[:, "label"] = [str(Y1[i]) if i < len(Y1) else "unknown" for i, y in enumerate(X1)]
    df = df.sort_values(by=["label"], ascending=True)

    fig, ax = plt.subplots()
    # dplot = df[df["label"] == "unknown"]
    # ax.scatter(dplot["x1"], dplot["x2"], label="unknown", s=15)

    for label in df[df["label"] != "unknown"].label.unique():
        dplot = df[df["label"] == label]
        ax.scatter(dplot["x1"], dplot["x2"], label=label, s=10)

    ax.scatter(np.array(X0).T[0], np.array(X0).T[1], label="Initial Data", s=15, marker="x", color="black")

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.legend()
    if return_ax:
        return ax
    plt.show()


def plot2d(model, return_ax=False, title=""):
    colors = ["#ed705c", "#79b1ed", "#ffb3ff", "#82d4d1", "#f5dd9f"]
    sns.set_palette(sns.color_palette(colors))
    # X = list(X0) + list(X1)
    X0 = model._X0
    X1 = model._X1
    Y1 = model.Y[len(X0):]
    Y0 = model.Y[:len(X0)]
    Y = model.Y

    xx = [x[0] for x in X1]
    xy = [x[1] for x in X1]
    max_lim = max(max(xx), max(xy)) + 0.1
    min_lim = min(min(xx), min(xy)) + 0.1
    df = pd.DataFrame(X1, columns=["x1", "x2"])
    df.loc[:, "label"] = [str(Y1[i]) if i < len(Y1) else "unknown" for i, y in enumerate(X1)]
    df = df.sort_values(by=["label"], ascending=True)

    fig, ax = plt.subplots()
    # dplot = df[df["label"] == "unknown"]
    # ax.scatter(dplot["x1"], dplot["x2"], label="unknown", s=15)

    for label in df[df["label"] != "unknown"].label.unique():
        dplot = df[df["label"] == label]
        ax.scatter(dplot["x1"], dplot["x2"], label=label, s=7)

    df = pd.DataFrame(X0, columns=["x1", "x2"])
    df.loc[:, "label"] = ["Initial_" + str(Y0[i]) if i < len(Y0) else "unknown" for i, y in enumerate(X0)]
    df = df.sort_values(by=["label"], ascending=True)
    colors = ["#691b05", "#071e7a", "#690664", "#006355", "#5c3f00"]
    for i, label in enumerate(df[df["label"] != "unknown"].label.unique()):
        dplot = df[df["label"] == label]
        ax.scatter(dplot["x1"], dplot["x2"], label=label, s=20, marker="x", color=colors[i])
    # ax.scatter(np.array(X0).T[0], np.array(X0).T[1], label="Initial Data", s=15, marker="x", color="red")

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_title(title)

    ax.legend()
    if return_ax:
        return ax
    plt.show()