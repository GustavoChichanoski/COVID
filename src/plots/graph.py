from pathlib import Path
from typing import Any, List, Tuple
import os
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from seaborn.matrix import heatmap


def normalize_confusion_matrix(matrix):
    norm = np.array([])
    if len(matrix.shape) > 2:
        for split in matrix:
            norm = np.append(norm, normalize_confusion_matrix(split))
            return np.reshape(norm, matrix.shape)
    for row in matrix:
        sum = np.sum(row)
        if sum == 0:
            sum = 1
        norm = np.append(norm, row / sum)
    return np.reshape(norm, matrix.shape)


def dist_dataset(
    matrix: Any,
    porc: Any,
    category_names: List[str],
    n_splits: int = 1
) -> str:

    results = {category_names[i]: porc[i][:] for i in range(len(category_names))}
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):

        widths = data[:, i]
        starts = data_cum[:, i] - widths

        # Barra lateral
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

        # Centro da edição horizontal
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "black"
        array = [matrix[j][i] for j in range(len(matrix))]

        # Escreve os valores sobre os grafico
        for y, (x, c) in enumerate(zip(xcenters, array)):
            ax.text(x, y, str(int(c)), ha="center", va="center", color=text_color)

    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )
    if n_splits == 1:
        ax.set_title(f"{n_splits} imagem")
    else:
        ax.set_title(f"{n_splits} imagens")
    return fig, ax


def plot_dataset(
    absolut: Any = None,
    n_images: List[int] = [1, 2, 3, 4],
    names: List[int] = ["COVID-19", "Normal", "Pneumonia"],
    path: Path = None,
    overwrite: bool = True,
):

    perc = normalize_confusion_matrix(absolut)
    if isinstance(n_images, list):
        for i in range(len(n_images)):
            dist_dataset(absolut[i], perc, names, n_images[i])
    else:
        dist_dataset(absolut, perc, names, n_images)
    plt.show()

    # fig, ax = plt.subplots()
    # im = ax.imshow(absolut)
    # ax.set_xticks(np.arange(len(names)))
    # ax.set_yticks(np.arange(len(names)))
    # ax.set_xticklabels(names)
    # ax.set_yticklabels(names)

    # plt.setp(ax.get_xticklabels(), rotation=45,
    #          ha='right', rotation_mode='anchor')

    # for i in range(len(names)):
    #     for j in range(len(names)):
    #         text = ax.text(j, i, absolut[i][j],
    #                        ha='center', va='center', color='w')

    # ax.set_title('Matriz Confusao')
    # fig.tight_layout()
    # plt.show()

    fig, ax = plt.subplots()
    im, cbar = heatmap(
        np.array(absolut), names, names, ax=ax, cmap="Blues", cbarlabel=None
    )

    texts = annotate_heatmap(im, valfmt="{x}")

    ax.set_title("Matriz de confusão")

    ax.set_xlabel("Rótulo Verdadeiro")
    ax.set_ylabel("Rótulo Predição")
    fig.tight_layout()
    # if path is not None:
    #     fig_path = os.path.join(path,'matriz_confusao.png')
    #     if overwrite:
    #         i = 0
    #         while os.path.exists(fig_path):
    #             fig_path = os.path.join(path,'matriz_confusao_{}.png'.format(i))
    #             i += 1
    #     plt.savefig(fig_path,dpi=fig.dpi)
    plt.show()
    mc_path = path / f"mc_{n_images}_pacotes.png"
    if mc_path.exists():
        if overwrite:
            i = 0
            mc_path = str(mc_path.absolute())[:-4]
            fig_path = f"{mc_path}_{i}.png"
            while os.path.exists(fig_path):
                i += 1
                fig_path = f"{mc_path}_{i}.png"
            plt.savefig(fig_path, dpi=fig.dpi)
            return fig_path
        print(f"[PLOT] Arquivo já existe: {str(path)}")
        return mc_path
    plt.savefig(mc_path, dpi=fig.dpi)


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels, rotation=90)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im: Any,
    data: Any = None,
    valfmt: str ="{x:.2f}",
    textcolors: Tuple[str,str] = ("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied. If None (the default) uses the middle of the colormap as
        separation. Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mlp.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def test():
    matrix1 = [[246, 8, 28], [8, 501, 41], [14, 26, 835]]
    matrix2 = [[264, 6, 12], [3, 518, 29], [4, 26, 845]]
    matrix3 = [[266, 4, 12], [5, 523, 22], [4, 24, 847]]
    matrix4 = [[269, 1, 12], [2, 526, 25], [1, 20, 854]]
    matrix5 = [[265, 3, 14], [1, 524, 22], [2, 16, 857]]
    matrix10 = [[274, 1, 7], [1, 524, 25], [1, 17, 857]]
    matrix50 = [[272, 1, 9], [1, 524, 25], [1, 18, 856]]
    ideal = [[282, 0, 0], [0, 550, 0], [0, 0, 875]]

    matrix = np.array([matrix1, matrix2, matrix3, matrix4, matrix5, matrix10, matrix50])
    plot_dataset(["Covid", "Normal", "Pneumonia"], matrix)


if __name__ == "__main__":
    test()
