import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_png(fig: Figure, path: Path, overwrite: bool = True) -> str:
    if os.path.exists(path) and not overwrite:
        i = 0
        fig_path = change_extension(path, i)
        while os.path.exists(fig_path):
            fig_path = change_extension(path, i)
            i += 1
        plt.savefig(fig_path, dpi=fig.dpi)
        print(f"[INFO] Arquivo já existe: {fig_path}")
        return fig_path
    plt.savefig(path, dpi=fig.dpi)
    print(f"[INFO] Arquivo salvo em: {path}")
    return path


def change_extension(path: Path, i: int) -> Path:
    return Path(str(path).split(".")[0] + f"_{i}.png")
