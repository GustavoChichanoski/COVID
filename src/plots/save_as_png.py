import os
import matplotlib.pyplot as plt

def save_png(fig, path: str, overwrite: bool = True) -> str:
    if os.path.exists(path):
        if overwrite:
            i = 0
            extension = path.split('.')
            fig_path = os.path.join(path,'mc_{}_pacotes_{}.png'.format(i))
            while os.path.exists(fig_path):

                fig_path = os.path.join(path,'mc_{}_pacotes_{}.png'.format(i))
                i += 1
            plt.savefig(fig_path,dpi=fig.dpi)
            return fig_path
        print('[save_png] Arquivo já existe: {}'.format(path))
        return path
    plt.savefig(path,dpi=fig.dpi)
    return path