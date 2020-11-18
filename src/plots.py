"""
    Bibliotecas referente aos plots do projeto
"""
import matplotlib.pyplot as plt

def plot_images(images, cmap=None):
    """Plotas as imagens passafas em images

    Args:
        images (list or np.array): imagens a serem plotadas
    """
    if isinstance(images, list):
        for img in images:
            plot_images(img, cmap)
    else:
        if cmap == 'gray':
            plt.imshow(images, 'gray')
        else:
            plt.imshow(images)
        plt.show()
