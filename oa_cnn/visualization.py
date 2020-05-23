import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from numpy import linalg as LA
import numpy as np

from scipy.special import gamma
from PIL import Image
from .ops import load


def img_loss_function(id_=1, mode='ana', formats=['png', 'stats']):
    """[summary]

    Keyword Arguments:
        id_ {number} -- [description] (default: {1})
        mode {str} -- [description] (default: {'ana'})
        formats {list} -- [description] (default: {['png', 'stats']})
    """
    data = load('./Spreaded/e_%s' % id_)
    atad = load('./Spreaded/g_%s_%s' % (mode, id_))
    side, span = data['side'], data['span']

    pixels = load('./xPix_arrays/Pix_array_%s' % side)
    pixels.sort(key=lambda x: x[0]*side + x[1]*side**2)

    M, S = [], []
    for i in pixels:
        if atad[i] != (0, 0, 0):
            j = LA.norm(np.matmul(np.diag(span), np.subtract(atad[i], data[i])))
            M.append(j), S.append(j)

        else:
            M.append(0)

    if 'png' in formats:
        img = Image.new('L', (side, side))
        v = max(M)
        img.putdata([int(i*255/v) for i in M])
        img.show()

    if 'stats' in formats:
        print(f"max {max(S)}, min {min(S)}, mean {(sum(S)/len(S))}, std {np.std(S)}")

        n, bins, _ = plt.hist(S, 100, normed=True, edgecolor='black', alpha=0.75)
        x = np.linspace(min(bins), max(bins), 10000)

        plt.plot(x, gamma_pdf(S, x), linewidth=2, color='r'), plt.grid(linestyle='dashed')
        plt.show()


def plot3d(*args, step=1, kind='scatter', color=['red', 'blue', 'green', 'orange']):
    """[summary]

    [description]

    Arguments:
        *args {[type]} -- [description]

    Keyword Arguments:
        step {number} -- [description] (default: {1})
        kind {str} -- [description] (default: {'scatter'})
        color {list} -- [description] (default: {['red']})
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n, i in enumerate(args):
        x, y, z = zip(*i[::step])
        if kind == 'wire':
            ax.plot_wireframe(x, y, z, color=color[n % 4])
        if kind == 'scatter':
            ax.scatter(x, y, z, s=0.1, color=color[n % 4])
        if kind == 'tri':
            ax.plot_trisurf(x, y, z, color=color[n % 4])
    plt.show()


def plot2d(a, save=False):
    """[summary]

    Arguments:
        a {[type]} -- [description]

    Keyword Arguments:
        save {bool} -- [description] (default: {False})
    """
    x, y, z = zip(*a)
    plt.scatter(x, y, c=z, s=0.5, cmap='jet')

    if save:
        plt.savefig(f'./Projections/{save}.png')

    plt.show()


def lognormal(S, x):
    m, s = np.mean(np.log(S)), np.std(np.log(S))
    pdf = (np.exp(-(np.log(x) - m)**2/(2 * s**2))/(x * s * np.sqrt(2 * np.pi)))
    return pdf


def gamma_pdf(S, x):
    m, v = np.mean(S), np.var(S)
    th = v/m
    k = m**2/v
    print('K: {k}, Theta: {th}')
    pdf = ((x**(k-1)*np.exp(-x/th))/(gamma(k)*th**k))
    return pdf
