from numpy import linalg as LA
from PIL import Image
import scipy.linalg
import numpy as np
import pickle


def norm(i, j): return LA.norm(np.subtract(i, j))


def cyl2cart(i): return (lambda x, y, z: (x*np.cos(y), x*np.sin(y), z))(*i)


def load(file):
    with open(file, 'rb') as fp:
        return pickle.load(fp)


def save(a, file):
    with open(file, 'wb') as fp:
        pickle.dump(a, fp)


def save_as_img(l, filename, pixels, side):
    """[summary]

    Arguments:
        l {[type]} -- [description]
        pixels {[type]} -- [description]
        side {[type]} -- [description]
    """

    img = Image.new('RGB', (side, side))
    for i in pixels:
        img.putpixel(np.subtract(np.array(i)*side, 1/2).astype(int),
                     tuple((np.array(l[i])*255).astype(int)))
    img.show()
    img.save(filename+".png", "PNG")


def pix_array(s):
    """[summary]

    [description]

    Arguments:
        s {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    try:
        return load(f'pix_array_{s}')

    except OSError:
        pixels = sorted([(x/s, y/s) for x in range(s) for y in range(s)], key=lambda x: norm((1/2, 0), x))
        save(pixels, f'pix_array_{s}')
        return pixels


def normal_unit_vector(data):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    data = np.array(data)
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
    n = [-C[0], -C[1], 1]
    return n/LA.norm(n)


def align_vectors(a, b):
    """[summary]

    Arguments:
        a {[type]} -- [description]
        b {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    a = -a if a[2] < 0 else a
    v, c = np.cross(a, b), np.dot(a, b)
    vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    return np.add(np.add(np.identity(3), vx), np.matmul(vx, vx)/(1+c))


def get_cusps(data, parts, bias=(0, 0), peaks=1):
    """[summary]

    [description]

    Returns:
        [type] -- [description]
    """
    l, out = [[] for _ in range(parts)], []
    x, y = bias

    for i in data:
        l[int(np.arctan2(i[1]-y, i[0]-x)*parts/np.pi)-1].append(i)

    L = [i for i in l if i != []]

    for i in L:
        for p in range(1, peaks+1):
            out.append(sorted(i, key=lambda x: x[2])[-p])
    return out


def level_corrector(data, z_drop=0.7, parts=100):
    """[summary]

    [description]

    Arguments:
        data {[type]} -- [description]

    Keyword Arguments:
        z_drop {number} -- [description] (default: {0.7})
        parts {number} -- [description] (default: {100})

    Returns:
        [type] -- [description]
    """
    atad = sorted(data, key=lambda x: x[2])[int(len(data)*(z_drop)):]
    top = get_cusps(atad, parts)
    R, I = align_vectors(normal_unit_vector(top), (0, 0, 1)), np.identity(4)
    for m in range(3):
        for n in range(3):
            I[m, n] = R[m, n]
    return I
