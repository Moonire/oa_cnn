from numpy import linalg as LA
from rtree import index
import trimesh

from dijkstar.graph import Graph as DGraph
from dijkstar.algorithm import single_source_shortest_paths as compute_paths
from dijkstar.algorithm import extract_shortest_path_from_predecessor_list as get_path

import numpy as np
from .icp import icp
from .ops import level_corrector, pix_array, cyl2cart, norm, save, load, save_as_img
import os


def custom_nearest_nei(src, dst, bijection=False):
    """
    Returns a list of the closest point in dst to each element of scr,
    either as a bijection or a surjection.

    Arguments:
        src {list of tuples} -- vertecies of the start mesh.
        dst {list of tuples} -- vertecies of the destination mesh.

    Keyword Arguments:
        bijection {bool} -- determines the mapping type i.e bi/surjection.
                            when False multiple elements of src may be
                            assinged the same dst value. (default: {False})

    Returns:
        [list of tuples] -- (t, s) where s is an element of src
                            and t an element dst
    """
    p, end = index.Property(), []
    p.dimension = 3

    idx = index.Index('G_E_3D', properties=p, overwrite=True)
    for n, t in enumerate(dst):
        idx.add(n, t)

    for s in src:
        n = list(idx.nearest(s, 1))[0]
        end.append((dst[n], s))
        idx.delete(n, dst[n]) if bijection else None

    del idx
    os.remove('G_E_3D.dat'), os.remove('G_E_3D.idx')

    return end


class Process(object):
    """docstring for Process"""

    def __init__(self, empreinte, adjonction, ensemble, id_,
                 mode='arc', z_sup=20, z_inf=20):
        """
        [summary]

        Arguments:
            empreinte {[type]} -- [description]
            adjonction {[type]} -- [description]
            ensemble {[type]} -- [description]
            id_ {[type]} -- [description]

        Keyword Arguments:
            mode {str} -- [description] (default: {'arc'})
        """
        self.id_ = id_
        self.bundle = [empreinte, adjonction, ensemble]

        print("The files are being processed, this may take several minutes...")
        self._stl_grapher()

        print("Assembly...")
        self._assembler(mode=mode, z_sup=z_sup, z_inf=z_inf)

    def flatten(self, z_drop=10, nei=10):
        """
        description
        """
        self._faltten_emp(z_drop=z_drop)
        print('The empreinte has been processed')

        # self.bundle = load('assemble')

        self._outer_surface(nei=nei)
        self._faltten_adj()
        print('The adjonction has been processed')

    def _stl_grapher(self):
        """
        Extracts edges and vertices from the mesh, adjust the occlusal plan
        to be the horizontal. Very efficent (less the 60s).
        """
        for n, file in enumerate(self.bundle):
            obj = trimesh.load_mesh(file)
            obj.process
            obj.fix_normals()

            v = np.min(obj.vertices, axis=0)
            obj.apply_translation(-v)
            obj.apply_transform(level_corrector(obj.vertices))

            v = np.min(obj.vertices, axis=0)
            obj.apply_translation(-v)

            self.bundle[n] = obj

    def _assembler(self, mode, z_sup, z_inf):
        """
        [summary]

        Arguments:
            mode {str} -- [description]
            z_sup {number} -- [description]
            z_inf {number} -- [description]
        """
        E, G, A = self.bundle

        f = E.vertices
        h, h_normals = G.vertices, G.vertex_normals
        a, a_normals = A.vertices, A.vertex_normals

        if mode == 'arc':
            def condition_1(i, j): return True

            def condition_2(i, j): return j[2] > z_sup

        elif mode == 'plate':
            def condition_1(i, j): return (h_normals[i]/LA.norm(h_normals[i]), (0, 0, 1)) < 0.8

            def condition_2(i, j): return (a_normals[i]/LA.norm(a_normals[i]), (0, 0, 1)) < 0.8

        e = [i for i in f if i[2] < z_inf]
        g = [j for i, j in enumerate(h) if condition_1(i, j)][::10]

        T_e = icp(e, a)
        T_g = icp(g, [j for i, j in enumerate(a) if condition_2(i, j)])

        E.apply_transform(T_e), G.apply_transform(T_g)
        v = np.min(E.vertices, axis=0)
        E.apply_translation(-v), G.apply_translation(-v)

        self.bundle = [E, G, A]

    def _outer_surface(self, nei):
        """
        [summary]

        Arguments:
            nei {number} -- [description]
        """
        mesh_asset, mesh_target, _ = self.bundle

        normals = mesh_target.vertex_normals
        target_vertices = [tuple(i) for i in mesh_target.vertices]
        asset_vertices = [tuple(i) for i in mesh_asset.vertices]

        p, ana, kata = index.Property(), [], []
        p.dimension = 3

        idx = index.Index('asset_vertices', properties=p, overwrite=True)
        for n, i in enumerate(asset_vertices):
            idx.add(n, i)

        for n, i in enumerate(target_vertices):
            m = list(idx.nearest(i, nei))
            v = normals[n]
            c = [np.dot(v, np.subtract(u, v)/norm(u, v)) for u in m]

            kata.append(i) if np.mean(c) < 0 else ana.append(i)

        del idx
        os.remove('asset_vertices.dat'), os.remove('asset_vertices.idx')

        self.surface = {'kata': kata, 'ana': ana}

    def _faltten_emp(self, z_drop):
        """
        Injective Transformation of the 3D mesh of the model into a 3
        channel squared image, with satisfying properties of continuity,
        locality preservance and invariability to scaling.

        Cons : Nearest Neighbor step is time consuming and unparallelable,
        taking up to 28min for the treatment of a full scan ~500_000 vertex.

        Arguments:
            z_drop {int} -- height in mm below which the points are dropped.

        Note : z_drop used to get ride of the socle in undirectly scanned
        models. set to zero for intraoral scans and socle_free models.
        """
        A, B, _ = self.bundle

        E = [tuple(i) for i in A.vertices]
        G = [(0, 0, 0)] if B is None else [tuple(i) for i in B.vertices]

        span = np.max(E+G, axis=0)
        data = [tuple(np.divide(i, span)) for i in E]

        graph, edge = DGraph(), A.edges

        # hyperparm for z of 2ndary importance
        org = sorted(E, key=lambda x: norm(x, (span[0]/2, 0, 25)))[0]

        # creats an undirected graph representing the mesh of the model
        # eliminating the socle points
        for e in edge:
            u, v = E[e[0]], E[e[1]]
            if min(u[2], v[2]) > z_drop:
                graph.add_edge(u, v, {'cost': norm(u, v)})
                graph.add_edge(v, u, {'cost': norm(u, v)})

        def cost_func(u, v, e, prev_e): return e['cost']
        predecessor = compute_paths(graph=graph, s=org, cost_func=cost_func)

        idx = index.Index('E_P', overwrite=True)
        projection = []
        nounce = 0

        for i in E:
            x, y, z = i
            try:
                # R is total_cost of getting to i
                R = get_path(predecessor, i)[3]
                projection.append(cyl2cart((R, np.arctan2(y, x-org[0]), z))[0:2])

            except LookupError:
                nounce += 1
                projection.append((0, 0))

        s = np.min(projection, axis=0)
        t = np.max(projection, axis=0)

        projection = [tuple(np.subtract(p, s)/(t-s)) for p in projection]

        for n, i in enumerate(projection):
            if i != (0, 0):
                idx.add(n, i)

        # side : numbre of pixels in the side of the image
        # nounce retreaves the unused socle points
        side = int((len(data)-nounce)**(1/2)) + 1
        side = side+1 if side % 2 == 0 else side
        pixels = pix_array(side)

        img = {'span': span, 'side': side}
        for i in pixels:
            try:
                n = list(idx.nearest(i, 1))[0]
                img[i] = data[n]
                idx.delete(n, img[i][0:2])

            except LookupError:
                img[i] = (0, 0, 0)

        del idx
        os.remove('E_P.dat'), os.remove('E_P.idx')

        save(img, f'flat_e_{self.id_}')
        save_as_img(img, f'flat_e_{self.id_}', pixels, side)

    def _faltten_adj(self):
        """
        [summary]
        """
        for name, G in self.surface.items():

            # for testing
            G = G[::7]

            p_to_e = load(f'flat_e_{self.id_}')
            side, span = p_to_e['side'], p_to_e['span']
            pixels = load(f'pix_array_{side}')

            e = [p_to_e[i] for i in pixels]
            g = [tuple(np.divide(i, span)) for i in G]
            g = [i[1] for i in sorted(custom_nearest_nei(g, e), key=lambda x: norm(*x))]

            e_to_g = {i[0]: i[1] for i in custom_nearest_nei(g, e, bijection=True)}
            p_to_g = {'span': span}

            for i in pixels:
                try:
                    p_to_g[i] = e_to_g[p_to_e[i]]

                except LookupError:
                    p_to_g[i] = (0, 0, 0)

            save(p_to_g, f'flat_g_{self.id_}_{name}')
            save_as_img(p_to_g, f'flat_g_{self.id_}_{name}', pixels, side)

    def remesher(self):
        """
        [description]
        """
        _, G, _ = self.bundle
