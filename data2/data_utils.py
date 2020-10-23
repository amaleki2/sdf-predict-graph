import torch
from torch_geometric.data import Data, DataLoader
import os
import cv2
import meshio
import numpy as np
import networkx as nx
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix, KDTree
from itertools import product


def compute_edges_img(img):
    edge_lower_tresh = 50
    edge_upper_tresh = 200
    running_img = img.copy()
    running_img *= 255
    running_img = running_img.astype(np.uint8)
    edges = cv2.Canny(running_img, edge_lower_tresh, edge_upper_tresh)
    return edges


def create_geo_file(img, lc1=0.2, lc2=0.1, skip=1, geo_file="img.geo", refined=True):
    edges = compute_edges_img(img.astype(float))
    edges = np.array(np.where(edges)).T
    edges = edges * 2 / (img.shape[0] - 1) - 1

    obj_center = np.mean(edges, axis=0)
    inner_edges = 2/3 * obj_center + 1/3 * edges

    with open(geo_file, "w") as fid:
        fid.write("lc1 = %0.3f; \n" % lc1)
        fid.write("lc2 = %0.3f; \n\n" % lc2)

        fid.write("// border points of rectangle \n")
        fid.write("Point(1) = {-1.0, -1.0, 0.0, lc1}; \n")
        fid.write("Point(2) = { 1.0, -1.0, 0.0, lc1}; \n")
        fid.write("Point(3) = { 1.0,  1.0, 0.0, lc1}; \n")
        fid.write("Point(4) = {-1.0,  1.0, 0.0, lc1}; \n\n")

        if refined:
            fid.write("// control points on the border of object \n")
            p_idx = 5
            EPS = 0.001
            for e in edges[::skip]:
                if e[1] == -1:
                    fid.write("Point(%d) = {%0.3f, %0.3f, 0.0, lc2}; \n" % (p_idx, e[1] + EPS, e[0]))
                elif e[1] == 1:
                    fid.write("Point(%d) = {%0.3f, %0.3f, 0.0, lc2}; \n" % (p_idx, e[1] - EPS, e[0]))
                elif e[0] == -1:
                    fid.write("Point(%d) = {%0.3f, %0.3f, 0.0, lc2}; \n" % (p_idx, e[1], e[0] + EPS))
                elif e[0] == 1:
                    fid.write("Point(%d) = {%0.3f, %0.3f, 0.0, lc2}; \n" % (p_idx, e[1], e[0] - EPS))
                else:
                    fid.write("Point(%d) = {%0.3f, %0.3f, 0.0, lc2}; \n" % (p_idx, e[1], e[0]))
                    p_idx += 1
            fid.write("\n")
            for e in inner_edges[::skip * 2]:
                fid.write("Point(%d) = {%0.3f, %0.3f, 0.0, lc1}; \n" % (p_idx, e[1], e[0]))
                p_idx += 1

        fid.write("// boundary lines of rectangle \n")
        fid.write("Line(1) = {1, 2}; \n")
        fid.write("Line(2) = {3, 2}; \n")
        fid.write("Line(3) = {3, 4}; \n")
        fid.write("Line(4) = {4, 1}; \n\n")

        fid.write("Curve Loop(1) = {4, 1, -2, 3}; \n")
        fid.write("Plane Surface(1) = {1}; \n\n")

        if refined:
            fid.write("// control points mesh size \n")
            for p in range(p_idx, 5, -1):
                fid.write("Point{%d} In Surface{1}; \n" % (p - 1))


def geo2vtk(geo_file, vtk_file=None):
    if vtk_file is None:
        vtk_file = geo_file[:-4] + ".vtk"

    str_cmd = "gmsh " + geo_file + " -2 -o " + vtk_file
    os.system(str_cmd)


### vtk to obj ####
def vtk2obj(vtk_file, obj_file=None, data=None):
    mesh = meshio.read(vtk_file)
    pnts = mesh.points
    if data is not None:
        assert all(pnts[:, 2] == 0.)
        pnts[:, 2] = data

    faces = [c for c in mesh.cells if c.type == "triangle"][0].data
    if obj_file is None:
        obj_file = vtk_file[:-4] + ".obj"
    file_name = os.path.basename(obj_file)
    with open(obj_file, 'w') as fid:
        # print header
        fid.write("####\n")
        fid.write("# OBJ File Generated from vtk\n")
        fid.write("#\n")
        fid.write("####\n")

        fid.write("# %s \n" % file_name)
        fid.write("#\n")
        fid.write("# Vertices: %d \n" % (len(pnts)))
        fid.write("# Faces: %d \n" % (len(faces)))
        fid.write("#\n")
        fid.write("#### \n\n")

        # print vertices
        for pnt in pnts:
            fid.write("v %0.4f %0.4f %0.4f\n" % (pnt[0], pnt[1], pnt[2]))

        fid.write("\n")
        # print vertices
        for face in faces:
            fid.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))

        fid.write("\n")
        fid.write("# End of File")


#### mesh functionalities ######
def vtk_to_mesh(vtk_file):
    mesh = meshio.read(vtk_file)
    return mesh


def plot_mesh(mesh, dims=2, node_labels=False, vals=None, with_colorbar=False, levels=None, border=None):
    if not isinstance(mesh.points, np.ndarray):
        mesh.points = np.array(mesh.points)
    nodes_x = mesh.points[:, 0]
    nodes_y = mesh.points[:, 1]
    if dims == 2:
        elements_tris = [c for c in mesh.cells if c.type == "triangle"][0].data
        #plt.figure(figsize=(8, 8))
        if vals is None:
            plt.triplot(nodes_x, nodes_y, elements_tris, alpha=0.9, color='r')
        else:
            triangulation = tri.Triangulation(nodes_x, nodes_y, elements_tris)
            p = plt.tricontourf(triangulation, vals, 30)
            if with_colorbar: plt.colorbar()
            if levels:
                cn = plt.tricontour(triangulation, vals, levels, colors='w')
                plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)
        if border:
            plt.hlines(1-border, -1+border, 1-border, 'r')
            plt.hlines(-1+border, -1+border, 1-border, 'r')
            plt.vlines(1-border, -1+border, 1-border, 'r')
            plt.vlines(-1+border, -1+border, 1-border, 'r')

    if node_labels:
            for i, (x, y) in enumerate(zip(nodes_x, nodes_y)):
                plt.text(x, y, i)

    if vals is not None:
        return p


def plot_mesh_onto_line(mesh, val, x=None, y=None, show=False, linestyle="-"):
    if not isinstance(mesh.points, np.ndarray):
        mesh.points = np.array(mesh.points)
    assert (x is None) ^ (y is None), "one of x or y has to be None"
    if x is None:
        x = np.linspace(-1, 1, 50)
        y = np.ones_like(x) * y
        plotting_axes = x
    else:  # y is None:
        y = np.linspace(-1, 1, 50)
        x = np.ones_like(y) * x
        plotting_axes = y

    nodes_x = mesh.points[:, 0]
    nodes_y = mesh.points[:, 1]
    elements_tris = [c for c in mesh.cells if c.type == "triangle"][0].data
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_tris)
    interpolator = tri.LinearTriInterpolator(triangulation, val)
    val_over_line = interpolator(x, y)
    plt.plot(plotting_axes, val_over_line, linestyle=linestyle)
    if show: plt.show()

def cells_to_edges(cells):
    edge_pairs = []
    for cell in cells:
        for p1, p2 in [(0, 1), (1, 2), (0, 2)]:
            edge = sorted([cell[p1], cell[p2]])
            if edge not in edge_pairs:
                edge_pairs.append(edge)
    return edge_pairs

### Graph funcitonalities ###
def mesh_to_graph(mesh, vals=None):
    if not isinstance(mesh.points, np.ndarray):
        mesh.points = np.array(mesh.points)

    points = mesh.points
    cells  = np.array([c.data for c in mesh.cells if c.type == 'triangle'][0])

    if vals is None:
        node_features = [(i, {"x": x, "y": y, "z": z}) for i, (x, y, z) in enumerate(points)]
    else:
        node_features = [(i, {"x": x, "y": y, "z": z, "val": val}) for i, ((x, y, z), val) in
                         enumerate(zip(points, vals))]
    edge_pairs = cells_to_edges(cells)

    graph = nx.Graph()
    graph.add_nodes_from(node_features)
    graph.add_edges_from(edge_pairs)
    return graph


def plot_graph(graph, node_labels=False, color=None, color_feature=None):
    if color_feature is not None:
        color = graph_to_features(graph, color_feature)

    plt.figure(figsize=(8, 8))
    nx.draw(graph, with_labels=node_labels, node_color=color)


def graph_to_cells(graph):
    cells = []
    for i in range(len(graph.nodes)):
        i_neighbours = set(graph.adj.get(i))
        for j in i_neighbours:
            j_neighbours = set(graph.adj.get(j))
            union_ij = i_neighbours.intersection(j_neighbours)
            for k in union_ij:
                cell = sorted([i, j, k])
                if cell not in cells:
                    cells.append(cell)
    return cells


def graph_to_points(graph):
    n_nodes = len(graph.nodes)
    points = [[graph.nodes.get(i)["x"],
               graph.nodes.get(i)["y"],
               graph.nodes.get(i)["z"]] for i in range(n_nodes)]
    return points


def graph_to_features(graph, key):
    n_nodes = len(graph.nodes)
    vals = [graph.nodes.get(i)[key] for i in range(n_nodes)]
    return vals


def graph_to_mesh(graph):
    points = np.array(graph_to_points(graph))
    cells = graph_to_cells(graph)
    cells = [("triangle", np.array(cells))]
    mesh = meshio.Mesh(points=points, cells=cells)
    return mesh


def expand_edge_connection(edges, k=2):
    assert isinstance(edges, np.ndarray)
    assert edges.shape[0] == 2

    n_edges = edges.shape[1]
    vertices_idx = np.unique(edges)
    n_vertices = len(vertices_idx)
    assert min(vertices_idx) == 0
    assert max(vertices_idx) == n_vertices - 1
    data = np.ones(n_edges * 2)
    rows = np.concatenate([edges[0, :], edges[1, :]])
    columns = np.concatenate([edges[1, :], edges[0, :]])
    adj_matrix = coo_matrix((data, (rows, columns)), shape=(n_vertices, n_vertices))
    adj_matrix = adj_matrix.toarray()
    if k == 2:
        new_adj_matrix = adj_matrix + np.matmul(adj_matrix, adj_matrix)
    elif k == 3:
        tmp2 = np.matmul(adj_matrix, adj_matrix)
        new_adj_matrix = adj_matrix + tmp2 + np.matmul(tmp2, adj_matrix)
    elif k == 4:
        tmp2 = np.matmul(adj_matrix, adj_matrix)
        tmp3 = np.matmul(tmp2, adj_matrix)
        new_adj_matrix = adj_matrix + tmp2 + tmp3 + np.matmul(tmp3, adj_matrix)
    else:
        raise("error")

    # set diagonal and lower diagonal values to 0, avoid counting edges twice.
    np.fill_diagonal(new_adj_matrix, 0)
    new_adj_matrix *= np.tri(*new_adj_matrix.shape)
    new_edges = np.array(np.where(new_adj_matrix > 0))
    return new_edges.T


def numpy_array_intersect(a, b):
    # trick is to treat the rows as a single value
    # assert np.ndim(a) == np.ndim(b) == 2
    # nrows, ncols = a.shape
    # dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [a.dtype]}
    # c = np.intersect1d(a.view(dtype), b.view(dtype))
    # c = c.view(a.dtype).reshape(-1, ncols)
    # return c
    aset = set([tuple(x) for x in a])
    bset = set([tuple(x) for x in b])
    return np.array([x for x in aset & bset])

def points_to_knn(points, k, max_radius):
    tree = KDTree(points)
    indices = tree.query(points, k+1)[1]
    edge_pairs = []
    for idx in indices:
        edge_pairs += [[idx[0], x] for x in idx[idx > idx[0]]]# list(product(left_node, right_node))
    edge_pairs = np.array(edge_pairs)
    neighbours = points_to_neighbours(points, [max_radius], type="circular").astype(int)
    knn_neighbours = numpy_array_intersect(edge_pairs, neighbours)
    return knn_neighbours


def points_to_neighbours(points, params, type="circular"):
    if type == "circular":
        assert len(params) == 1
        radius = params[0]
        dist = distance_matrix(points, points)
        dist += 2 * radius * np.tril(np.ones_like(dist))  # to avoid duplication later
        neighbours = np.array(np.where(dist < radius))
    elif type == "rectangular":
        assert len(params) == 2
        lx, ly = params
        distx = distance_matrix(points[:, 0].reshape(-1, 1), points[:, 0].reshape(-1, 1))
        disty = distance_matrix(points[:, 1].reshape(-1, 1), points[:, 1].reshape(-1, 1))
        in_filter = np.logical_and(distx < lx, disty < ly)
        in_filter[np.triu_indices(points.shape[0])] = False  # to avoid duplication later
        neighbours = np.array(np.where(in_filter))
    else:
        raise(NotImplementedError("this type is not defined."))
    return neighbours.T


def mesh_to_edge_neighbours(graph_nodes, mesh_edges, mesh_cells, connection_type):
    edge_neighbours = []
    edges = [set(x) for x in mesh_edges]
    if connection_type == "vertex":
        for i, edge_i in enumerate(edges):
            for j, edge_j in enumerate(edges[i+1:]):
                if len(edge_i.intersection(edge_j)) > 0:
                    if [i, j + i + 1] not in edge_neighbours:
                        edge_neighbours.append([i, j + i + 1])
    elif connection_type == "cell":
        cells = [set(x) for x in mesh_cells]
        for i, edge_i in enumerate(edges):
            for j, edge_j in enumerate(edges[i+1:]):
                if len(edge_i.intersection(edge_j)) > 0:
                    if edge_i.union(edge_j) in cells:
                        if [i, j + i + 1] not in edge_neighbours:
                            edge_neighbours.append([i, j + i + 1])
    else:
        raise("connection_type %s is not defined." %connection_type)

    return edge_neighbours


def compute_edge_weight(nodes, edge, method):
    if method == "length":
        dist = distance_matrix(nodes, nodes)
        idx = tuple(np.array(edge).T)
        weights = dist[idx]
    else:
        raise(NotImplementedError("method %s is NOT supported for edge weight." %method))
    return 1 / weights


def get_sdf_data_loader(n_objects, data_folder, batch_size, mesh_folder=None,
                        reversed_edge_already_included=False, edge_weight=False):
    graph_data_list = []
    print("preparing sdf data loader")
    if mesh_folder:
        print('reading pixel data for target')
        pxl_size = 128
        xc, yc = np.meshgrid(np.linspace(-1, 1, pxl_size), np.linspace(-1, 1, pxl_size))
        xc, yc = xc.reshape(-1, 1), yc.reshape(-1, 1)

    for i in range(n_objects):
        graph_nodes = np.load(data_folder + "graph_nodes%d.npy" % i).astype(float)
        x = graph_nodes.copy()
        x[:, 2] = (x[:, 2] < 0).astype(float)
        if mesh_folder:
            y = np.load(mesh_folder + "sdf_pxl%d.npy"% i).reshape(-1, 1)
            y = np.concatenate([xc, yc, y], axis=-1)
        else:
            y = graph_nodes.copy()[:, 2]
            y = y.reshape(-1, 1)

        graph_cells = np.load(data_folder + "graph_cells%d.npy" % i).astype(int)
        graph_cells = graph_cells.T
        graph_edges = np.load(data_folder + "graph_edges%d.npy" % i).astype(int)
        if not reversed_edge_already_included:
            graph_edges = add_reversed_edges(graph_edges)
        graph_edges = graph_edges.T
        n_edges = graph_edges.shape[1]
        if edge_weight:
            graph_edge_weights = np.load(data_folder + "graph_weights%d.npy" %i).astype(float)
            if graph_edge_weights.shape[0] == n_edges:
                pass
            elif graph_edge_weights.shape[0] == n_edges // 2:
                graph_edge_weights = np.concatenate([graph_edge_weights, graph_edge_weights])
            else:
                raise("edge weight size is wrong.")
            #graph_edge_weights = graph_edge_weights.reshape(-1, 1)
        else:
            graph_edge_weights = np.ones(n_edges)

        graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                          y=torch.from_numpy(y).type(torch.float32),
                          edge_index=torch.from_numpy(graph_edges).type(torch.long),
                          edge_attr=torch.from_numpy(graph_edge_weights).type(torch.float32),
                          face=torch.from_numpy(graph_cells).type(torch.long))
        graph_data_list.append(graph_data)
    train_data = DataLoader(graph_data_list, batch_size=batch_size)
    return train_data


def add_reversed_edges(edges):
    edges_reversed = np.fliplr(edges)
    edges = np.concatenate([edges, edges_reversed], axis=0)
    return edges


def pixel_to_neighbour(n_pxl=128):
    l1 = np.unravel_index(np.arange(n_pxl**2), (n_pxl, n_pxl))
    neighbours = []
    for i, (irow, icol) in enumerate(l1):
        l2 = []
        if irow > 0:
            l2.append([irow-1, icol])
            if icol > 0:
                l2.append([irow-1, icol-1])
            if icol < n_pxl - 1:
                l2.append([irow-1, icol+1])
        if irow < n_pxl - 1:
            l2.append([irow+1, icol])
            if icol > 0:
                l2.append([irow+1, icol-1])
            if icol < n_pxl - 1:
                l2.append([irow+1, icol+1])
        if icol > 0:
            l2.append([irow, icol-1])
        if icol < n_pxl - 1:
            l2.append([irow, icol+1])
        l2_r = np.ravel_multi_index(l2, (n_pxl, n_pxl))
        neighbour = np.array(np.repeat([i], l2_r.shape[0]))
        neighbour = np.concatenate([neighbour, l2_r], axis=-1)
        neighbours.append(neighbour)
    return neighbours


def get_sdf_data_loader_from_sdf_pixels(n_objects, mesh_folder, batch_size, filter_params=None,
                                        reversed_edge_already_included=False):
    print("preparing sdf data loader")
    sdf_pxl = np.load(mesh_folder + "sdf_pxl0.npy").astype(float)
    pxl_size = 128
    assert sdf_pxl.shape[0] == sdf_pxl.shape[1] == pxl_size
    if filter_params is None:
        filter_params = [2 * np.sqrt(2) / pxl_size + 5e-4]
    xc, yc = np.meshgrid(np.linspace(-1, 1, pxl_size), np.linspace(-1, 1, pxl_size))
    xc, yc = xc.reshape(-1, 1), yc.reshape(-1, 1)
    graph_edges = points_to_neighbours(np.concatenate([xc, yc], axis=-1), filter_params)
    if not reversed_edge_already_included:
        graph_edges = add_reversed_edges(graph_edges)
    graph_edges = graph_edges.T
    n_edges = graph_edges.shape[1]
    graph_edge_weights = np.ones(n_edges)
    graph_data_list = []
    for i in range(n_objects):
        sdf_pxl = np.load(mesh_folder + "sdf_pxl%d.npy" % i).astype(float)
        sdf_pxl = sdf_pxl.reshape(-1, 1)
        img_pxl = (sdf_pxl < 0).astype(float)
        x = np.concatenate([xc, yc, img_pxl], axis=-1)
        y = sdf_pxl
        graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                          y=torch.from_numpy(y).type(torch.float32),
                          edge_index=torch.from_numpy(graph_edges).type(torch.long),
                          edge_attr=torch.from_numpy(graph_edge_weights).type(torch.float32))
        graph_data_list.append(graph_data)
    train_data = DataLoader(graph_data_list, batch_size=batch_size)
    return train_data


# def graph_to_pgdata(xgraph, ygraph, x_keys=("x", "y", "z"), y_keys=("val",)):
#     x = [graph_to_features(xgraph, key) for key in x_keys]
#     x = np.array(x).T
#     y = [graph_to_features(ygraph, key) for key in y_keys]
#     y = np.array(y).T
#     cells = graph_to_cells(graph)
#     edges = cells_to_edges(cells)
#     edges = np.array(edges).T
#     pgdata = Data(x=torch.from_numpy(x),
#                   y=torch.from_numpy(y),
#                   edge_index=torch.from_numpy(edges).type(torch.long))
#     return pgdata


if __name__ == "__main__":
    graph_data_list = []
    folder_name = "mesh_files/data1/"
    for i in range(4):
        geom_vtk_name = folder_name + "geom%d.vtk"%(i+1)
        geom_mesh = vtk_to_mesh(geom_vtk_name)
        geom_graph = mesh_to_graph(geom_mesh)

        sdf_vtk_name = folder_name + "sdf%d.vtk"%(i+1)
        sdf_mesh = vtk_to_mesh(sdf_vtk_name)
        sdf_graph = mesh_to_graph(sdf_mesh)

        plot_mesh(sdf_mesh)
        plot_graph(sdf_graph)
        plot_graph(sdf_graph, color_feature="z")

        x = [graph_to_features(geom_graph, key) for key in ['x', 'y', 'z']]
        x = np.array(x).T
        y = [graph_to_features(sdf_graph, key) for key in ['z']]
        y = np.array(y).T

        assert graph_to_cells(geom_graph) == graph_to_cells(sdf_graph)
        cells = graph_to_cells(geom_graph)
        edges = cells_to_edges(cells)
        edges = np.array(edges).T
        graph_data = Data(x=torch.from_numpy(x).type(torch.float64),
                          y=torch.from_numpy(y).type(torch.float64),
                          edge_index=torch.from_numpy(edges).type(torch.long))

        graph_data_list.append(graph_data)

    #np.save("data/graph_data_list.npy", graph_data_list)
