from tqdm import tqdm
from scipy.spatial import Delaunay
from data2.data_utils import *


class GraphData:
    def __init__(self, graph_node, graph_edge, **kwargs):
        self.graph_node = graph_node
        self.graph_edge = graph_edge
        self.kwargs = kwargs

        self.check_sanity()

    def check_sanity(self):
        if "edge_length" not in self.kwargs:
            self.kwargs["edge_length"] = 1
        # check graph node
        assert self.graph_node in ["vertex", "edge", "cell"], \
            "graph node has to be either of `vertex`, `edge` or `cell` "
        assert self.graph_node != "vertex" or self.graph_edge in ["edge", "neighbour", "knn"], \
            "with `vertex` as graph node, graph edges can only be set to `edge`, `neighbour` or `knn`"
        assert self.graph_node != "edge" or self.graph_edge in ["vertex", "cell"], \
            "with `edge` as graph node, graph edges can only be set to `vertex` or `cell`"
        assert self.graph_node != "cell" or self.graph_edge in ["vertex", "edge", "neighbour"], \
            "width `cell` as graph node, graph edges can only be set to `neighbour`"

        #check graph edge
        assert self.graph_edge != "edge" or "edge_length" in self.kwargs, \
            "if graph edge is `edge`, length of edge (`edge_length`) should be specified"
        assert self.graph_edge != "neighbour" or "filter_type" in self.kwargs, \
            "if graph edge is neighbour, filter type (`filter_type`) should be specified: circular or rectangular"
        assert self.graph_edge != "neighbour" or "radius" in self.kwargs, \
            "if graph edge is neighbour, radius of neighbourhood (`radius`) should be specified"
        assert "radius" not in self.kwargs or isinstance(self.kwargs["radius"], list), \
            "radius should be specified as a list"
        assert self.graph_edge != "knn" or "max_radius" in self.kwargs,\
            "if graph edge is knn, maximum radius of neighbourhood (`max_radius`) should be specified"
        assert self.graph_edge != "knn" or "k" in self.kwargs,\
            "if graph edge is knn, number of nearest neighbours (`k`) should be specified"
        assert self.graph_edge != "neighbour" or "radius" in self.kwargs, \
            "if graph edge is neighbour, radius must be specified"
        assert self.graph_edge != "neighbour" or "filter_type" in self.kwargs, \
            "if graph edge is neighbour, filter type (`filter_type`) should be specified: circular or rectangular"

        # check edge weigth
        assert "edge_weight" not in self.kwargs or self.graph_node == "vertex", \
            "only when graph nodes are `vertex`, edge weight is defined"
        assert "edge_weight" not in self.kwargs or self.kwargs["edge_weight"] in ["length"], \
            "only `length` is supported for edge_weight"

    def write_graph_features_to_file(self, graph_folder):
        if not os.path.isdir(graph_folder):
            os.mkdir(graph_folder)

        with open(graph_folder + "graph_description.txt", 'w') as fid:
            fid.write("This file contains description of the graph.\n")
            fid.write("graph nodes are mesh %s\n" %self.graph_node)
            fid.write("graph edges are mesh %s\n" %self.graph_edge)
            for key, val in self.kwargs.items():
                fid.write(str(key) + " is " + str(val))
                fid.write("\n")

    def read_mesh(self, mesh_file):
        sdf_mesh = meshio.read(mesh_file)
        mesh_nodes = sdf_mesh.points
        mesh_cells = np.array([c.data for c in sdf_mesh.cells if c.type == 'triangle'][0])
        return mesh_nodes, mesh_cells

    def generate_graph_data(self, n_geoms, mesh_folder, graph_folder=None):
        graph_folder = mesh_folder if graph_folder is None else graph_folder
        if not os.path.isdir(graph_folder):
            os.mkdir(graph_folder)
        if self.graph_node == "vertex":
            for i in tqdm(range(n_geoms)):
                self.generate_graph_from_vertices(mesh_folder, graph_folder, name=str(i))
        elif self.graph_node == "edge":
            for i in tqdm(range(n_geoms)):
                self.generate_graph_from_edges(mesh_folder, graph_folder, name=str(i))
        elif self.graph_node == "cell":
            for i in tqdm(range(n_geoms)):
                self.generate_graph_from_cells(mesh_folder, graph_folder, name=str(i))
        else:
            raise(ValueError("node type%s is not supported"%self.graph_node))

    def generate_graph_from_vertices(self, mesh_folder, graph_folder, name=""):
        graph_nodes, graph_cells = self.read_mesh(mesh_folder + "sdf" + name + ".vtk")
        if self.graph_edge == "edge":
            graph_edges = cells_to_edges(graph_cells)
        elif self.graph_edge == "neighbour":
            filter_type = self.kwargs["filter_type"]
            filter_radius = self.kwargs["radius"]
            graph_edges = points_to_neighbours(graph_nodes[:, :2], filter_radius, type=filter_type)
        elif self.graph_edge == "knn":
            filter_k = self.kwargs["k"]
            filter_radius = self.kwargs["max_radius"]
            graph_edges = points_to_knn(graph_nodes[:, :2], filter_k, filter_radius)
        else:
            raise(NotImplementedError())

        edge_length = self.kwargs["edge_length"]
        if edge_length > 1:
            graph_edges = expand_edge_connection(np.array(graph_edges).T, k=edge_length)

        np.save(graph_folder + "graph_nodes" + name + ".npy", graph_nodes)
        np.save(graph_folder + "graph_cells" + name + ".npy", graph_cells)
        np.save(graph_folder + "graph_edges" + name + ".npy", graph_edges)

        if "edge_weight" in self.kwargs:
            edge_weight = self.kwargs["edge_weight"]
            edge_weights = compute_edge_weight(graph_nodes[:, :2], graph_edges, edge_weight)
            np.save(graph_folder + "graph_weights" + name + ".npy", edge_weights)

    def generate_graph_from_edges(self, mesh_folder, graph_folder, name=""):
        mesh_nodes, mesh_cells = self.read_mesh(mesh_folder + "sdf" + name + ".vtk")
        mesh_edges = cells_to_edges(mesh_cells)
        graph_nodes = []
        for (e1, e2) in mesh_edges:
            node = np.concatenate([mesh_nodes[e1, :], mesh_nodes[e2, :]])
            graph_nodes.append(node)

        graph_edges = mesh_to_edge_neighbours(graph_nodes, mesh_edges, mesh_cells, self.graph_edge)
        graph_cells = mesh_cells

        edge_length = self.kwargs["edge_length"]
        if edge_length > 1:
            graph_edges = expand_edge_connection(np.array(graph_edges).T, k=edge_length)

        np.save(graph_folder + "graph_nodes" + name + ".npy", graph_nodes)
        np.save(graph_folder + "graph_cells" + name + ".npy", graph_cells)
        np.save(graph_folder + "graph_edges" + name + ".npy", graph_edges)

    def generate_graph_from_cells(self, mesh_folder, graph_folder, name=""):
        mesh_nodes, mesh_cells = self.read_mesh(mesh_folder + "sdf" + name + ".vtk")
        graph_nodes = []
        for (n1, n2, n3) in mesh_cells:
            node = np.stack((mesh_nodes[n1, :], mesh_nodes[n2, :], mesh_nodes[n3, :]))
            graph_nodes.append(node)
        graph_nodes = np.array(graph_nodes)
        radius = None if "radius" not in self.kwargs else self.kwargs["radius"]
        graph_edges = mesh_to_cell_neighbours(graph_nodes, mesh_cells, method=self.graph_edge, radius=radius)

        edge_length = self.kwargs["edge_length"]
        if edge_length > 1:
            graph_edges = expand_edge_connection(np.array(graph_edges).T, k=edge_length)

        mid_points = np.mean(graph_nodes[:, :, :2], axis=1)
        tri = Delaunay(mid_points)
        graph_cells = tri.simplices.T
        np.save(graph_folder + "graph_nodes" + name + ".npy", graph_nodes)
        np.save(graph_folder + "graph_cells" + name + ".npy", graph_cells)
        np.save(graph_folder + "graph_edges" + name + ".npy", graph_edges)

    # def trim_neighbour(self, graph_folder, n_theta=8, name="", n_edge=1):
    #     assert self.graph_node == "vertex"
    #     theta_c = np.linspace(-np.pi - 1e-3, np.pi + 1e-3, n_theta+1)
    #     graph_nodes = np.load(graph_folder + "graph_nodes" + name + ".npy")
    #     graph_nodes = graph_nodes[:, :2]
    #     graph_edges = np.load(graph_folder + "graph_edges" + name + ".npy")
    #     graph_edges_trimmed = []
    #     for i, node in enumerate(graph_nodes):
    #         edges = [e[1] for e in graph_edges if e[0] == i]
    #         edges += [e[0] for e in graph_edges if e[1] == i]
    #         neighbours = graph_nodes[edges, :]
    #         dist = neighbours - node
    #         theta = np.arctan2(dist[:, 1], dist[:, 0])
    #         for j in range(n_theta):
    #             mask = np.logical_and(theta >= theta_c[j], theta < theta_c[j+1])
    #             neighbours_i = neighbours[mask]
    #             neighbours_i = sorted(neighbours_i, key=lambda x: (x[0] - node[0])**2 + (x[1] - node[1])**2)
    #             for neighbour in neighbours_i[:n_edge]:
    #                 i_edge = edges[np.where(neighbours == neighbour)[0][0]]
    #                 graph_edges_trimmed.append([i, i_edge])
    #
    #     graph_edges_trimmed = np.array(graph_edges_trimmed)
    #     np.save(graph_folder + "graph_edge_trimmed" + name + ".npy", graph_edges_trimmed)
    #     if self.edge_weight is not None:
    #         edge_weights = compute_edge_weight(graph_nodes[:, :2], graph_edges_trimmed, self.edge_weight)
    #         np.save(graph_folder + "graph_weights_trimmed" + name + ".npy", edge_weights)

if __name__ == "__main__":

    n_objects = 50
    batch_size = 1
    data_folder_1 = "mesh_files/vertex_edge_refined/"
    graph_data_1 = GraphData("vertex", "edge", edge_length=1)
    graph_data_1.generate_graph_data(n_objects, data_folder_1)


    data_folder_2 = "mesh_files/vertex_neighbour_refined/"
    graph_data_2 = GraphData("vertex", "neighbour", filter_params=0.1)
    graph_data_2.generate_graph_data(n_objects, data_folder_2)
    get_sdf_data_loader(n_objects, data_folder_2, batch_size, reversed_edge_already_included=False)

