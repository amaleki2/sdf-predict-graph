from tqdm import tqdm
from data2.data_utils import *


class GraphData:
    def __init__(self, graph_node, graph_edge, filter_params=None, edge_length=None, edge_weight=None, filter_type=None):
        self.graph_node = graph_node
        self.graph_edge = graph_edge
        self.filter_params = filter_params
        self.filter_type = filter_type
        self.edge_length = edge_length
        self.edge_weight = edge_weight
        self.check_sanity()

    def write_graph_features_to_file(self, graph_folder):
        if not os.path.isdir(graph_folder):
            os.mkdir(graph_folder)

        with open(graph_folder + "graph_description.txt", 'w') as fid:
            fid.write("This file contains description of the graph.\n")
            fid.write("graph nodes are mesh %s\n" %self.graph_node)
            fid.write("graph edges are mesh %s\n" %self.graph_edge)
            if self.edge_length:
                fid.write("edge of length %d are considered as graph edge\n"%self.edge_length)

            if self.filter_params:
                if self.filter_type == "circular":
                    fid.write("circular filter with radius of %0.3f\n" % self.filter_params[0])
                elif self.filter_type == "rectangular":
                    fid.write("rectangular filter with length and width %0.3f, %0.3f\n" % (self.filter_params[0], self.filter_params[1]))
                elif self.filter_type == "knn":
                    fid.write("knn filter with k = %d, max radius of %0.3f\n" % (self.filter_params[0], self.filter_params[1]))

            if self.edge_weight:
                fid.write("edge weight is computed as %s\n" % self.edge_weight)

    def check_sanity(self):
        if self.graph_node == "vertex":
            assert self.graph_edge in ["edge", "neighbour", "knn"], \
                "with `vertex` as graph node, graph edges can only be set to `edge`, `neighbour` or `knn`"
            assert self.graph_edge != "edge" or self.edge_length, \
                "if graph edge is `edge`, length of edge (`edge_length`) should be determined"
            assert self.graph_edge != "neighbour" or self.filter_params, \
                "if graph edge is neighbour, filter_params should be determined"
            assert self.graph_edge != "knn" or self.filter_params, \
                "if graph edge is knn, filter_params should be determined"
        elif self.graph_node == "edge":
            assert self.graph_edge in ["vertex", "cell"], \
                "with `edge` as graph node, graph edges can only be set to `vertex` or `cell`"
        else:
            raise(IOError("graph node has to be `vertex` or `edge`."))

        assert self.edge_weight is None or self.graph_node == "vertex", \
            "only when graph nodes are `vertex`, edge weight is defined"
        assert self.edge_weight is None or self.edge_weight in ["length"], \
            "only `length` is supported for edge_weight"

    def generate_graph_data(self, n_geoms, mesh_folder, graph_folder=None, edges_trimmed=False):
        graph_folder = mesh_folder if graph_folder is None else graph_folder
        if not os.path.isdir(graph_folder):
            os.mkdir(graph_folder)
        if self.graph_node == "vertex":
            for i in tqdm(range(n_geoms)):
                self.generate_graph_from_vertices(mesh_folder, graph_folder, name=str(i))

        elif self.graph_node == "edge":
            for i in tqdm(range(n_geoms)):
                self.generate_graph_from_edges(mesh_folder, graph_folder, name=str(i))
        else:
            raise(NotImplementedError("Not implemented."))

    def generate_graph_from_vertices(self, mesh_folder, graph_folder, name=""):
        sdf_mesh = meshio.read(mesh_folder + "sdf" + name + ".vtk")
        graph_nodes = sdf_mesh.points
        graph_cells = np.array([c.data for c in sdf_mesh.cells if c.type == 'triangle'][0])
        if self.graph_edge == "edge":
            graph_edges = cells_to_edges(graph_cells)
            if self.edge_length > 1:
                graph_edges = expand_edge_connection(np.array(graph_edges).T, k=self.edge_length)
        elif self.graph_edge == "neighbour":
            graph_edges = points_to_neighbours(graph_nodes[:, :2], self.filter_params, type=self.filter_type)
        elif self.graph_edge == "knn":
            graph_edges = points_to_knn(graph_nodes[:, :2], *self.filter_params)
        else:
            raise(NotImplementedError())

        np.save(graph_folder + "graph_nodes" + name + ".npy", graph_nodes)
        np.save(graph_folder + "graph_cells" + name + ".npy", graph_cells)
        np.save(graph_folder + "graph_edges" + name + ".npy", graph_edges)

        if self.edge_weight is not None:
            edge_weights = compute_edge_weight(graph_nodes[:, :2], graph_edges, self.edge_weight)
            np.save(graph_folder + "graph_weights" + name + ".npy", edge_weights)

    def generate_graph_from_edges(self, mesh_folder, graph_folder, name=""):
        sdf_mesh = meshio.read(mesh_folder + "sdf" + name + ".vtk")
        mesh_nodes = sdf_mesh.points
        mesh_cells = np.array([c.data for c in sdf_mesh.cells if c.type == 'triangle'][0])
        mesh_edges = cells_to_edges(mesh_cells)
        graph_nodes = []
        for (e1, e2) in mesh_edges:
            node = np.concatenate([mesh_nodes[e1, :], mesh_nodes[e2, :]])
            graph_nodes.append(node)

        graph_edges = mesh_to_edge_neighbours(graph_nodes, mesh_edges, mesh_cells, self.graph_edge)
        graph_cells = mesh_cells

        np.save(graph_folder + "graph_nodes" + name + ".npy", graph_nodes)
        np.save(graph_folder + "graph_cells" + name + ".npy", graph_cells)
        np.save(graph_folder + "graph_edges" + name + ".npy", graph_edges)

    def trim_neighbour(self, graph_folder, n_theta=8, name="", n_edge=1):
        assert self.graph_node == "vertex"
        theta_c = np.linspace(-np.pi - 1e-3, np.pi + 1e-3, n_theta+1)
        graph_nodes = np.load(graph_folder + "graph_nodes" + name + ".npy")
        graph_nodes = graph_nodes[:, :2]
        graph_edges = np.load(graph_folder + "graph_edges" + name + ".npy")
        graph_edges_trimmed = []
        for i, node in enumerate(graph_nodes):
            edges = [e[1] for e in graph_edges if e[0] == i]
            edges += [e[0] for e in graph_edges if e[1] == i]
            neighbours = graph_nodes[edges, :]
            dist = neighbours - node
            theta = np.arctan2(dist[:, 1], dist[:, 0])
            for j in range(n_theta):
                mask = np.logical_and(theta >= theta_c[j], theta < theta_c[j+1])
                neighbours_i = neighbours[mask]
                neighbours_i = sorted(neighbours_i, key=lambda x: (x[0] - node[0])**2 + (x[1] - node[1])**2)
                for neighbour in neighbours_i[:n_edge]:
                    i_edge = edges[np.where(neighbours == neighbour)[0][0]]
                    graph_edges_trimmed.append([i, i_edge])

        graph_edges_trimmed = np.array(graph_edges_trimmed)
        np.save(graph_folder + "graph_edge_trimmed" + name + ".npy", graph_edges_trimmed)
        if self.edge_weight is not None:
            edge_weights = compute_edge_weight(graph_nodes[:, :2], graph_edges_trimmed, self.edge_weight)
            np.save(graph_folder + "graph_weights_trimmed" + name + ".npy", edge_weights)

#if __name__ == "__main__":

    # n_objects = 50
    # batch_size = 1
    # data_folder_1 = "mesh_files/vertex_edge_refined/"
    # graph_data_1 = GraphData("vertex", "edge", edge_length=1)
    # graph_data_1.generate_graph_data(n_objects, data_folder_1)


    # data_folder_2 = "mesh_files/vertex_neighbour_refined/"
    # graph_data_2 = GraphData("vertex", "neighbour", filter_params=0.1)
    # graph_data_2.generate_graph_data(n_objects, data_folder_2)
    #get_sdf_data_loader(n_objects, data_folder_2, batch_size, reversed_edge_already_included=False)

