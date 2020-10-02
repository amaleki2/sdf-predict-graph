from tqdm import tqdm
from data2.data_utils import *


class GraphData:
    def __init__(self, graph_node, graph_edge, circle_radius=None, edge_length=None):
        self.graph_node = graph_node
        self.graph_edge = graph_edge
        self.circle_radius = circle_radius
        self.edge_length = edge_length
        self.check_sanity()

    def write_graph_features_to_file(self, graph_folder):
        if not os.path.isdir(graph_folder):
            os.mkdir(graph_folder)

        with open(graph_folder + "graph_description.txt", 'w') as fid:
            fid.write("This file contains description of the graph.\n")
            fid.write("graph nodes are mesh %s\n" %self.graph_node)
            fid.write("graph edges are mesh %s\n" %self.graph_edge)
            if self.edge_length:
                fid.write("edge of length %d are considered as graph edge"%self.edge_length)

            if self.circle_radius:
                fid.write("radius of neighbourhood is %0.3f" % self.circle_radius)

    def check_sanity(self):
        if self.graph_node == "vertex":
            assert self.graph_edge in ["edge", "neighbour"], \
                "with `vertex` as graph node, graph edges can only be set to `edge` or `neighbour`"
            assert self.graph_edge != "edge" or self.edge_length, \
                "if graph edge is `edge`, length of edge (`edge_length`) should be determined"
            assert self.graph_edge != "neighbour" or self.circle_radius, \
                "if graph edge is neighbour, circle of neighbourhood (`cirlce_radius`) should be determined"
        elif self.graph_node == "edge":
            assert self.graph_edge in ["vertex", "cell"], \
                "with `edge` as graph node, graph edges can only be set to `vertex` or `cell`"
        else:
            raise(IOError("graph node has to be `vertex` or `edge`."))

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
        else:
            raise(NotImplementedError("Not implemented."))

    def generate_graph_from_vertices(self, mesh_folder, graph_folder, name=""):
        sdf_mesh = meshio.read(mesh_folder + "sdf" + name + ".vtk")
        graph_nodes = sdf_mesh.points
        graph_cells = np.array([c.data for c in sdf_mesh.cells if c.type == 'triangle'][0])
        if self.graph_edge == "edge":
            graph_edges = cells_to_edges(graph_cells)
            if self.edge_length > 1:
                graph_edges = expand_edge_connection(np.array(graph_edges).T, k=2)
        elif self.graph_edge == "neighbour":
            graph_edges = points_to_neighbours(graph_nodes[:, :2], self.circle_radius)
        else:
            raise(NotImplementedError())

        np.save(graph_folder + "graph_nodes" + name + ".npy", graph_nodes)
        np.save(graph_folder + "graph_cells" + name + ".npy", graph_cells)
        np.save(graph_folder + "graph_edges" + name + ".npy", graph_edges)

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




#if __name__ == "__main__":

    # n_objects = 50
    # batch_size = 1
    # data_folder_1 = "mesh_files/vertex_edge_refined/"
    # graph_data_1 = GraphData("vertex", "edge", edge_length=1)
    # graph_data_1.generate_graph_data(n_objects, data_folder_1)


    # data_folder_2 = "mesh_files/vertex_neighbour_refined/"
    # graph_data_2 = GraphData("vertex", "neighbour", circle_radius=0.1)
    # graph_data_2.generate_graph_data(n_objects, data_folder_2)
    #get_sdf_data_loader(n_objects, data_folder_2, batch_size, reversed_edge_already_included=False)

