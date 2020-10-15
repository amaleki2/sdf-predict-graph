from data2 import *
from tqdm import tqdm

geometries = ["Rectangle"]
n_objects = 100

root_folder = "data2/dataset_5/"
mesh_folder = root_folder + "mesh/"
mesh_coarse_size, mesh_fine_size, skip_every, refined = 0.06, 0.02, 10, True
# mesh_data = MeshData(geometries, mesh_coarse_size, mesh_fine_size, refined, skip_every, save_sdf_pxl=True)
# mesh_data.write_mesh_features_to_file(mesh_folder)
# for i in tqdm(range(n_objects)):
#     plot = True if i % 10 == 0 else False
#     mesh_data.generate_sdf_mesh(mesh_folder, name=str(i), plot=plot)
#
# graph_folder = root_folder + "graph1/"
# graph_data = GraphData(graph_node="vertex", graph_edge="edge", edge_length=1, edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)
#
# graph_folder = root_folder + "graph2/"
# graph_data = GraphData(graph_node="vertex", graph_edge="edge", edge_length=2, edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)

# graph_folder = root_folder + "graph9/"
# graph_data = GraphData(graph_node="vertex", graph_edge="neighbour", filter_params=[0.1],
#                        filter_type="circular", edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder, edges_trimmed=True)
# graph_data.write_graph_features_to_file(graph_folder)

#
# graph_folder = root_folder + "graph6/"
# graph_data = GraphData(graph_node="vertex", graph_edge="knn", filter_params=[25, 0.3],
#                        filter_type="circular", edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder, edges_trimmed=True)
# graph_data.write_graph_features_to_file(graph_folder)


graph_folder = root_folder + "graph7/"
graph_data = GraphData(graph_node="vertex", graph_edge="knn", filter_params=[40, 0.15],
                       filter_type="circular", edge_weight="length")
graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder, edges_trimmed=True)
graph_data.write_graph_features_to_file(graph_folder)

# graph_folder = root_folder + "graph12/"
# graph_data = GraphData(graph_node="vertex", graph_edge="neighbour", filter_params=[0.05, 0.1],
#                        filter_type="rectangular", edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)

