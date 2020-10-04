from data2 import *
from tqdm import tqdm

geometries = ["Rectangle"]
n_objects = 100

root_folder = "data2/dataset_2/"
mesh_folder = root_folder + "mesh/"
mesh_coarse_size, mesh_fine_size, skip_every, refined = 0.03, 0.01, 10, True
# mesh_data = MeshData(geometries, mesh_coarse_size, mesh_fine_size, refined, skip_every)
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
#
# graph_folder = root_folder + "graph3/"
# graph_data = GraphData(graph_node="vertex", graph_edge="neighbour", circle_radius=mesh_coarse_size*1.05, edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)
#
# graph_folder = root_folder + "graph4/"
# graph_data = GraphData(graph_node="vertex", graph_edge="neighbour", circle_radius=mesh_coarse_size*1.55, edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)
#
# graph_folder = root_folder + "graph5/"
# graph_data = GraphData(graph_node="vertex", graph_edge="neighbour", circle_radius=mesh_coarse_size*2.05, edge_weight="length")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)


graph_folder = root_folder + "graph6/"
graph_data = GraphData(graph_node="vertex", graph_edge="neighbour", circle_radius=mesh_coarse_size*3.05, edge_weight="length")
graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
graph_data.write_graph_features_to_file(graph_folder)
