from data2 import *
from tqdm import tqdm

# geometries = ["Rectangle"]
n_objects = 100
#
mesh_folder = "data2/dataset_1/mesh/"
# mesh_coarse_size, mesh_fine_size, skip_every, refined = 0.06, 0.02, 10, True
# mesh_data = MeshData(geometries, mesh_coarse_size, mesh_fine_size, refined, skip_every)
# mesh_data.write_mesh_features_to_file(mesh_folder)
# for i in tqdm(range(n_objects)):
#     plot = True if i % 10 == 0 else False
#     mesh_data.generate_sdf_mesh(mesh_folder, name=str(i), plot=plot)
#
# graph_folder = "data2/dataset_1/graph1/"
# graph_data = GraphData("vertex", "edge", edge_length=1)
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)
#
# graph_folder = "data2/dataset_1/graph2/"
# graph_data = GraphData("vertex", "edge", edge_length=2)
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)
#
# graph_folder = "data2/dataset_1/graph3/"
# graph_data = GraphData("vertex", "neighbour", circle_radius=mesh_coarse_size*1.5)
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)
#
# graph_folder = "data2/dataset_1/graph4/"
# graph_data = GraphData("vertex", "neighbour", circle_radius=mesh_coarse_size*1.05)
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)

# graph_folder = "data2/dataset_1/graph5/"
# graph_data = GraphData("edge", "cell")
# graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
# graph_data.write_graph_features_to_file(graph_folder)

graph_folder = "data2/dataset_1/graph6/"
graph_data = GraphData("edge", "vertex")
graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
graph_data.write_graph_features_to_file(graph_folder)
