from data2 import *
from tqdm import tqdm

n_objects = 100

root_folder = "data2/dataset_3/"
mesh_folder = root_folder + "mesh/"

# geometries = ["Rectangle", "Circle", "nGon", "Diamond"]
# mesh_coarse_size, mesh_fine_size, skip_every, refined = 0.06, 0.03, 5, False
# mesh_data = MeshData(geometries, mesh_coarse_size, mesh_fine_size, refined, skip_every, save_sdf_pxl=True)
# mesh_data.write_mesh_features_to_file(mesh_folder)
# for i in tqdm(range(n_objects)):
#     plot = True if i < 50 and i % 4 == 0 else False
#     geom = mesh_data.generate_sdf_mesh(mesh_folder, name=str(i), plot=plot)
#     mesh_data.generated_augmented_sdf_mesh(geom, mesh_folder, name=str(i+n_objects))

graph_folder = root_folder + "graph11/"
graph_data = GraphData(graph_node="cell", graph_edge="neighbour", radius=[0.062], filter_type="circular")
#graph_data.generate_graph_data(n_objects, mesh_folder, graph_folder=graph_folder)
graph_data.write_graph_features_to_file(graph_folder)
