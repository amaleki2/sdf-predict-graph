import meshio
from data2.data_utils import plot_mesh
import matplotlib.pyplot as plt

n_objects = 50
mesh_folder = "dataset_3/mesh/"
for i in range(n_objects):
    mesh_file = mesh_folder + "geom" + str(i) + ".vtk"
    mesh = meshio.read(mesh_file)
    plot_mesh(mesh)
    plt.show()
    # plot_mesh(mesh, vals=(nodes_sdf < 0).astype(int))
    # plt.subplot(2, 2, 4)
    # plot_mesh(mesh, vals=nodes_sdf, with_colorbar=True)
    # plt.show()
