import sys
from data2.data_utils import *
import matplotlib.pyplot as plt
import meshio

# Todo: I don't like this. fix it
sys.path.insert(1, 'C:/Users/amaleki/OneDrive - ANSYS, Inc/Projects/sdf-prediction/')
from data.geoms import Rectangle, Circle, plot_sdf


class MeshData:
    def __init__(self, geometries, mesh_coarse_size, mesh_fine_size, refined, skip_every):
        self.geometries = geometries
        self.mesh_coarse_size = mesh_coarse_size
        self.mesh_fine_size = mesh_fine_size
        self.refined = refined
        self.skip_points_every = skip_every

    def write_mesh_features_to_file(self, mesh_folder):
        if not os.path.isdir(mesh_folder):
            os.mkdir(mesh_folder)

        with open(mesh_folder + "mesh_description.txt", 'w') as fid:
            fid.write("This file contains description of the mesh\n")
            fid.write("mesh geometris: ")
            for g in self.geometries:
                fid.write(g)
            fid.write("\n")
            fid.write("mesh coarse size: %0.3f\n"%self.mesh_coarse_size)
            fid.write("mesh fine size: %0.3f\n"%self.mesh_fine_size)
            if self.refined:
                fid.write("mesh is refined at boundary with skip parameter: %d"%self.skip_points_every)
            else:
                fid.write("mesh is NOT refined")


    def generate_random_geometries(self):
        img_res = 128
        x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))

        # random parameters of geometry
        l = np.random.random() * 0.5 + 0.2
        w = np.random.random() * 0.5 + 0.2
        r = np.random.random() * np.pi
        tx = np.random.random() * 1.0 - 0.5
        ty = np.random.random() * 1.0 - 0.5

        # create geometry from scipy model
        g = np.random.choice(self.geometries)
        if g == "Rectangle":
            geom = Rectangle((l, w)).rotate(r).translate((tx, ty))
        elif g == "Circle":
            geom = Circle(l).translate((tx, ty))
        else:
            raise(NotImplementedError())

        # evaluate sdf and img
        sdf = geom.eval_sdf(x, y)
        img = sdf < 0

        return geom, img, sdf

    def generate_sdf_mesh(self, data_folder, name="", plot=False, save_sdf_pxl=False):
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)

        geom, img, sdf = self.generate_random_geometries()
        if save_sdf_pxl:
            sdf_pxl_file = data_folder + "sdf_pxl" + name + ".npy"
            np.save(sdf_pxl_file, sdf)

        geo_file = data_folder + "geom" + name + ".geo"
        create_geo_file(img, lc1=self.mesh_coarse_size, lc2=self.mesh_fine_size,
                        skip=self.skip_points_every, geo_file=geo_file, refined=self.refined)
        geo2vtk(geo_file)
        vtk_file = data_folder + "geom" + name + ".vtk"
        mesh = meshio.read(vtk_file)
        nodes_x = mesh.points[:, 0]
        nodes_y = mesh.points[:, 1]
        nodes_sdf = geom.eval_sdf(nodes_x, nodes_y)

        # generate sdf.vtk mesh
        sdf_points = np.array([nodes_x, nodes_y, nodes_sdf]).T
        sdf_cells = np.array([c.data for c in mesh.cells if c.type == 'triangle'][0])

        # save sdf.vtk files
        sdf_mesh = meshio.Mesh(points=sdf_points, cells=[('triangle', sdf_cells)])
        meshio.write(data_folder + "sdf" + name + ".vtk", sdf_mesh)

        # plot
        if plot:
            plot_sdf(img, sdf, show=False)
            plt.subplot(2, 2, 3)
            plot_mesh(sdf_mesh)
            plot_mesh(sdf_mesh, vals=(nodes_sdf < 0).astype(int))
            plt.subplot(2, 2, 4)
            plot_mesh(sdf_mesh, vals=nodes_sdf, with_colorbar=True)
            plt.show()


