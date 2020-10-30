import sys
from data2.data_utils import *
import matplotlib.pyplot as plt
import meshio

# Todo: I don't like this. fix it
import os
if os.name == 'posix':
    sys.path.insert(1, '/home/ansysai/amaleki/sdf-prediction/')
else:
    sys.path.insert(1, 'C:/Users/amaleki/OneDrive - ANSYS, Inc/Projects/sdf-prediction/')
from data.geoms import Rectangle, Circle, nGon, Diamond, plot_sdf


class MeshData:
    _img_res = 128
    _x, _y = np.meshgrid(np.linspace(-1, 1, _img_res), np.linspace(-1, 1, _img_res))

    def __init__(self, geometries, mesh_coarse_size, mesh_fine_size, refined, skip_every, save_sdf_pxl=False):
        self.geometries = geometries
        self.mesh_coarse_size = mesh_coarse_size
        self.mesh_fine_size = mesh_fine_size
        self.refined = refined
        self.skip_points_every = skip_every
        self.save_sdf_pxl = save_sdf_pxl

    def write_mesh_features_to_file(self, mesh_folder):
        if not os.path.isdir(mesh_folder):
            os.mkdir(mesh_folder)

        with open(mesh_folder + "mesh_description.txt", 'w') as fid:
            fid.write("This file contains description of the mesh\n")
            fid.write("mesh geometris: ")
            for g in self.geometries: fid.write(g)
            fid.write("\n")
            fid.write("mesh coarse size: %0.3f\n"%self.mesh_coarse_size)
            fid.write("mesh fine size: %0.3f\n"%self.mesh_fine_size)
            if self.refined: fid.write("mesh is refined at boundary with skip parameter: %d\n"%self.skip_points_every)

    def generate_random_geometries(self):
        # create geometry from scipy model
        g = np.random.choice(self.geometries)
        if g == "Circle":
            l = np.random.random() * 0.5 + 0.2
            geom = Circle(l)
        elif g == "Rectangle":
            l = np.random.random() * 0.5 + 0.2
            w = np.random.random() * 0.5 + 0.2
            geom = Rectangle((l, w))
        elif g == "nGon":
            n_vertices = 5
            vertices = []
            for i in range(n_vertices):
                th = np.random.random() * np.pi / n_vertices + 2 * i * np.pi / n_vertices
                ra = np.random.random() * 0.5 + 0.2
                vertices.append([ra * np.cos(th), ra * np.sin(th)])
            geom = nGon(vertices)
        elif g == "Diamond":
            l = np.random.random() * 0.5 + 0.2
            w = np.random.random() * 0.5 + 0.2
            geom = Diamond((l, w))
        else:
            raise(ValueError("geometry %s not recognized"%g))

        # randomly rotate and translate geometry
        r = np.random.random() * np.pi
        tx = np.random.random() * 1.0 - 0.5
        ty = np.random.random() * 1.0 - 0.5

        geom = geom.rotate(r).translate((tx, ty))

        # evaluate sdf and img
        sdf = geom.eval_sdf(self._x, self._y)
        img = sdf < 0

        return geom, img, sdf

    def generate_sdf_mesh(self, data_folder, name="", plot=False):
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)

        geom, img, sdf = self.generate_random_geometries()
        if self.save_sdf_pxl:
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

        return geom

    def generated_augmented_sdf_mesh(self, geom, data_folder, name=""):
        if np.random.random() < 0.5:
            r = np.random.random() * 2 * np.pi
            geom = geom.rotate(r)
        else:
            tx = np.random.random() - 0.5
            ty = np.random.random() - 0.5
            geom = geom.translate((tx, ty))

        # evaluate sdf and img
        sdf = geom.eval_sdf(self._x, self._y)
        img = sdf < 0
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