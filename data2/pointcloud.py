from scipy.spatial import Delaunay


class PointCloudData:
    def __init__(self):

def tovtk(points):
    tri = Delaunay(points)
    simplices = tri.simplices
    with open("geom.vtk", "w") as fid:
        fid.write("# vtk DataFile Version 2.0\n")
        fid.write("geom, Created by Gmsh\n")
        fid.write("ASCII\n")
        fid.write("DATASET UNSTRUCTURED_GRID\n")
        n_points = points.shape[0]
        fid.write("POINTS %d double\n" % n_points)

