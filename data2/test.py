import numpy as np
from data2 import GraphData

data_folder_test = "test/"
graph_data_test = GraphData("vertex", "edge", edge_length=1)
graph_data_test.generate_graph_data(1, data_folder_test)
nodes = np.load(data_folder_test + "graph_nodes0.npy")
cells = np.load(data_folder_test + "graph_cells0.npy")
edges = np.load(data_folder_test + "graph_edges0.npy")

cells_sets = [set(x) for x in cells]
assert len(cells_sets) == 14
assert {3, 6, 11} in cells_sets
assert {2, 6, 10} in cells_sets
assert {3, 7, 11} in cells_sets
assert {6, 10, 11} in cells_sets
assert {9, 7, 11} in cells_sets
assert {9, 10, 11} in cells_sets
assert {10, 9, 8} in cells_sets
assert {10, 8, 5} in cells_sets
assert {10, 2, 5} in cells_sets
assert {5, 8, 1} in cells_sets
assert {8, 1, 4} in cells_sets
assert {8, 9, 4} in cells_sets
assert {0, 9, 4} in cells_sets
assert {7, 9, 0} in cells_sets

edges_set = [set(x) for x in edges]
assert len(edges_set) == 25
assert {0, 4} in edges_set
assert {0, 7} in edges_set
assert {0, 9} in edges_set
assert {1, 4} in edges_set
assert {1, 5} in edges_set
assert {1, 8} in edges_set
assert {2, 5} in edges_set
assert {2, 6} in edges_set
assert {2, 10} in edges_set
assert {3, 6} in edges_set
assert {3, 7} in edges_set
assert {3, 11} in edges_set
assert {4, 8} in edges_set
assert {4, 9} in edges_set
assert {5, 8} in edges_set
assert {5, 10} in edges_set
assert {6, 10} in edges_set
assert {6, 11} in edges_set
assert {7, 9} in edges_set
assert {7, 11} in edges_set
assert {8, 9} in edges_set
assert {8, 10} in edges_set
assert {9, 10} in edges_set
assert {9, 11} in edges_set
assert {10, 11} in edges_set

graph_data_test = GraphData("vertex", "edge", edge_length=2)
graph_data_test.generate_graph_data(1, data_folder_test)
edges = np.load(data_folder_test + "graph_edges0.npy")

edges_set = [set(x) for x in edges]
assert len(edges_set) == 52
assert {0, 4} in edges_set
assert {0, 7} in edges_set
assert {0, 9} in edges_set
assert {1, 4} in edges_set
assert {1, 5} in edges_set
assert {1, 8} in edges_set
assert {2, 5} in edges_set
assert {2, 6} in edges_set
assert {2, 10} in edges_set
assert {3, 6} in edges_set
assert {3, 7} in edges_set
assert {3, 11} in edges_set
assert {4, 8} in edges_set
assert {4, 9} in edges_set
assert {5, 8} in edges_set
assert {5, 10} in edges_set
assert {6, 10} in edges_set
assert {6, 11} in edges_set
assert {7, 9} in edges_set
assert {7, 11} in edges_set
assert {8, 9} in edges_set
assert {8, 10} in edges_set
assert {9, 10} in edges_set
assert {9, 11} in edges_set
assert {10, 11} in edges_set
assert {0, 1} in edges_set
assert {0, 3} in edges_set
assert {0, 8} in edges_set
assert {0, 10} in edges_set
assert {0, 11} in edges_set
assert {1, 2} in edges_set
assert {1, 9} in edges_set
assert {1, 10} in edges_set
assert {2, 3} in edges_set
assert {2, 11} in edges_set
assert {2, 8} in edges_set
assert {2, 9} in edges_set
assert {3, 9} in edges_set
assert {3, 10} in edges_set
assert {4, 5} in edges_set
assert {4, 7} in edges_set
assert {4, 10} in edges_set
assert {4, 11} in edges_set
assert {5, 6} in edges_set
assert {5, 9} in edges_set
assert {5, 11} in edges_set
assert {6, 7} in edges_set
assert {6, 8} in edges_set
assert {6, 9} in edges_set
assert {7, 8} in edges_set
assert {7, 10} in edges_set
assert {8, 11} in edges_set

graph_data_test = GraphData("vertex", "neighbour", circle_radius=0.5)
graph_data_test.generate_graph_data(1, data_folder_test)
edges = np.load(data_folder_test + "graph_edges0.npy")
assert edges.shape == (0, 2)

graph_data_test = GraphData("vertex", "neighbour", circle_radius=1.5)
graph_data_test.generate_graph_data(1, data_folder_test)
edges = np.load(data_folder_test + "graph_edges0.npy")

edges_set = [set(x) for x in edges]
assert len(edges_set) == 38
assert {0, 4} in edges_set
assert {0, 7} in edges_set
assert {0, 9} in edges_set
assert {1, 4} in edges_set
assert {1, 5} in edges_set
assert {1, 8} in edges_set
assert {2, 5} in edges_set
assert {2, 6} in edges_set
assert {2, 10} in edges_set
assert {3, 6} in edges_set
assert {3, 7} in edges_set
assert {3, 11} in edges_set
assert {4, 8} in edges_set
assert {4, 9} in edges_set
assert {5, 8} in edges_set
assert {5, 10} in edges_set
assert {6, 10} in edges_set
assert {6, 11} in edges_set
assert {7, 9} in edges_set
assert {7, 11} in edges_set
assert {8, 9} in edges_set
assert {8, 10} in edges_set
assert {9, 10} in edges_set
assert {9, 11} in edges_set
assert {10, 11} in edges_set

assert {1, 9} in edges_set
assert {1, 10} in edges_set
assert {3, 9} in edges_set
assert {3, 10} in edges_set
assert {4, 5} in edges_set
assert {4, 7} in edges_set
assert {4, 10} in edges_set
assert {5, 6} in edges_set
assert {5, 9} in edges_set
assert {6, 7} in edges_set
assert {6, 9} in edges_set
assert {7, 10} in edges_set
assert {8, 11} in edges_set


graph_data_test = GraphData("edge", "cell", edge_length=1)
graph_data_test.generate_graph_data(1, data_folder_test)
nodes = np.load(data_folder_test + "graph_nodes0.npy")
cells = np.load(data_folder_test + "graph_cells0.npy")
edges = np.load(data_folder_test + "graph_edges0.npy")

edges_set = [set(x) for x in edges]
assert len(edges_set) == 42
assert {0, 1} in edges_set
assert {0, 2} in edges_set
assert {1, 2} in edges_set
assert {1, 12} in edges_set
assert {1, 20} in edges_set
assert {2, 8} in edges_set
assert {2, 9} in edges_set
assert {3, 4} in edges_set
assert {3, 5} in edges_set
assert {4, 5} in edges_set
assert {4, 6} in edges_set
assert {4, 7} in edges_set
assert {5, 14} in edges_set
assert {5, 24} in edges_set
assert {6, 7} in edges_set
assert {7, 17} in edges_set
assert {7, 21} in edges_set
assert {8, 9} in edges_set
assert {9, 19} in edges_set
assert {9, 22} in edges_set
assert {10, 11} in edges_set
assert {10, 12} in edges_set
assert {11, 12} in edges_set
assert {11, 13} in edges_set
assert {11, 14} in edges_set
assert {12, 20} in edges_set
assert {13, 14} in edges_set
assert {14, 24} in edges_set
assert {15, 16} in edges_set
assert {15, 17} in edges_set
assert {16, 17} in edges_set
assert {16, 18} in edges_set
assert {16, 19} in edges_set
assert {17, 21} in edges_set
assert {18, 19} in edges_set
assert {19, 22} in edges_set
assert {20, 23} in edges_set
assert {20, 24} in edges_set
assert {21, 22} in edges_set
assert {21, 23} in edges_set
assert {22, 23} in edges_set
assert {23, 24} in edges_set

print("all tests passed!")