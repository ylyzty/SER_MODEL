import os
import glob
import platform

import numpy as np


class Node:
    """
    Graph node define
    """

    def __init__(self, node_id: int = -1, node_lit: int = -1, node_type: int = -1, node_ser: float = 0.0):
        self.node_id = node_id
        self.node_lit = node_lit
        self.node_type = node_type
        self.node_ser = node_ser


def parse_circuit_data(data_dir: str):
    if os.path.isfile(data_dir):
        data_files = data_dir
    elif os.path.isdir(data_dir):
        data_files = data_dir + "/*.txt"
    else:
        raise KeyError("Error: circuit data file or directory does not exist!")

    graphs = {}
    circuit_idx = 0
    for file in glob.glob(data_files):
        if platform.system() == "Windows":
            name = file.split("\\")[-1].split(".")[0]
        else:
            name = file.split("/")[-1].split(".")[0]

        # print("." * 10)
        # print(f"No. {++circuit_idx}  Circuit Name: {name}")

        # Read circuit data file
        with open(file, 'r') as f:
            circuit_data = f.readlines()

        graph_nodes = []
        edge_index_data = []
        edge_feat_data = []
        for line_idx, line_data in enumerate(circuit_data):
            tokens = line_data.strip().split(" ")
            if len(tokens) == 4:
                node = Node(int(tokens[0]), int(tokens[1]), int(tokens[2]))
                if tokens[3] != "/":
                    node.node_ser = float(tokens[3])
                graph_nodes.append(node)

            elif len(tokens) == 3:
                edge_index_data.append([int(tokens[0]), int(tokens[1])])
                edge_feat_data.append([int(tokens[2])])

            else:
                raise KeyError("Error: the number of tokens in line data is wrong!")

        x = [[node.node_id, node.node_type] for node in graph_nodes]
        y = [[node.node_id, node.node_ser] for node in graph_nodes]

        graphs[name] = {'x': np.array(x).astype('float32'),
                        'edge_index': np.array(edge_index_data),
                        'edge_feat': np.array(edge_feat_data),
                        'y': np.array(y).astype('float32')}

    return graphs


if __name__ == "__main__":
    path = "../../circuit_data/raw"
    res = os.listdir(path)
    print(type(res[0]))
    # graphs = parse_circuit_data(path)
    # print(graphs)
