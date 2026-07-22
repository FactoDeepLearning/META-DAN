import os.path

import matplotlib.pyplot as plt
import json


def generate_graph_result(file_paths, metric, x):
    values = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            values.append(json.load(f)[metric]*100)

    plt.subplots()
    plt.plot(x, values)
    plt.title("Evolution of the CER with respect to the decoding threshold")
    plt.xlabel("Threshold")
    plt.ylabel("CER (%)")
    plt.tight_layout()

    dir_path = os.path.dirname(file_paths[0])
    save_path = os.path.join(dir_path, "visualization_impact_threshold.png")

    plt.savefig(save_path)


