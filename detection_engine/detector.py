import darknet
import cv2


class Detector:

    def __init__(config_file, data_file, weights, batch_size):
    network, class_names, class_colors = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=batch_size
        )
    self.width = darknet.network_width(network)
    height = darknet.network_height(network)