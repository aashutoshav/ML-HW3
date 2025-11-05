import os
import unittest

import numpy as np
from matplotlib import pyplot as plt


class EF_Test(unittest.TestCase):

    def __init__(self):
        images = []
        for filename in sorted(os.listdir("./data/faces")):
            img_path = os.path.join("./data/faces", filename)
            image = plt.imread(img_path)
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            images.append(image.flatten())
        self.faces = np.array(images)
        img_path = "./data/faces/subject01.centerlight.png"
        image = plt.imread(img_path)
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        self.single_image = image
        self.svd_U_sum = 5.477225575051667
        self.svd_S_sum = 1084.0169462365868
        self.svd_V_sum = -357.86692337533333
        self.shape_of_U = 30, 30
        self.shape_of_S = (30,)
        self.shape_of_V = 30, 75625
        self.Uc_sum = 1.3114509478384662e-15
        self.Sc_sum = 360.53864183860935
        self.Vc_sum = -352.5763241228769
        self.shape_of_Uc = 30, 2
        self.shape_of_Sc = (2,)
        self.shape_of_Vc = 2, 75625
        self.rebuilt_sum = -2.3305801732931286e-11
        self.shape_of_rebuilt = 30, 75625
        self.compression_ratio_2 = 0.014571900826446282
        self.rvp_2 = 0.7202800285733673
        self.eigenfaces_sum = -352.5763241228769
        self.eigenfaces_shape = 2, 75625
        self.projections = np.array(
            [
                [-4815.80772585, -5671.00274212],
                [4085.22577218, 3538.47250329],
                [730.58195367, 2132.53023883],
            ]
        )


if __name__ == "__main__":
    unittest.main()
