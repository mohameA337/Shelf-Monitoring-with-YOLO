# utils/data_classes.py

import numpy as np

class Shelf:
    def __init__(self, box):
        points = box.numpy().astype(np.int32).flatten()

        if len(points) == 8:
            self.points = [[int(points[i]), int(points[i+1])] for i in range(0,8,2)]
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.p1 = [min(xs), min(ys)]
            self.p2 = [max(xs), max(ys)]
        else:
            self.p1 = [int(points[0]), int(points[1])]
            self.p2 = [int(points[2]), int(points[3])]
            self.points = [self.p1, [self.p2[0], self.p1[1]], self.p2, [self.p1[0], self.p2[1]]]



class Product:
    def __init__(self, box):
        points = box.numpy().astype(np.int32)
        self.p1 = [points[0], points[1]]
        self.p2 = [points[2], points[3]]
