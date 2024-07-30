import numpy as np
from numpy import linalg as LA

class SingleCamera:
    def __init__(self, world_coor, pixel_coor, n):
        self.world_coor = world_coor
        self.pixel_coor = pixel_coor
        self.point_num = n

        self.P = np.empty([self.point_num * 2, 12], dtype=float)
        self.M = np.empty([3, 4], dtype=float)
        self.A = np.empty([3, 3], dtype=float)
        self.b = np.empty([3, 1], dtype=float)
        self.K = np.empty([3, 3], dtype=float)
        self.R = np.empty([3, 3], dtype=float)
        self.t = np.empty([3, 1], dtype=float)

    def composeP(self):
        i = 0
        P = np.empty([self.point_num * 2, 12], dtype=float)
        while i < self.point_num * 2:
            c = i // 2
            p1 = self.world_coor[c]
            p2 = np.array([0, 0, 0, 0])
            if i % 2 == 0:
                p3 = -p1 * self.pixel_coor[c][0]
                P[i] = np.hstack((p1, p2, p3))
            else:
                p3 = -p1 * self.pixel_coor[c][1]
                P[i] = np.hstack((p2, p1, p3))
            i += 1
        self.P = P

    def svdP(self):
        U, sigma, VT = LA.svd(self.P)
        V = np.transpose(VT)
        preM = V[:, -1]
        
