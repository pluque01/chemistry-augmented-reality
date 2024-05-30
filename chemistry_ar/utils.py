import numpy as np


def circumference_points(radius: float, n_points: int):
    points = []
    for i in range(n_points):
        coords = np.array([0.0, 0.0, 0.0])
        angle = 2 * np.pi * i / n_points
        coords[0] += radius * np.cos(angle)
        coords[1] -= radius * np.sin(angle)
        points.append(coords)
    return np.asarray(points)
