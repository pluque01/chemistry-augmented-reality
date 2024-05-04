import numpy as np
from typing import List, Dict
from shapes.sphere import Sphere
import camera
import toml

atom_data = toml.load("chemistry_ar/data/atoms.toml")


class Atom(Sphere):
    def __init__(self, ctx, name: str, radius: float, projection_matrix):
        super().__init__(ctx, radius, projection_matrix)
        self.name = name

    def render(self, view_matrix):
        super().render(view_matrix)


class Molecule:
    def __init__(
        self,
        ctx,
        atoms: List[str],
        aruco: int,
        position: np.ndarray,
        projection_matrix,
    ):
        self.ctx = ctx
        self.aruco = aruco
        self.projection_matrix = projection_matrix
        self.atoms = []

        # Movement related
        self.ACCELERATION = 2
        self.position = position

        for atom in atoms:
            radius = registered_atoms[atom]
            self.atoms.append(Atom(ctx, atom, radius, projection_matrix))

    def render(self, view_matrix: np.ndarray, frame_time: float):
        # Get the 4th column of the view matrix
        aruco_position = camera.extrinsic2Position(view_matrix)
        for i in range(len(self.position)):
            self.position[i] += (
                (aruco_position[i] - self.position[i]) * self.ACCELERATION * frame_time
            )
        new_position = np.copy(view_matrix)
        new_position[12:15] = self.position

        for atom in self.atoms:
            atom.render(new_position)

    def set_aruco_position(self, position):
        self.aruco_position = position
