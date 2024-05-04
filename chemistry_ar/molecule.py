import numpy as np

from numpy._typing import ArrayLike
from shapes.sphere import Sphere
from rdkit import Chem
from rdkit.Chem import AllChem
import camera
import toml

atom_data = toml.load("chemistry_ar/data/atoms.toml")


class Atom(Sphere):
    def __init__(
        self,
        ctx,
        name: str,
        radius: float,
        position: ArrayLike,
        color: ArrayLike,
        projection_matrix,
    ):
        super().__init__(ctx, radius, position, color, projection_matrix)
        self.name = name
        self.position = position

    def render(self, view_matrix):
        new_position = np.copy(view_matrix)
        new_position[12:15] += self.position
        super().render(new_position)


class Molecule:
    def __init__(
        self,
        ctx,
        atoms: str,
        aruco: int,
        position: np.ndarray,
        projection_matrix,
    ):
        self.ctx = ctx
        self.aruco = aruco
        self.projection_matrix = projection_matrix
        self.atoms = []

        self.mol = Chem.MolFromSmiles(atoms)

        # Movement related
        self.ACCELERATION = 2
        self.position = position

        for atom in self.get_atom_properties():
            print(f"Creating atom {atom[0]} at {atom[1]}")
            self.atoms.append(
                Atom(
                    ctx,
                    atom_data[atom[0]]["name"],
                    atom_data[atom[0]]["size"],
                    atom[1] * 0.1,
                    np.asarray(atom_data[atom[0]]["color"], np.float32) / 255.0,
                    projection_matrix,
                )
            )

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

    def get_atom_properties(self):
        print(self.mol.GetNumAtoms())
        # Generate 3D coordinates
        AllChem.EmbedMolecule(self.mol)
        AllChem.MMFFOptimizeMolecule(self.mol)

        # Extract atom coordinates
        # Obtiene las coordenadas 3D de los átomos
        conformer = self.mol.GetConformer()

        atoms = []
        # Itera sobre cada átomo y obtiene su nombre y posición
        for atom in self.mol.GetAtoms():
            atom_index = atom.GetIdx()
            atom_name = atom.GetSymbol()
            atom_position = conformer.GetAtomPosition(atom_index)
            atoms.append((atom_name, atom_position))
        return atoms
