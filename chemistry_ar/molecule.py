import numpy as np
from typing import Tuple
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
        offset: np.ndarray,
        color: ArrayLike,
        projection_matrix,
    ):
        super().__init__(ctx, radius, offset, color, projection_matrix)
        self.name = name
        self.offset = offset
        print("Atom position", self.offset[0], self.offset[1], self.offset[2])

    def renderAtom(
        self,
        marker_extrinsics: Tuple[np.ndarray, np.ndarray],
        molecule_position: np.ndarray,
    ):
        modelview_matrix = camera.extrinsic2ModelView(
            marker_extrinsics[0], molecule_position, self.offset
        )
        super().render(modelview_matrix)


class Molecule:
    def __init__(
        self,
        ctx,
        atoms: str,
        aruco: int,
        marker_position: Tuple[np.ndarray, np.ndarray],
        projection_matrix,
    ):
        self.ctx = ctx
        self.marker_id = aruco
        self.marker_extrinsics = marker_position
        self.projection_matrix = projection_matrix
        self.atoms = []
        self.INITIAL_OFFSET = np.array([0.0, 0.0, 1.0])

        self.mol = Chem.MolFromSmiles(atoms)

        # Movement related
        self.ACCELERATION = 2
        self.position = marker_position[1][0][0] + self.INITIAL_OFFSET

        for atom in self.get_atom_properties():
            print("debug", atom[1][0], atom[1][1], atom[1][2])
            atom_position = atom[1] * 0.1 + self.INITIAL_OFFSET
            self.atoms.append(
                Atom(
                    ctx,
                    atom_data[atom[0]]["name"],
                    atom_data[atom[0]]["size"],
                    atom_position,
                    np.asarray(atom_data[atom[0]]["color"], np.float32) / 255.0,
                    projection_matrix,
                )
            )
            print(f"Created atom {atom[0]} at {atom_position}")

    def update_marker_extrinsics(self, new_extrinsics: Tuple[np.ndarray, np.ndarray]):
        self.marker_extrinsics = new_extrinsics

    def update_position(self, frame_time: float):
        tvecs = self.marker_extrinsics[1][0][0]
        self.position += (tvecs - self.position) * self.ACCELERATION * frame_time

    def render(self, frame_time: float):
        self.update_position(frame_time)
        for atom in self.atoms:
            atom.renderAtom(self.marker_extrinsics, self.position)

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
