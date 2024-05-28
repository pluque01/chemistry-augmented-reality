import numpy as np
from enum import Enum
from typing import Tuple
from molecule import Molecule
from levels import LevelMarker


class MarkerState(Enum):
    ACTIVE = 0
    NOT_FOUND = 1
    INACTIVE = 2


class MarkerAtoms:
    def __init__(self, element: str, count: int):
        self.element = element
        self.count = count


class Marker:
    def __init__(
        self,
        *,
        ctx,
        id: int,
        marker_extrinsics: Tuple[np.ndarray, np.ndarray],
        projection_matrix,
        level_marker: LevelMarker,
    ):
        self.ctx = ctx
        self.id = id
        self.marker_pos = marker_extrinsics
        self.state = MarkerState.ACTIVE
        self.frames_lost = 0
        self.molecule = None
        self.projection_matrix = projection_matrix
        self.level_marker = level_marker

        self.INACTIVE_THRESHOLD = 20

        self.create_molecule(marker_atoms=level_marker.atoms)
        self.is_part_of_solution = level_marker.required

    def update_marker_pos(self, marker_pos: Tuple[np.ndarray, np.ndarray]):
        self.marker_pos = marker_pos

    def update_marker_state(self, state: MarkerState):
        if (
            self.state == MarkerState.ACTIVE
            or MarkerState.NOT_FOUND
            and state == MarkerState.NOT_FOUND
        ):
            self.state = MarkerState.NOT_FOUND
            self.frames_lost += 1
        if self.state == MarkerState.NOT_FOUND and state == MarkerState.ACTIVE:
            self.state = MarkerState.ACTIVE
            self.frames_lost = 0

        if self.frames_lost > self.INACTIVE_THRESHOLD:
            self.state = MarkerState.INACTIVE

    def get_marker_pos(self):
        return self.marker_pos

    def get_marker_state(self):
        return self.state

    def get_frames_lost(self):
        return self.frames_lost

    def create_molecule(self, smiles: str = "", marker_atoms=None):
        self.molecule = Molecule(
            ctx=self.ctx,
            name=self.level_marker.get_name(),
            aruco_id=self.id,
            marker_position=self.marker_pos,
            projection_matrix=self.projection_matrix,
            smiles=smiles,
            marker_atoms=marker_atoms,
        )

    def delete_molecule(self):
        self.molecule = None

    def render_molecule(self, frame_time: float):
        if self.molecule is not None:
            self.molecule.update_marker_extrinsics(self.marker_pos)
            self.molecule.render(frame_time)

    def render(self, frame_time: float):
        if self.molecule is not None:
            self.render_molecule(frame_time)

    def get_molecule_name(self) -> str:
        if self.molecule is not None:
            return self.molecule.get_name()
        else:
            return "No molecule created"
