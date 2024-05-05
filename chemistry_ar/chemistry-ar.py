import os
import cv2
import moderngl_window as mglw
import moderngl
from moderngl_window.context.base.window import Tuple
import numpy as np
from numpy._typing import ArrayLike
import camera
from dotenv import load_dotenv
from typing import List, Dict
from molecule import Molecule

from shapes.rectangle import Rectangle

from enum import Enum

load_dotenv()
MARKER_SIZE = 0.48
DEBUG = os.environ.get("DEBUG", False)


class MarkerState(Enum):
    ACTIVE = 0
    NOT_FOUND = 1
    INACTIVE = 2


class Marker:
    def __init__(self, id: int, marker_extrinsics: Tuple[np.ndarray, np.ndarray]):
        self.id = id
        self.marker_pos = marker_extrinsics
        self.state = MarkerState.ACTIVE
        self.frames_lost = 0
        self.molecule = None

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

        if self.frames_lost > 10:
            self.state = MarkerState.INACTIVE

    def get_marker_pos(self):
        return self.marker_pos

    def get_marker_state(self):
        return self.state

    def get_frames_lost(self):
        return self.frames_lost

    def create_molecule(
        self,
        ctx,
        id: int,
        marker_pos: Tuple[np.ndarray, np.ndarray],
        atoms: str,
        projection_matrix,
    ):
        self.molecule = Molecule(ctx, atoms, id, marker_pos, projection_matrix)

    def delete_molecule(self):
        self.molecule = None

    def render_molecule(self, frame_time: float):
        if self.molecule is not None:
            self.molecule.update_marker_extrinsics(self.marker_pos)
            self.molecule.render(frame_time)


class ChemistryAR(mglw.WindowConfig):
    title = "OpenGL Window"
    gl_version = (3, 3)
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.molecules: Dict[int, Molecule] = dict()
        self.markers: Dict[int, Marker] = dict()
        self.background = Rectangle(self.ctx, self.wnd.width, self.wnd.height)
        self.cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.projection_matrix = camera.intrinsic2Project(
            self.wnd.width, self.wnd.height, near_plane=1.0, far_plane=1000.0
        )

    def update_markers(self, frame_markers: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        # Create and update the found markers
        for marker_id, marker_pos in frame_markers.items():
            if marker_id not in self.markers:
                self.markers[marker_id] = Marker(marker_id, marker_pos)
                self.markers[marker_id].create_molecule(
                    self.ctx,
                    marker_id,
                    marker_pos,
                    "OS(=O)(=O)O",
                    self.projection_matrix,
                )
            else:
                self.markers[marker_id].update_marker_pos(marker_pos)

        # Check if any marker is lost
        for marker_id, _ in self.markers.items():
            if marker_id not in frame_markers:
                self.markers[marker_id].update_marker_state(MarkerState.NOT_FOUND)
                if self.markers[marker_id].get_marker_state() == MarkerState.INACTIVE:
                    self.markers[marker_id].delete_molecule()

        # Delete inactive markers
        for marker in self.markers.copy():
            if self.markers[marker].get_marker_state() == MarkerState.INACTIVE:
                del self.markers[marker]

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        ret, frame = self.cap.read()
        # Convertir a escala de grises para mejorar la detección
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame_gray, self.aruco_dict, parameters=self.aruco_params
            )
            frame_markers = dict()
            if ids is not None:  # Si se detectó algún marcador
                for i in range(len(ids)):
                    aruco_id = ids[i][0]
                    rvecs, tvecs = camera.solvePnPAruco(
                        corners[i], MARKER_SIZE, camera.cameraMatrix, camera.distCoeffs
                    )
                    imgpts, _ = cv2.projectPoints(
                        np.array([0, 0, 0], dtype=np.float32),
                        rvecs[0],
                        tvecs[0],
                        camera.cameraMatrix,
                        camera.distCoeffs,
                    )
                    cv2.putText(
                        frame,
                        f"Marcador {ids[i]}",
                        tuple(imgpts[0][0][0:2].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    #     corners, MARKER_SIZE, camera.cameraMatrix, camera.distCoeffs
                    # )
                    frame_markers[aruco_id] = (rvecs, tvecs)
                    if DEBUG:
                        cv2.drawFrameAxes(
                            frame,
                            camera.cameraMatrix,
                            camera.distCoeffs,
                            rvecs,
                            tvecs,
                            0.1,
                        )

            self.update_markers(frame_markers)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)

        # Dibuja el rectángulo
        self.background.render(frame.tobytes())

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Dibuja la esfera
        for _, marker in self.markers.items():
            marker.render_molecule(frame_time)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().close()


# Ejecuta la aplicación
if __name__ == "__main__":
    ChemistryAR.run()
