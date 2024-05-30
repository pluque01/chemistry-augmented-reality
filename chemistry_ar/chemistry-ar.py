import os
import cv2
import moderngl_window as mglw
import moderngl
import numpy as np
import camera
from dotenv import load_dotenv
from typing import Dict, Tuple, List
from molecule import Molecule
from marker import Marker, MarkerState
from levels import GameLevels, LevelMarker


from shapes.rectangle import Rectangle

load_dotenv()
MARKER_SIZE = 0.48
DEBUG = os.environ.get("DEBUG", False)


class ChemistryAR(mglw.WindowConfig):
    title = "OpenGL Window"
    gl_version = (3, 3)
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.game_levels = GameLevels()
        self.molecules: Dict[int, Molecule] = dict()
        self.markers: Dict[int, Marker] = dict()
        self.level_markers: List[LevelMarker] = []
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
        self.load_level(0)

    def load_level(self, level_number: int) -> None:
        # Reset the markers
        self.markers = dict()
        self.game_levels.set_current_level(level_number)
        self.level_markers = self.game_levels.get_current_level().get_markers()

    def load_next_level(self) -> None:
        self.load_level(self.game_levels.get_current_level_number() + 1)

    def update_markers(self, frame_markers: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        # Create and update the found markers
        for marker_id, marker_pos in frame_markers.items():
            if marker_id not in self.markers:
                self.markers[marker_id] = Marker(
                    ctx=self.ctx,
                    id=marker_id,
                    marker_extrinsics=marker_pos,
                    projection_matrix=self.projection_matrix,
                    level_marker=self.level_markers.pop(),
                )
                # "CC(=O)NCCC1=CNc2c1cc(OC)cc2",
                # self.markers[marker_id].create_molecule()
            else:
                self.markers[marker_id].update_marker_pos(marker_pos)
                self.markers[marker_id].update_marker_state(MarkerState.ACTIVE)

        # Check if any marker is lost
        for marker_id, _ in self.markers.items():
            if marker_id not in frame_markers:
                self.markers[marker_id].update_marker_state(MarkerState.NOT_FOUND)
                if self.markers[marker_id].get_marker_state() == MarkerState.INACTIVE:
                    self.markers[marker_id].delete_molecule()

        # Delete inactive markers
        for marker in self.markers.copy():
            if self.markers[marker].get_marker_state() == MarkerState.INACTIVE:
                # Append the level marker back to the level markers list
                self.level_markers.append(self.markers[marker].level_marker)
                del self.markers[marker]

    def draw_markers_text(self, frame):
        for marker in self.markers:
            if self.markers[marker].get_marker_state() == MarkerState.ACTIVE:
                marker_extrinsics = self.markers[marker].get_marker_pos()
                imgpts, _ = cv2.projectPoints(
                    np.array([0, 0, 0], dtype=np.float32),
                    marker_extrinsics[0][0],
                    marker_extrinsics[1][0],
                    camera.cameraMatrix,
                    camera.distCoeffs,
                )
                cv2.putText(
                    frame,
                    self.markers[marker].get_molecule_name(),
                    tuple(imgpts[0][0][0:2].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        return frame

    def draw_objective_text(self, frame):
        cv2.putText(
            frame,
            f"Objective: {self.game_levels.get_current_level().get_objective_name()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return frame

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
            frame = self.draw_markers_text(frame)
            frame = self.draw_objective_text(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)

        # Dibuja el rectángulo
        self.background.render(frame.tobytes())

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Dibuja la esfera
        for _, marker in self.markers.items():
            marker.render(frame_time)

    def key_event(self, key, action, modifiers):
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.load_next_level()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().close()


# Ejecuta la aplicación
if __name__ == "__main__":
    ChemistryAR.run()
