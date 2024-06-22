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
from cluster import Cluster
from speech import TTS, SpeechRecognizer


from shapes.rectangle import Rectangle

load_dotenv()
MARKER_SIZE = 0.48
DEBUG = os.environ.get("DEBUG", False)


class ChemistryAR(mglw.WindowConfig):
    title = "OpenGL Window"
    gl_version = (3, 3)
    resizable = False
    CLUSTER_THRESHOLD = 1.6
    LOOP_DELAY = 1.0
    CLUSTER_VALID_SOLUTION = 3  # Seconds needed to merge into solution

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.game_levels = GameLevels()
        self.molecules: Dict[int, Molecule] = dict()
        self.markers: Dict[int, Marker] = dict()
        self.clusters: Dict[int, List[int]] = {}
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
        self.level_completed = False

        self.last_loop_time = 0.0
        self.cluster_valid_solution_count = 0
        self.merged_molecule_cluster = []
        self.recognizer = SpeechRecognizer()
        self.ask_for_next_level = False
        self.listening_started = False

    def load_level(self, level_number: int) -> None:
        # Reset the markers
        self.markers = dict()
        self.game_levels.set_current_level(level_number)
        self.level_markers = self.game_levels.get_current_level().get_markers()

    def load_next_level(self) -> None:
        self.cluster_valid_solution_count = 0
        self.level_completed = False
        self.load_level(self.game_levels.get_current_level_number() + 1)

    def markers_distance(self, marker1: int, marker2: int):
        marker1_pos = self.markers[marker1].get_marker_pos()
        marker2_pos = self.markers[marker2].get_marker_pos()
        return np.linalg.norm(marker1_pos[1][0] - marker2_pos[1][0])

    def update_clusters(self) -> None:
        self.clusters = {}
        elements = list(self.markers.keys())
        n = len(elements)
        uf = Cluster(n)

        for i in range(n):
            for j in range(i + 1, n):
                if (
                    self.markers_distance(elements[i], elements[j])
                    <= self.CLUSTER_THRESHOLD
                ):
                    uf.union(i, j)

        for i in range(n):
            root = uf.find(i)
            if root not in self.clusters:
                self.clusters[root] = []
            self.clusters[root].append(elements[i])

    def find_solution_in_clusters(self) -> List[int] | None:
        for cluster in self.clusters.values():
            if (
                len(cluster)
                == self.game_levels.get_current_level().get_objetive_markers_amount()
            ):
                solution_found = True
                for marker in cluster:
                    if not self.markers[marker].is_part_of_solution:
                        solution_found = False
                        break
                if solution_found:
                    return cluster

    def is_solution_in_clusters(self) -> bool:
        if self.find_solution_in_clusters() is not None:
            return True
        return False

    def check_solution(self) -> None:
        self.update_clusters()

        if not self.level_completed:
            if self.is_solution_in_clusters():
                self.cluster_valid_solution_count += 1
                if self.cluster_valid_solution_count >= self.CLUSTER_VALID_SOLUTION:
                    self.level_completed = True
                    self.ask_for_next_level = True
                    self.cluster_valid_solution_count = 0
                    print("Solution found!")
                    TTS(
                        "Solution found! Want to proceed to the next level? Say yes or no"
                    )
                    self.merge_into_molecule(self.find_solution_in_clusters())
            elif self.cluster_valid_solution_count > 0:
                self.cluster_valid_solution_count = 0
        else:
            if not self.is_solution_in_clusters():
                self.cluster_valid_solution_count += 1
                if self.cluster_valid_solution_count >= self.CLUSTER_VALID_SOLUTION:
                    self.unmerge_molecule(self.merged_molecule_cluster)
                    self.level_completed = False
            elif self.cluster_valid_solution_count > 0:
                self.cluster_valid_solution_count = 0

    def merge_into_molecule(self, cluster) -> None:
        self.markers[cluster[0]].create_molecule(
            name=self.game_levels.get_current_level().get_objective_name(),
            smiles=self.game_levels.get_current_level().get_objective_smiles(),
        )
        self.markers[cluster[0]].is_merged = True
        for i in range(1, len(cluster)):
            # self.markers[cluster[i]].delete_molecule()
            self.markers[cluster[i]].is_merged = True

        self.level_markers = []

        self.merged_molecule_cluster = cluster

    def unmerge_molecule(self, cluster) -> None:
        self.markers[cluster[0]].delete()
        self.markers[cluster[0]].is_merged = False
        for i in range(1, len(cluster)):
            self.markers[cluster[i]].is_merged = False

        self.merged_molecule_cluster = []

    def update_markers(self, frame_markers: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        # Create and update the found markers
        for marker_id, marker_pos in frame_markers.items():
            if marker_id not in self.markers:
                # Only create a new marker when there are level markers available
                if len(self.level_markers) > 0:
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
                    self.markers[marker_id].delete()

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

        self.last_loop_time += frame_time
        # Perform checks every LOOP_DELAY seconds
        if self.last_loop_time >= self.LOOP_DELAY:
            self.last_loop_time -= self.LOOP_DELAY
            self.check_solution()
            if self.ask_for_next_level:
                if not self.listening_started:
                    self.recognizer.listen()
                    self.listening_started = True

                if self.listening_started and not self.recognizer.is_listening():
                    self.ask_for_next_level = False
                    self.listening_started = False
                    print("Checking response...")
                    if self.recognizer.user_accepted():
                        print("Loading next level...")
                        self.load_next_level()

    def key_event(self, key, action, modifiers):
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.load_next_level()
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.W:
                self.update_clusters()
                self.is_solution_in_clusters()

    def close(self):
        self.cap.release()
        self.recognizer.close()
        cv2.destroyAllWindows()
        super().close()


# Ejecuta la aplicación
if __name__ == "__main__":
    ChemistryAR.run()
