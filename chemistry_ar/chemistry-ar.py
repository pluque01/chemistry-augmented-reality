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

load_dotenv()
MARKER_SIZE = 0.48
DEBUG = os.environ.get("DEBUG", False)


class ChemistryAR(mglw.WindowConfig):
    title = "OpenGL Window"
    gl_version = (3, 3)
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.molecules: Dict[int, Molecule] = dict()
        self.marker_extrinsics: Dict[int, Tuple[np.ndarray, np.ndarray]] = dict()
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

    def create_molecule(
        self, id: int, marker_pos: Tuple[np.ndarray, np.ndarray], atoms: str
    ):
        self.molecules[id] = Molecule(
            self.ctx, atoms, id, marker_pos, self.projection_matrix
        )
        # print(self.molecules[id].get_atom_coordinates())

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
                    self.marker_extrinsics[aruco_id] = (rvecs, tvecs)
                    if aruco_id not in self.molecules:
                        self.create_molecule(
                            aruco_id,
                            self.marker_extrinsics[aruco_id],
                            "OS(=O)(=O)O",
                        )
                    if DEBUG:
                        cv2.drawFrameAxes(
                            frame,
                            camera.cameraMatrix,
                            camera.distCoeffs,
                            rvecs,
                            tvecs,
                            0.1,
                        )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)

        # Dibuja el rectángulo
        self.background.render(frame.tobytes())

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Dibuja la esfera
        for index, m in self.molecules.items():
            m.update_marker_extrinsics(self.marker_extrinsics[index])
            m.render(frame_time)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().close()


# Ejecuta la aplicación
if __name__ == "__main__":
    ChemistryAR.run()
