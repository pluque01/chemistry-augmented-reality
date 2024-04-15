import os
import cv2
import moderngl_window as mglw
import numpy as np
import camera
from dotenv import load_dotenv

from shapes.rectangle import Rectangle
from shapes.sphere import Sphere

load_dotenv()
MARKER_SIZE = 0.48
DEBUG = os.environ.get("DEBUG", False)


class ChemistryAR(mglw.WindowConfig):
    title = "OpenGL Window"
    gl_version = (3, 3)
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sphere = None
        self.rectangle = None
        self.cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.projection_matrix = camera.intrinsic2Project(
            self.wnd.width, self.wnd.height, near_plane=1.0, far_plane=10.0
        )

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)

        if not self.sphere:
            print("Creating sphere")
            self.sphere = Sphere(self.ctx, 0.3, self.projection_matrix)

        if not self.rectangle:
            print("Creating rectangle")
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.rectangle = Rectangle(self.ctx, width, height)

        ret, frame = self.cap.read()
        # Convertir a escala de grises para mejorar la detección
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        view_matrix = np.empty((4, 4), dtype=np.float32)
        if ret:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame_gray, self.aruco_dict, parameters=self.aruco_params
            )
            if ids is not None:
                rvecs, tvecs = camera.solvePnPAruco(
                    corners, MARKER_SIZE, camera.cameraMatrix, camera.distCoeffs
                )
                # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                #     corners, MARKER_SIZE, camera.cameraMatrix, camera.distCoeffs
                # )
                view_matrix = camera.extrinsic2ModelView(rvecs, tvecs, offset=1.0)
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
        self.rectangle.render(frame.tobytes())

        # Dibuja la esfera
        self.sphere.render(view_matrix)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().close()


# Ejecuta la aplicación
if __name__ == "__main__":
    ChemistryAR.run()
