from cv2.typing import MatLike
import numpy as np
import cv2

cameraMatrix = np.array(
    [
        [606.02487909, 0.0, 342.33938202],
        [0.0, 606.02487909, 253.29263638],
        [0.0, 0.0, 1.0],
    ]
)
distCoeffs = np.array(
    [
        [-2.51667799e00],
        [-1.92650420e01],
        [2.91682913e-03],
        [1.63418219e-03],
        [4.92436753e01],
        [-2.53449712e00],
        [-1.90507209e01],
        [4.86504998e01],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
    ]
)


def solvePnPAruco(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )

    rvecs = np.empty((0, 1, 3))
    tvecs = np.empty((0, 1, 3))
    for corner in corners:
        _, r, t = cv2.solvePnP(
            marker_points,
            corner,
            mtx,
            distortion,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        r, t = cv2.solvePnPRefineLM(
            marker_points,
            corner,
            mtx,
            distortion,
            r,
            t,
            criteria=criteria,
        )
        rvecs = np.append(rvecs, r.reshape(1, 1, 3), axis=0)
        tvecs = np.append(tvecs, t.reshape(1, 1, 3), axis=0)
    return rvecs, tvecs


def extrinsic2ModelView(
    RVEC: np.ndarray, TVEC: np.ndarray, offset: np.ndarray = np.array([0.0, 0.0, 0.0])
) -> MatLike:
    """[Get modelview matrix from RVEC and TVEC]

    Arguments:
        RVEC {[vector]} -- [Rotation vector]
        TVEC {[vector]} -- [Translation vector]
    # TODO
    Keyword Arguments:
        offset {float} -- [Offset]
    """
    R, _ = cv2.Rodrigues(RVEC)

    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    offset_vector = offset.reshape((3, 1))
    offset_vector = R @ offset_vector
    TVEC = TVEC.flatten().reshape((3, 1))
    TVEC = TVEC + offset_vector

    transform_matrix = Rx @ np.hstack((R, TVEC))
    M = np.eye(4)
    M[:3, :] = transform_matrix
    return M.T.flatten()


def intrinsic2Project(
    width: float,
    height: float,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    MTX=cameraMatrix,
):
    """[Get ]

    Arguments:
        MTX {[np.array]} -- [The camera instrinsic matrix that you get from calibrating your chessboard]
        width {[float]} -- [width of viewport]]
        height {[float]} -- [height of viewport]

    Keyword Arguments:
        near_plane {float} -- [near_plane] (default: {0.01})
        far_plane {float} -- [far plane] (default: {100.0})

    Returns:
        [np.array] -- [1 dim array of project matrix]
    """
    P = np.zeros(shape=(4, 4), dtype=np.float32)

    fx, fy = MTX[0, 0], MTX[1, 1]
    cx, cy = MTX[0, 2], MTX[1, 2]

    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / height
    P[2, 0] = 1 - 2 * cx / width
    P[2, 1] = 2 * cy / height - 1
    P[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    P[2, 3] = -1.0
    P[3, 2] = -(2 * far_plane * near_plane) / (far_plane - near_plane)

    return P.flatten()


def ModelView2Position(matrix: MatLike) -> np.ndarray:
    """[Get position from extrinsic matrix]
    Arguments:
        matrix {[MatLike]} -- [View matrix]
    Returns:
        np.ndarray -- [Position]
    """
    return matrix[12:15]


def CVPosition2GLPosition(ocv: np.ndarray) -> np.ndarray:
    """[Convert OpenCV position to OpenGL position]
    Arguments:
        ocv {[np.ndarray]} -- [OpenCV position]
    Returns:
        np.ndarray -- [OpenGL position]
    """
    return np.array([ocv[0], -ocv[1], -ocv[2]])
