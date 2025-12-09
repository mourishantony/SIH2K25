import argparse
import glob
import json
import os
from typing import List, Tuple

import cv2 as cv
import numpy as np


def _collect_image_paths(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No images found for pattern: {pattern}")
    return paths


def calibrate_from_images(
    image_paths: List[str],
    board_cols: int = 9,
    board_rows: int = 6,
    square_size_m: float = 0.024,
    use_sb: bool = True,
    debug_out: str | None = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    pattern_size = (board_cols, board_rows)
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    imsize = None

    for p in image_paths:
        img = cv.imread(p)
        if img is None:
            print(f"[warn] Unreadable image: {p}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = gray.shape[::-1]

        found = False
        corners = None

        if use_sb and hasattr(cv, "findChessboardCornersSB"):
            try:
                ok_sb, corners_sb = cv.findChessboardCornersSB(gray, pattern_size)
                if ok_sb and corners_sb is not None:
                    corners = corners_sb.astype(np.float32)
                    found = True
            except Exception:
                pass

        if not found:
            flags = (
                cv.CALIB_CB_ADAPTIVE_THRESH
                | cv.CALIB_CB_NORMALIZE_IMAGE
                | cv.CALIB_CB_FAST_CHECK
            )
            ok_legacy, corners_legacy = cv.findChessboardCorners(gray, pattern_size, flags)
            if ok_legacy:
                corners = cv.cornerSubPix(
                    gray,
                    corners_legacy,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=(
                        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                found = True

        if not found:
            print(f"[warn] Chessboard not found: {p}")
            if debug_out:
                os.makedirs(debug_out, exist_ok=True)
                cv.imwrite(os.path.join(debug_out, os.path.basename(p)), img)
            continue

        objpoints.append(objp)
        imgpoints.append(corners)

        if debug_out:
            dbg = img.copy()
            cv.drawChessboardCorners(dbg, pattern_size, corners, True)
            os.makedirs(debug_out, exist_ok=True)
            cv.imwrite(os.path.join(debug_out, f"corners_{os.path.basename(p)}"), dbg)

    if not objpoints:
        raise RuntimeError("No valid calibration views collected.")

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imsize, None, None)
    fx, fy = float(K[0, 0]), float(K[1, 1])

    calib = {
        "ret": float(ret),
        "image_size": {"width": int(imsize[0]), "height": int(imsize[1])},
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "fx": fx,
        "fy": fy,
        "board_cols": board_cols,
        "board_rows": board_rows,
        "square_size_m": float(square_size_m),
    }
    return calib, K, dist


def save_calibration(calib: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[info] Saved calibration to: {out_path}\n       fx={calib['fx']:.2f}, fy={calib['fy']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Camera calibration from checkerboard images")
    parser.add_argument("--images", type=str, required=True, help="Glob pattern to calibration images, e.g. 'data/calib/*.jpg'")
    parser.add_argument("--board-cols", type=int, default=9, help="Internal corners (columns)")
    parser.add_argument("--board-rows", type=int, default=6, help="Internal corners (rows)")
    parser.add_argument("--square-size-m", type=float, default=0.024, help="Square size in meters")
    parser.add_argument("--out", type=str, default="data/calibration_cam.json", help="Output JSON path")
    parser.add_argument("--no-sb", dest="use_sb", action="store_false", help="Disable SB corner detector (use legacy only)")
    parser.add_argument("--debug-out", type=str, default="data/calib/debug", help="Folder to write debug images (corners/failed)")
    args = parser.parse_args()

    imgs = _collect_image_paths(args.images)
    calib, K, dist = calibrate_from_images(
        imgs,
        args.board_cols,
        args.board_rows,
        args.square_size_m,
        use_sb=args.use_sb,
        debug_out=args.debug_out,
    )
    save_calibration(calib, args.out)


if __name__ == "__main__":
    main()
