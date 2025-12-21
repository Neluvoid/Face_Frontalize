import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"  # suppress QT warnings
os.environ["QT_DEBUG_PLUGINS"] = "0"  # disable Qt plugin debug output

from engine import Engine
import asyncio
from loader import initialize_models
import cv2
import dlib
import numpy as np
from imutils import face_utils
import gc
import torch
import argparse
import math
import pathlib
import sys
import urllib.request
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" # dlib 68点ランドマークモデルのパス

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
model_points = np.array([
    (0.0, 0.0, 0.0),             # 鼻先 (point 30)
    (0.0, -330.0, -65.0),        # 顎      (point 8)
    (-225.0, 170.0, -135.0),     # 左目外角(point 36)
    (225.0, 170.0, -135.0),      # 右目外角(point 45)
    (-150.0, -150.0, -125.0),    # 左口角(point 48)
    (150.0, -150.0, -125.0)      # 右口角(point 54)
], dtype=np.float64)

MODEL_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
MODEL_WEIGHT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)


def ensure_model_files(model_dir: pathlib.Path) -> tuple[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    proto_path = model_dir / "deploy.prototxt"
    weight_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    def download(url: str, dest: pathlib.Path):
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(
                f"モデルのダウンロードに失敗しました: {url}\n"
                f"エラー: {e}\n"
                "手動でダウンロードしてモデルディレクトリに配置してください。"
            ) from e

    if not proto_path.exists():
        print(f"[info] downloading {proto_path.name}")
        download(MODEL_PROTO_URL, proto_path)
    if not weight_path.exists():
        print(f"[info] downloading {weight_path.name}")
        download(MODEL_WEIGHT_URL, weight_path)
    return str(proto_path), str(weight_path)


def load_face_net(proto: str, weight: str):
    net = cv2.dnn.readNetFromCaffe(proto, weight)
    return net


def estimate_fisheye_layout(width: int, height: int):
    """二眼魚眼の中心と半径を概算する。"""
    radius = min(width * 0.25, height * 0.5) * 0.95
    centers = [
        (width * 0.25, height * 0.5),
        (width * 0.75, height * 0.5),
    ]
    return centers, radius


def detect_faces(net, image: np.ndarray, center, radius, conf_thres: float):
    """指定した魚眼円領域で顔を検出し、(score, bbox) のリストを返す。"""
    cx, cy = center
    r = radius
    x0 = max(int(cx - r), 0)
    y0 = max(int(cy - r), 0)
    x1 = min(int(cx + r), image.shape[1])
    y1 = min(int(cy + r), image.shape[0])
    crop = image[y0:y1, x0:x1].copy()

    # マスクして円外の影響を減らす
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(r), int(r)), int(r), (255, 255, 255), -1)
    crop = cv2.bitwise_and(crop, crop, mask=mask)

    blob = cv2.dnn.blobFromImage(
        crop,
        1.0,
        (300, 300),
        (104, 117, 123),
        swapRB=False,
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    h, w = crop.shape[:2]
    for i in range(detections.shape[2]):
        score = float(detections[0, 0, i, 2])
        if score < conf_thres:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x, y, x2, y2 = box.astype(int)
        x = max(x, 0)
        y = max(y, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 <= x or y2 <= y:
            continue
        # グローバル座標に変換
        faces.append(
            (
                score,
                (
                    x + x0,
                    y + y0,
                    x2 + x0,
                    y2 + y0,
                ),
            )
        )
    return faces


def vec_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def pixel_to_dir_equidistant(px, py, center, f, eps=1e-9):
    """魚眼画像上のピクセルから3D方向ベクトルを推定 (equidistant モデル)。"""
    cx, cy = center
    dx = px - cx
    dy = py - cy
    r = math.hypot(dx, dy)
    theta = r / (f + eps)
    phi = math.atan2(dy, dx)
    sin_t = math.sin(theta)
    dir_vec = np.array(
        [sin_t * math.cos(phi), sin_t * math.sin(phi), math.cos(theta)],
        dtype=np.float32,
    )
    return vec_normalize(dir_vec)


def build_rotation(forward: np.ndarray):
    """forward を z 軸に合わせる回転行列を作る。"""
    f = vec_normalize(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = vec_normalize(np.cross(up, f))
    up2 = vec_normalize(np.cross(f, right))
    R = np.stack([right, up2, f], axis=1)  # columns: right, up, forward
    return R


def rectify_patch(
    image: np.ndarray,
    face_bbox,
    fisheye_center,
    fisheye_radius,
    output_size=640,
    fov_deg=90.0,
):
    """顔中心を視線方向とした透視画像を生成。"""
    x1, y1, x2, y2 = face_bbox
    fx = (x1 + x2) / 2.0
    fy = (y1 + y2) / 2.0

    # 魚眼の仮想焦点 (equidistant): f = r / (FOV/2)
    fisheye_f = fisheye_radius / math.radians(95.0)  # 190deg 想定
    face_dir = pixel_to_dir_equidistant(fx, fy, fisheye_center, fisheye_f)
    R = build_rotation(face_dir)

    out_h = out_w = int(output_size)
    fov_h = math.radians(fov_deg)
    fov_v = fov_h * (out_h / out_w)
    xs = np.linspace(-math.tan(fov_h / 2), math.tan(fov_h / 2), out_w)
    ys = np.linspace(-math.tan(fov_v / 2), math.tan(fov_v / 2), out_h)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dirs_cam = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=2, keepdims=True)

    # カメラ座標からワールド(魚眼基準)へ
    dirs_world = dirs_cam @ R.T
    dirs_world = dirs_world.reshape(-1, 3)

    # ワールド方向を魚眼平面へ投影
    theta = np.arccos(np.clip(dirs_world[:, 2], -1.0, 1.0))
    phi = np.arctan2(dirs_world[:, 1], dirs_world[:, 0])
    r = fisheye_f * theta
    map_x = fisheye_center[0] + r * np.cos(phi)
    map_y = fisheye_center[1] + r * np.sin(phi)
    map_x = map_x.reshape(out_h, out_w).astype(np.float32)
    map_y = map_y.reshape(out_h, out_w).astype(np.float32)

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rectified

async def main():
    live_portrait = await initialize_models()
    engine = Engine(live_portrait=live_portrait)
    proto, weight = ensure_model_files(pathlib.Path("models"))
    net = load_face_net(proto, weight)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("Starting video stream...")
    print("Press 'q' to quit.")
    
    # Configuration to reduce GPU load
    PROCESS_EVERY_N_FRAMES = 5  # only run expensive transform every N frames
    SEND_MAX_WIDTH = 512  # resize image before sending to engine to reduce memory

    # Disable grad globally to reduce memory used by autograd
    torch.set_grad_enabled(False)
    # try to put model in eval mode if available
    try:
        if hasattr(engine.live_portrait, 'live_portrait_wrapper'):
            wrapper = engine.live_portrait.live_portrait_wrapper
            if hasattr(wrapper, 'model'):
                try:
                    wrapper.model.eval()
                except Exception:
                    pass
    except Exception:
        pass

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 魚眼顔検出と向き推定は毎 N フレームに一度だけ実行
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            centers, radius = estimate_fisheye_layout(frame.shape[1], frame.shape[0])

            all_faces = []
            for i, c in enumerate(centers):
                faces = detect_faces(net, frame, c, radius, 0.4)
                all_faces.extend(faces)

            # 検出できない場合、しきい値を下げて再試行
            if not all_faces:
                lower_conf = max(0.1, 0.4 * 0.5)
                all_faces = []
                for c in centers:
                    faces = detect_faces(net, frame, c, radius, lower_conf)
                    all_faces.extend(faces)

            if not all_faces:
                cv2.imshow("OUTPUT", np.zeros((SEND_MAX_WIDTH, SEND_MAX_WIDTH, 3), dtype=np.uint8))
                continue

            # 最大の顔を選択
            def area(bbox):
                x1, y1, x2, y2 = bbox
                return (x2 - x1) * (y2 - y1)

            best = max(all_faces, key=lambda x: area(x[1]))
            best_bbox = best[1]
            face_center_x = (best_bbox[0] + best_bbox[2]) / 2
            # どちらの魚眼に属するかで中心を決める
            chosen_center = min(centers, key=lambda c: abs(face_center_x - c[0]))

            frame = rectify_patch(
                frame,
                best_bbox,
                chosen_center,
                radius,
                output_size=SEND_MAX_WIDTH,
                fov_deg=90.0,
            )
            # cv2.imshow("OUTPUT", frame)
            frame0 = frame.copy()
            size = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray, 0)
            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                # 2D イメージ上の対応点（dlib 68 点のインデックス）
                image_points = np.array([
                    shape[30],     # 鼻先
                    shape[8],      # 顎
                    shape[36],     # 左目外角
                    shape[45],     # 右目外角
                    shape[48],     # 左口角
                    shape[54]      # 右口角
                ], dtype=np.float64)

                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_coeffs = np.zeros((4,1))  # レンズ歪みゼロに仮定
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                # 回転ベクトル -> 回転行列 -> オイラー角（度）
                rmat, _ = cv2.Rodrigues(rotation_vector)
                proj_matrix = np.hstack((rmat, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                #yaw, pitch, roll = float(euler_angles[1]), float(euler_angles[0]), float(euler_angles[2])
                yaw, pitch, roll = map(lambda x: float(np.asarray(x).item()), [euler_angles[1], euler_angles[0], euler_angles[2]])
        
            # ensure variables exist
            if 'success' in locals() and success:
                # resize before sending to engine to reduce memory & compute
                h, w = frame0.shape[:2]
                if w > SEND_MAX_WIDTH:
                    new_h = int(h * (SEND_MAX_WIDTH / w))
                    send_img = cv2.resize(frame0, (SEND_MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
                else:
                    send_img = frame0

                binary_data = cv2.imencode('.jpg', send_img)[1].tobytes()

                # load and transform (these may allocate GPU memory internally)
                try:
                    initImage = None
                    try:
                        initImage = await engine.load_image(binary_data)
                    except Exception as e:
                        # gradio.exceptions.Error for no face detected or other issues
                        msg = str(e)
                        if 'No face detected' in msg:
                            # 顔検出失敗は静かにスキップ
                            pass
                        else:
                            print(f"Skipping frame: failed to load image: {msg}")
                        initImage = None
                    if pitch >= 170.0 or pitch <= -170.0:
                        cv2.imshow("OUTPUT", frame0)
                    elif initImage is not None:
                        if pitch >= 0.0:
                            params = {'rotate_pitch': 180 - pitch}
                        else:
                            params = {'rotate_pitch': -180 - pitch}
                        buffer = await engine.transform_image(uid=initImage['u'], params=params)

                        restored_img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
                        cv2.imshow("OUTPUT", restored_img)
                    else:
                        # initImage is None (face detection failed), show original frame
                        cv2.imshow("OUTPUT", frame0)

                finally:
                    # Try to free caches and temporary objects to release GPU memory
                    try:
                        # remove processed data from engine cache if present
                        if initImage and isinstance(initImage, dict) and initImage.get('u'):
                            engine.processed_cache.pop(initImage['u'], None)
                    except Exception:
                        pass

                    # delete large objects and run GC + empty CUDA cache
                    for name in ('binary_data', 'initImage', 'buffer', 'restored_img', 'send_img', 'frame0', 'gray'):
                        if name in locals():
                            try:
                                del locals()[name]
                            except Exception:
                                try:
                                    del globals()[name]
                                except Exception:
                                    pass

                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        
        # 不要な変数をクリア（存在チェックしてから）
        for _name in ('frame','ret'):
            if _name in locals():
                try:
                    del locals()[_name]
                except Exception:
                    pass
        
        # 定期的にガベージコレクション実行
        frame_count += 1
        if frame_count % 30 == 0:
            gc.collect()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Qt スレッド終了の完全性を待つ
    import time
    time.sleep(0.2)

    return 0

# 実行
result = asyncio.run(main())