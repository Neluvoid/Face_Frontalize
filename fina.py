from engine import Engine
import asyncio
from loader import initialize_models
import cv2
import dlib
import numpy as np
from imutils import face_utils
import gc
import torch
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

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

async def main():
    live_portrait = await initialize_models()
    engine = Engine(live_portrait=live_portrait)
    cap = cv2.VideoCapture(0)
    print("Starting video stream...")
    print("Press 'q' to quit.")
    
    # Configuration to reduce GPU load
    PROCESS_EVERY_N_FRAMES = 5  # only run expensive transform every N frames
    SEND_MAX_WIDTH = 640  # resize image before sending to engine to reduce memory

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

            # 結果描画
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            # 軸を描画（視覚確認用）
            axis = np.float32([[300,0,0], [0,300,0], [0,0,300]])
            imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            p1 = tuple(image_points[0].ravel().astype(int))
            for pt in imgpts:
                pt = tuple(pt.ravel().astype(int))
                cv2.line(frame, p1, pt, (255,0,0), 2)

        cv2.imshow("Head Pose", frame)

        # Only perform heavy processing every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
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
                    for name in ('binary_data', 'initImage', 'buffer', 'restored_img', 'send_img'):
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
        for _name in ('frame', 'frame0', 'gray', 'size'):
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
    return 0

# 実行
result = asyncio.run(main())