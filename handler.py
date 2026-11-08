import os
import base64
import urllib.request
import cv2
import numpy as np
import runpod

MODEL_URL = "https://github.com/facefusion/facefusion-assets/releases/download/3.0.0/inswapper_128.onnx"
MODEL_PATH = "/tmp/inswapper_128.onnx"

_swapper = None
_app = None

def get_swapper():
    global _swapper, _app
    if _swapper is not None:
        return _swapper, _app

    if not os.path.exists(MODEL_PATH):
        print("Downloading inswapper model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

    import insightface
    from insightface.app import FaceAnalysis

    _app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _app.prepare(ctx_id=0, det_size=(640, 640))

    _swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Swapper ready.")
    return _swapper, _app

def b64_to_img(b64_str):
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def img_to_b64(img):
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("utf-8")

def get_face(app, img):
    faces = app.get(img)
    if not faces:
        return None
    # biggest face
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

def handler(event):
    try:
        inp = event.get("input", {})
        source_b64 = inp.get("source_image")
        target_b64 = inp.get("target_image")

        if not source_b64 or not target_b64:
            return {"error": "Need source_image and target_image as base64 jpg"}

        swapper, app = get_swapper()

        source_img = b64_to_img(source_b64)
        target_img = b64_to_img(target_b64)

        source_face = get_face(app, source_img)
        target_face = get_face(app, target_img)

        if source_face is None:
            return {"error": "No face found in source_image"}
        if target_face is None:
            return {"error": "No face found in target_image"}

        result = swapper.get(target_img, target_face, source_face, paste_back=True)

        return {"image": img_to_b64(result)}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
