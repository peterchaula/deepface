# built-in dependencies
import os
import traceback
import shutil
from typing import Optional, Union

# 3rd party dependencies
import numpy as np
from PIL import Image
import uuid
import cv2

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger
from deepface.commons import folder_utils, image_utils

logger = Logger()


# pylint: disable=broad-except


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400

REGISTERED_FACES_DIR = "../data/registered_faces"

def register(
    img_path: Union[str, np.ndarray],
    label: str,
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
):
    try:
        label_directory = os.path.join(REGISTERED_FACES_DIR, label)
        os.makedirs(label_directory, exist_ok=True)
        
        result = {}
        
        try:
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
            )
            
            if len(faces) == 0:
                return {"error": "No face detected in the image"}, 400
            
            face_img = faces[0]["face"]
            
            import time
            import mimetypes
            
            extension = ".jpg"
            
            if isinstance(img_path, str) and os.path.isfile(img_path):
                _, ext = os.path.splitext(img_path)
                if ext:
                    extension = ext
            
            filename = f"{uuid.uuid4()}{extension}"
            file_path = os.path.join(label_directory, filename)
            
            if isinstance(img_path, str):
                if os.path.isfile(img_path):
                    shutil.copy(img_path, file_path)
                else:
                    raise ValueError("Image path is not a file")
            else:
                cv2.imwrite(file_path, img_path)
            
            result["success"] = True
            result["label"] = label
            return result
            
        except Exception as detection_err:
            tb_str = traceback.format_exc()
            logger.error(str(detection_err))
            logger.error(tb_str)
            return {"error": f"Face detection failed: {str(detection_err)}"}, 400
            
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception during registration: {str(err)} - {tb_str}"}, 400


def find_by_image(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    model_name: str = "Facenet",
    distance_metric: str = "cosine",
    threshold: float = None,
):
    try:
        if not os.path.exists(REGISTERED_FACES_DIR):
            return {"error": "No registered faces found in the database"}, 404
            
        dfs = DeepFace.find(
            img_path=img_path,
            db_path=REGISTERED_FACES_DIR,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            threshold=threshold,
            silent=True,
        )
        
        result = {"results": []}
        
        if isinstance(dfs, list) and len(dfs) > 0:
            for df in dfs:
                if not df.empty:
                    matches = []
                    for _, row in df.iterrows():
                        identity = row["identity"]
                        label = os.path.basename(os.path.dirname(identity))
                        matches.append({
                            "label": label,
                            "distance": row["distance"],
                            "file_path": identity,
                            "confidence": row["confidence"],
                        })
                    
                    result["results"].append({
                        "matches": matches,
                        "count": len(matches),
                    })
                else:
                    result["results"].append({"matches": [], "count": 0})
        else:
            result["message"] = "No matches found"
            
        return result
        
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception during face recognition: {str(err)} - {tb_str}"}, 400


def remove_by_label(
    label: str,
):
    try:
        label_directory = os.path.join(REGISTERED_FACES_DIR, label)

        if os.path.exists(label_directory):
            shutil.rmtree(label_directory)

        result = {
            "success": True,
            "message": f"Successfully removed all faces for label '{label}'"
        }
        
        return result
        
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception during face removal: {str(err)} - {tb_str}"}, 400


