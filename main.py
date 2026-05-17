from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import tensorflow as tf
import numpy as np
from PIL import Image

import io
import os
import time
import logging

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("PlantDoc")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================================================
# GLOBAL REQUEST/RESPONSE LOGGER
# =========================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.time()

    logger.info(
        f"REQUEST -> METHOD={request.method} "
        f"PATH={request.url.path}"
    )

    try:
        response = await call_next(request)

        process_time = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"RESPONSE -> METHOD={request.method} "
            f"PATH={request.url.path} "
            f"STATUS={response.status_code} "
            f"TIME={process_time}ms"
        )

        return response

    except Exception as e:

        logger.error(
            f"SERVER ERROR -> {str(e)}",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error"
            }
        )

# =========================================================
# MODEL LOADING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_model.h5")

logger.info("Loading TensorFlow model...")

model = tf.keras.models.load_model(MODEL_PATH)

logger.info("Model loaded successfully!")

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# =========================================================
# PYDANTIC MODEL
# =========================================================

class ChatRequest(BaseModel):
    message: str

# =========================================================
# HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    response = {
        "message": "PlantDoc API is running!"
    }

    logger.info(f"HOME RESPONSE -> {response}")

    return response

# =========================================================
# CONFIG
# =========================================================

ALLOWED_EXTENSIONS = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp"
}

CONFIDENCE_THRESHOLD = 0.70

# =========================================================
# LEAF DETECTION FUNCTION
# =========================================================

def is_plant_leaf(image: Image.Image) -> bool:

    img_rgb = image.convert("RGB")

    img_array = np.array(img_rgb)

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    green_pixels = np.sum(
        (g > r + 15) & (g > b + 15)
    )

    total_pixels = img_array.shape[0] * img_array.shape[1]

    green_ratio = green_pixels / total_pixels

    logger.info(
        f"GREEN PIXEL RATIO -> {green_ratio:.2f}"
    )

    return green_ratio > 0.10

# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    logger.info("PREDICT ENDPOINT CALLED")

    logger.info(f"FILENAME -> {file.filename}")
    logger.info(f"CONTENT TYPE -> {file.content_type}")

    # -----------------------------------------------------
    # FILE TYPE VALIDATION
    # -----------------------------------------------------

    if file.content_type not in ALLOWED_EXTENSIONS:

        response = {
            "error": "Invalid file type. Upload JPG/PNG/WebP image.",
            "disease": None,
            "confidence": None
        }

        logger.warning(f"INVALID FILE TYPE -> {response}")

        return response

    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    contents = await file.read()

    file_size_mb = len(contents) / (1024 * 1024)

    logger.info(
        f"FILE SIZE -> {file_size_mb:.2f} MB"
    )

    # -----------------------------------------------------
    # FILE SIZE CHECK
    # -----------------------------------------------------

    if len(contents) > 10 * 1024 * 1024:

        response = {
            "error": "File too large. Max size is 10MB.",
            "disease": None,
            "confidence": None
        }

        logger.warning(f"FILE TOO LARGE -> {response}")

        return response

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    try:

        image = Image.open(io.BytesIO(contents))

        logger.info(
            f"IMAGE OPENED -> "
            f"SIZE={image.size} "
            f"MODE={image.mode}"
        )

    except Exception as e:

        response = {
            "error": "Could not open image.",
            "disease": None,
            "confidence": None
        }

        logger.error(
            f"IMAGE OPEN ERROR -> {str(e)}"
        )

        return response

    # -----------------------------------------------------
    # LEAF VALIDATION
    # -----------------------------------------------------

    if not is_plant_leaf(image):

        response = {
            "error": "This does not appear to be a plant leaf image.",
            "disease": None,
            "confidence": None
        }

        logger.warning(
            f"NOT A LEAF IMAGE -> {response}"
        )

        return response

    # -----------------------------------------------------
    # PREPROCESS IMAGE
    # -----------------------------------------------------

    image_resized = image.resize((128, 128)).convert("RGB")

    logger.info("IMAGE RESIZED -> 128x128")

    input_arr = np.array([np.array(image_resized)])

    logger.info(
        f"INPUT SHAPE -> {input_arr.shape}"
    )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    logger.info("RUNNING MODEL PREDICTION...")

    predictions = model.predict(input_arr)

    result_index = np.argmax(predictions)

    confidence = float(np.max(predictions))

    predicted_class = CLASS_NAMES[result_index]

    logger.info(
        f"PREDICTION RESULT -> {predicted_class}"
    )

    logger.info(
        f"CONFIDENCE -> {confidence:.4f}"
    )

    # -----------------------------------------------------
    # CONFIDENCE CHECK
    # -----------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        response = {
            "error": "Low confidence prediction. Upload clearer image.",
            "disease": None,
            "confidence": round(confidence * 100, 2)
        }

        logger.warning(
            f"LOW CONFIDENCE RESPONSE -> {response}"
        )

        return response

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    response = {
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
        "error": None
    }

    logger.info(
        f"FINAL RESPONSE -> {response}"
    )

    return response

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    logger.info("CHAT ENDPOINT CALLED")

    message = request.message.lower().strip()

    logger.info(f"USER MESSAGE -> {message}")

    # =====================================================
    # SIMPLE DEMO CHAT DATABASE
    # =====================================================

    DISEASE_DB = {

        "potato early blight": {
            "about": "Potato Early Blight is caused by Alternaria solani fungus.",
            "prevent": "Rotate crops and avoid wet leaves.",
            "treat": "Use chlorothalonil or mancozeb fungicide."
        },

        "tomato late blight": {
            "about": "Tomato Late Blight is caused by Phytophthora infestans.",
            "prevent": "Avoid overhead watering and use resistant varieties.",
            "treat": "Apply copper fungicide immediately."
        },

        "powdery mildew": {
            "about": "Powdery Mildew appears as white powder on leaves.",
            "prevent": "Ensure proper air circulation.",
            "treat": "Use sulfur fungicide or neem oil."
        }
    }

    # =====================================================
    # DISEASE FINDER
    # =====================================================

    def find_disease(msg):

        if "potato early blight" in msg:
            return "potato early blight"

        elif "tomato late blight" in msg:
            return "tomato late blight"

        elif "powdery mildew" in msg:
            return "powdery mildew"

        return None

    disease = find_disease(message)

    logger.info(f"DISEASE DETECTED -> {disease}")

    # =====================================================
    # GREETING
    # =====================================================

    if any(word in message for word in ["hello", "hi", "hey"]):

        response = {
            "reply": "🌿 Hello! Welcome to PlantDoc AI Assistant!"
        }

        logger.info(f"CHAT RESPONSE -> {response}")

        return response

    # =====================================================
    # PREVENTION
    # =====================================================

    elif "prevent" in message and disease:

        response = {
            "reply": DISEASE_DB[disease]["prevent"]
        }

        logger.info(f"PREVENT RESPONSE -> {response}")

        return response

    # =====================================================
    # TREATMENT
    # =====================================================

    elif "treat" in message and disease:

        response = {
            "reply": DISEASE_DB[disease]["treat"]
        }

        logger.info(f"TREAT RESPONSE -> {response}")

        return response

    # =====================================================
    # ABOUT DISEASE
    # =====================================================

    elif disease:

        response = {
            "reply": DISEASE_DB[disease]["about"]
        }

        logger.info(f"ABOUT RESPONSE -> {response}")

        return response

    # =====================================================
    # UNKNOWN MESSAGE
    # =====================================================

    else:

        response = {
            "reply": "Sorry, I didn't understand your request."
        }

        logger.warning(
            f"UNKNOWN USER MESSAGE -> {message}"
        )

        logger.info(f"CHAT RESPONSE -> {response}")

        return response

# =========================================================
# START MESSAGE
# =========================================================

logger.info("PlantDoc Backend Server Started Successfully!")