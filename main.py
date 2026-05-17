from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'plant_model.h5'))

CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "PlantDoc API is running!"}

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
CONFIDENCE_THRESHOLD = 0.70  # 70% — adjust if needed

def is_plant_leaf(image: Image.Image) -> bool:
    """Check if image has enough green pixels to be a plant leaf."""
    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    green_pixels = np.sum((g > r + 15) & (g > b + 15))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.10  # at least 10% green pixels

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Check file type
    if file.content_type not in ALLOWED_EXTENSIONS:
        return {
            "error": "Invalid file type. Please upload a JPG or PNG image.",
            "disease": None,
            "confidence": None
        }

    contents = await file.read()

    # 2. Check file size (max 10MB)
    if len(contents) > 10 * 1024 * 1024:
        return {
            "error": "File too large. Please upload an image under 10MB.",
            "disease": None,
            "confidence": None
        }

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        return {
            "error": "Could not open image. Please upload a valid image file.",
            "disease": None,
            "confidence": None
        }

    # 3. Check if image looks like a plant leaf (green pixel check)
    if not is_plant_leaf(image):
        return {
            "error": "This does not appear to be a plant leaf image. Please upload a clear photo of a plant leaf.",
            "disease": None,
            "confidence": None
        }

    # 4. Run the model
    image_resized = image.resize((128, 128)).convert("RGB")
    input_arr = np.array([np.array(image_resized)])
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    warning = None
    if confidence < CONFIDENCE_THRESHOLD:
        warning = "Low confidence result. For better accuracy, upload a clearer, closer photo of the leaf."

    return {
        "disease": CLASS_NAMES[result_index],
        "confidence": round(confidence * 100, 2),
        "error": None,
        "warning": warning
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.lower().strip()
    
    DISEASE_DB = {
        "apple scab": {
            "about": "🍎 Apple Scab Disease\n\nWhat is it?\nApple Scab is one of the most serious diseases of apple trees worldwide, caused by the fungus Venturia inaequalis.\n\nHow does it happen?\n• The fungus overwinters in infected fallen leaves\n• In spring, spores are released and carried by wind and rain\n• Infection occurs when leaves stay wet for 6+ hours\n• Cool temperatures (15-25°C) speed up spreading\n• Dark olive-green spots appear on leaves and fruit\n• Infected fruit becomes deformed and cracked\n\nWould you like to know:\n1️⃣ How to PREVENT Apple Scab?\n2️⃣ How to TREAT Apple Scab?",
            "prevent": "🛡️ How to PREVENT Apple Scab\n\n• Plant resistant apple varieties like Liberty or Freedom\n• Rake and destroy all fallen leaves in autumn\n• Prune trees to improve air circulation\n• Apply fungicide spray starting from bud break in spring\n• Avoid overhead irrigation — water at the base only\n• Apply spray every 7-10 days during wet spring weather\n• Use copper-based fungicide as a preventive measure\n\nWould you like to know how to TREAT Apple Scab? Type 'treat apple scab'",
            "treat": "💊 How to TREAT Apple Scab\n\n• Apply fungicide (myclobutanil or captan) immediately\n• Spray every 7-10 days during wet weather\n• Remove and destroy all infected leaves and fruit\n• Do NOT compost infected material — burn or bag it\n• Continue treatment until dry weather arrives\n• Apply copper spray as follow-up protection\n• Monitor tree regularly for new infections\n\nRecovery time: 2-4 weeks with proper treatment"
        },
        "apple black rot": {
            "about": "🍎 Apple Black Rot Disease\n\nWhat is it?\nApple Black Rot is a serious fungal disease caused by Botryosphaeria obtusa that affects leaves, fruit and bark.\n\nHow does it happen?\n• Fungus survives in dead wood and mummified fruit\n• Spores spread during warm, humid weather (24-29°C)\n• Enters through wounds, insect damage or natural openings\n• Brown circular spots appear on leaves with purple borders\n• Fruit turns black and shrivels into hard mummies\n• Infected bark develops reddish-brown cankers\n\nWould you like to know:\n1️⃣ How to PREVENT Apple Black Rot?\n2️⃣ How to TREAT Apple Black Rot?",
            "prevent": "🛡️ How to PREVENT Apple Black Rot\n\n• Prune out all dead and diseased wood every winter\n• Remove all mummified fruit from trees and ground\n• Apply wound sealant after pruning\n• Maintain tree vigor with proper fertilization\n• Apply protective fungicide from pink bud stage\n• Control insects that create entry wounds\n• Keep orchard clean and free of plant debris\n\nWould you like to know how to TREAT Apple Black Rot? Type 'treat apple black rot'",
            "treat": "💊 How to TREAT Apple Black Rot\n\n• Remove all infected fruit, leaves and branches immediately\n• Apply copper-based fungicide every 10-14 days\n• Cut out cankers from bark and apply wound sealant\n• Destroy all removed plant material — do not compost\n• Apply captan or thiophanate-methyl fungicide\n• Repeat treatment throughout growing season\n• Monitor closely after heavy rain periods\n\nRecovery time: 3-6 weeks with consistent treatment"
        },
        "potato early blight": {
            "about": "🥔 Potato Early Blight Disease\n\nWhat is it?\nPotato Early Blight is a very common fungal disease caused by Alternaria solani that affects potato and tomato plants.\n\nHow does it happen?\n• Fungus survives in infected plant debris in soil\n• Warm temperatures (24-29°C) with high humidity trigger it\n• Spores spread through wind, rain splash and insects\n• Older lower leaves are attacked first\n• Dark brown spots with yellow halos appear on leaves\n• Spots have distinctive concentric ring pattern like a target\n• Severely infected leaves turn yellow and drop off\n• Yield can be reduced by up to 50% in severe cases\n\nWould you like to know:\n1️⃣ How to PREVENT Potato Early Blight?\n2️⃣ How to TREAT Potato Early Blight?",
            "prevent": "🛡️ How to PREVENT Potato Early Blight\n\n• Use certified disease-free seed potatoes\n• Rotate crops — do not plant potato in same spot for 3 years\n• Ensure adequate nitrogen fertilization for plant vigor\n• Apply preventive fungicide before symptoms appear\n• Water plants at base — avoid wetting leaves\n• Remove and destroy infected plant debris after harvest\n• Plant in well-drained soil with good air circulation\n• Choose resistant potato varieties when available\n\nWould you like to know how to TREAT Potato Early Blight? Type 'treat potato early blight'",
            "treat": "💊 How to TREAT Potato Early Blight\n\n• Apply fungicide (chlorothalonil or mancozeb) immediately\n• Spray every 7-10 days throughout growing season\n• Remove infected lower leaves to slow disease spread\n• Water only at base of plants — never overhead\n• Apply copper-based fungicide as additional protection\n• Ensure plants receive adequate nutrition\n• Destroy all infected plant material after harvest\n• Do not save infected tubers for next season\n\nRecovery time: 2-3 weeks with proper treatment"
        },
        "potato late blight": {
            "about": "🥔 Potato Late Blight Disease\n\nWhat is it?\nPotato Late Blight is caused by Phytophthora infestans — the same disease that caused the devastating Irish Potato Famine in 1845!\n\nHow does it happen?\n• Pathogen survives in infected tubers left in soil\n• Cool temperatures (10-20°C) with high humidity trigger it\n• Spreads extremely rapidly in wet, foggy weather\n• Large dark water-soaked lesions appear on leaves\n• White fuzzy growth visible on leaf undersides\n• Brown rot spreads to stems and tubers quickly\n• Entire field can be destroyed within days\n• One of the most destructive plant diseases in history\n\nWould you like to know:\n1️⃣ How to PREVENT Potato Late Blight?\n2️⃣ How to TREAT Potato Late Blight?",
            "prevent": "🛡️ How to PREVENT Potato Late Blight\n\n• Use certified disease-free seed potatoes only\n• Plant resistant varieties like Sarpo Mira or Defender\n• Avoid overhead irrigation — water at base only\n• Hill up soil around plants to protect tubers\n• Apply preventive copper fungicide before wet weather\n• Monitor weather forecasts — spray before rainy periods\n• Ensure good drainage in planting area\n• Remove volunteer potato plants from previous season\n\nWould you like to know how to TREAT Potato Late Blight? Type 'treat potato late blight'",
            "treat": "💊 How to TREAT Potato Late Blight\n\n• Act IMMEDIATELY — this disease spreads very fast!\n• Apply fungicide (metalaxyl or chlorothalonil) right away\n• Spray every 5-7 days during wet weather conditions\n• Remove and DESTROY all infected plants — do not compost!\n• Do not harvest tubers for at least 2 weeks after treatment\n• Check stored tubers regularly for signs of rot\n• Burn or deeply bury all infected plant material\n• Report severe outbreaks to local agricultural authority\n\nRecovery time: Very difficult — prevention is much better than cure!"
        },
        "tomato early blight": {
            "about": "🍅 Tomato Early Blight Disease\n\nWhat is it?\nTomato Early Blight is caused by the fungus Alternaria solani — the same fungus that causes Potato Early Blight.\n\nHow does it happen?\n• Fungus lives in infected plant debris and soil\n• Warm temperatures (24-29°C) with wet conditions activate it\n• Older lower leaves are infected first\n• Dark brown spots with distinctive target-board ring pattern\n• Yellow area surrounds each dark spot\n• Leaves turn yellow and fall off progressively upward\n• Fruit can also develop dark sunken lesions near stem\n• Weakened plants produce less fruit\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Early Blight?\n2️⃣ How to TREAT Tomato Early Blight?",
            "prevent": "🛡️ How to PREVENT Tomato Early Blight\n\n• Rotate tomatoes to different bed every 2-3 years\n• Use mulch around plants to prevent soil splash\n• Stake or cage plants for better air circulation\n• Water at base — never wet the leaves\n• Apply preventive fungicide at first sign of wet weather\n• Remove lower leaves touching the soil\n• Plant resistant tomato varieties\n• Space plants adequately for good air flow\n\nWould you like to know how to TREAT Tomato Early Blight? Type 'treat tomato early blight'",
            "treat": "💊 How to TREAT Tomato Early Blight\n\n• Apply fungicide (chlorothalonil or copper) immediately\n• Spray every 7-10 days — more often in rainy weather\n• Remove all infected leaves starting from bottom\n• Dispose of infected leaves — do not compost them\n• Apply neem oil as organic alternative\n• Ensure plants are well-fertilized for strong immunity\n• Continue treatment until harvest\n• Clean up all plant debris at end of season\n\nRecovery time: 2-4 weeks with consistent treatment"
        },
        "tomato late blight": {
            "about": "🍅 Tomato Late Blight Disease\n\nWhat is it?\nTomato Late Blight is caused by Phytophthora infestans — an extremely destructive disease that can destroy entire crops rapidly.\n\nHow does it happen?\n• Pathogen thrives in cool, wet conditions (10-20°C)\n• Spreads through wind-blown spores over long distances\n• Large dark water-soaked patches appear on leaves\n• White fuzzy mold grows on underside of leaves\n• Stems develop dark brown lesions and collapse\n• Fruit develops large brown greasy-looking patches\n• Can destroy entire crop within 1-2 weeks\n• Spreads to neighboring farms through air\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Late Blight?\n2️⃣ How to TREAT Tomato Late Blight?",
            "prevent": "🛡️ How to PREVENT Tomato Late Blight\n\n• Plant resistant varieties like Legend or Mountain Magic\n• Avoid overhead watering — use drip irrigation\n• Ensure excellent air circulation between plants\n• Apply copper fungicide preventively before wet season\n• Monitor weather — spray before forecasted rain\n• Remove any volunteer tomato plants nearby\n• Never plant tomatoes near potatoes\n• Inspect plants daily during cool, wet weather\n\nWould you like to know how to TREAT Tomato Late Blight? Type 'treat tomato late blight'",
            "treat": "💊 How to TREAT Tomato Late Blight\n\n• Act IMMEDIATELY — disease spreads extremely fast!\n• Apply copper fungicide or chlorothalonil right away\n• Spray every 5-7 days in wet weather conditions\n• Remove and DESTROY all infected plants immediately\n• Never compost infected material — burn or bag it\n• Wash hands and tools after handling infected plants\n• Check neighboring plants for signs of spread\n• Consider removing entire crop if infection is severe\n\nRecovery time: Extremely difficult — act fast or lose entire crop!"
        },
        "tomato bacterial spot": {
            "about": "🍅 Tomato Bacterial Spot Disease\n\nWhat is it?\nTomato Bacterial Spot is caused by Xanthomonas vesicatoria bacteria — it affects leaves, stems and fruit of tomato plants.\n\nHow does it happen?\n• Bacteria survive in infected seed and plant debris\n• Warm temperatures (24-30°C) with rain trigger spreading\n• Spreads through rain splash, wind and contaminated tools\n• Small water-soaked spots appear on leaves\n• Spots turn brown with yellow halo around them\n• Severely infected leaves turn yellow and drop\n• Fruit develops raised scabby spots reducing market value\n• Bacteria enter through natural openings and wounds\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Bacterial Spot?\n2️⃣ How to TREAT Tomato Bacterial Spot?",
            "prevent": "🛡️ How to PREVENT Tomato Bacterial Spot\n\n• Use certified disease-free seeds and transplants\n• Rotate crops — avoid tomatoes in same spot for 2-3 years\n• Avoid overhead watering — use drip irrigation\n• Apply copper spray preventively at transplanting\n• Do not work in garden when plants are wet\n• Disinfect tools with bleach solution regularly\n• Remove plant debris thoroughly after harvest\n• Choose resistant tomato varieties when possible\n\nWould you like to know how to TREAT Tomato Bacterial Spot? Type 'treat tomato bacterial spot'",
            "treat": "💊 How to TREAT Tomato Bacterial Spot\n\n• Apply copper-based bactericide plus mancozeb weekly\n• Begin treatment at very first sign of disease\n• Remove and destroy all heavily infected leaves\n• Avoid working with plants when they are wet\n• Apply treatment every 7 days during wet weather\n• Disinfect all tools after use with 10% bleach solution\n• Remove plant debris completely at end of season\n• Do not save seeds from infected plants\n\nRecovery time: 2-4 weeks with consistent copper treatment"
        },
        "grape black rot": {
            "about": "🍇 Grape Black Rot Disease\n\nWhat is it?\nGrape Black Rot is caused by the fungus Guignardia bidwellii — it is one of the most destructive grape diseases in warm, humid climates.\n\nHow does it happen?\n• Fungus overwinters in mummified berries and infected wood\n• Warm temperatures (26-29°C) with wet weather activate spores\n• Spores released during rain and spread by wind\n• Reddish-brown circular spots appear on leaves\n• Infected berries turn brown then shrivel into black mummies\n• Mummified berries stay on vine and spread disease next year\n• Can destroy 100% of fruit crop in severe cases\n• Most damage occurs from bloom to 4 weeks after\n\nWould you like to know:\n1️⃣ How to PREVENT Grape Black Rot?\n2️⃣ How to TREAT Grape Black Rot?",
            "prevent": "🛡️ How to PREVENT Grape Black Rot\n\n• Remove all mummified berries from vines and ground\n• Prune vines properly for good air circulation\n• Apply fungicide starting from bud break in spring\n• Spray every 10-14 days through fruit development\n• Remove all infected plant material after harvest\n• Choose resistant grape varieties where possible\n• Avoid planting in low-lying areas with poor drainage\n• Monitor vines regularly especially after rain\n\nWould you like to know how to TREAT Grape Black Rot? Type 'treat grape black rot'",
            "treat": "💊 How to TREAT Grape Black Rot\n\n• Apply fungicide (myclobutanil or mancozeb) immediately\n• Spray every 10-14 days during growing season\n• Remove ALL mummified fruit from vines and ground\n• Prune out infected shoots and canes\n• Destroy removed material — do not compost\n• Continue spraying until grapes reach full size\n• Apply copper spray as additional protection\n• Keep detailed records of treatment for next season\n\nRecovery time: Current season fruit may be lost — focus on preventing next season"
        },
        "corn common rust": {
            "about": "🌽 Corn Common Rust Disease\n\nWhat is it?\nCorn Common Rust is caused by the fungus Puccinia sorghi — it appears as small powdery pustules on corn leaves.\n\nHow does it happen?\n• Fungus spreads through airborne spores from southern regions\n• Cool temperatures (15-21°C) with high humidity favor infection\n• Spores land on leaves and germinate in moisture\n• Brick-red oval pustules appear on both sides of leaves\n• Pustules release more spores that spread to other plants\n• Severe infection causes leaves to yellow and die early\n• Reduces photosynthesis and grain fill\n• More severe in early-planted or stressed crops\n\nWould you like to know:\n1️⃣ How to PREVENT Corn Common Rust?\n2️⃣ How to TREAT Corn Common Rust?",
            "prevent": "🛡️ How to PREVENT Corn Common Rust\n\n• Plant resistant corn hybrids — most modern hybrids have resistance\n• Plant early to avoid peak rust season\n• Monitor fields regularly especially after cool, wet periods\n• Maintain proper plant nutrition especially potassium\n• Avoid excessive nitrogen which increases susceptibility\n• Apply preventive fungicide in high-risk areas\n• Scout fields weekly from V6 stage onwards\n• Keep records of rust pressure for future planning\n\nWould you like to know how to TREAT Corn Common Rust? Type 'treat corn common rust'",
            "treat": "💊 How to TREAT Corn Common Rust\n\n• Apply foliar fungicide (strobilurin or triazole) early\n• Treatment most effective before tasseling stage\n• Focus spray on upper leaves where rust is most damaging\n• Apply when pustules first appear on lower leaves\n• One application usually sufficient if done early\n• Use resistant hybrids in next planting season\n• Economic threshold: treat when 50% plants show rust before silking\n• Late-season infection after dough stage — treatment not usually needed\n\nRecovery time: 2-3 weeks — yield loss minimal if treated before tasseling"
        },
        "strawberry leaf scorch": {
            "about": "🍓 Strawberry Leaf Scorch Disease\n\nWhat is it?\nStrawberry Leaf Scorch is caused by the fungus Diplocarpon earlianum — it makes strawberry leaves look burnt or scorched.\n\nHow does it happen?\n• Fungus overwinters in infected leaves on the ground\n• Cool, wet spring weather (18-24°C) triggers spore release\n• Spores spread through rain splash and wind\n• Small dark purple spots appear on upper leaf surface\n• Spots have reddish-purple borders that look scorched\n• Centers of spots turn gray or white as disease progresses\n• Severely infected leaves curl, wither and die\n• Plant vigor and fruit production reduced significantly\n\nWould you like to know:\n1️⃣ How to PREVENT Strawberry Leaf Scorch?\n2️⃣ How to TREAT Strawberry Leaf Scorch?",
            "prevent": "🛡️ How to PREVENT Strawberry Leaf Scorch\n\n• Plant resistant strawberry varieties\n• Remove and destroy old infected leaves after harvest\n• Renovate strawberry beds by mowing and thinning\n• Avoid overhead irrigation — use drip system\n• Ensure good air circulation between plants\n• Apply preventive fungicide in early spring\n• Maintain proper plant spacing for air flow\n• Remove all plant debris at end of season\n\nWould you like to know how to TREAT Strawberry Leaf Scorch? Type 'treat strawberry leaf scorch'",
            "treat": "💊 How to TREAT Strawberry Leaf Scorch\n\n• Apply fungicide (captan or myclobutanil) every 10-14 days\n• Begin treatment at first sign of purple spotting\n• Remove and destroy all heavily infected leaves\n• Avoid wetting foliage when watering plants\n• Apply neem oil as organic alternative treatment\n• Continue treatment through entire growing season\n• Renovate beds after harvest to remove infected material\n• Replace severely infected beds with new certified plants\n\nRecovery time: 3-4 weeks with consistent fungicide application"
        },
        "tomato leaf mold": {
            "about": "🍅 Tomato Leaf Mold Disease\n\nWhat is it?\nTomato Leaf Mold is caused by the fungus Passalora fulva — it is most common in greenhouse tomatoes where humidity is high.\n\nHow does it happen?\n• Fungus thrives when humidity exceeds 85%\n• Spores spread through air, water and contaminated tools\n• Yellow patches appear on upper surface of leaves\n• Distinctive olive-green to brown velvety mold grows beneath\n• Infected leaves curl upward and eventually die\n• Disease progresses from older lower leaves upward\n• Fruit infection rare but can occur in severe cases\n• Warm temperatures (21-24°C) with high humidity are ideal for spread\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Leaf Mold?\n2️⃣ How to TREAT Tomato Leaf Mold?",
            "prevent": "🛡️ How to PREVENT Tomato Leaf Mold\n\n• Maintain greenhouse humidity below 85% at all times\n• Ensure excellent ventilation in greenhouse\n• Space plants adequately for good air circulation\n• Water early in day so leaves dry before evening\n• Use drip irrigation instead of overhead watering\n• Plant resistant tomato varieties\n• Remove lower leaves to improve air flow\n• Disinfect greenhouse structures between seasons\n\nWould you like to know how to TREAT Tomato Leaf Mold? Type 'treat tomato leaf mold'",
            "treat": "💊 How to TREAT Tomato Leaf Mold\n\n• Improve ventilation immediately to reduce humidity\n• Apply fungicide (chlorothalonil or copper) every 7 days\n• Remove all infected leaves from plant and greenhouse\n• Avoid wetting leaves when watering\n• Apply sulfur-based fungicide as organic option\n• Reduce plant density by removing some plants\n• Disinfect all tools and surfaces regularly\n• Monitor humidity daily with hygrometer\n\nRecovery time: 2-3 weeks with improved ventilation and fungicide"
        },
        "tomato septoria": {
            "about": "🍅 Tomato Septoria Leaf Spot Disease\n\nWhat is it?\nTomato Septoria Leaf Spot is caused by the fungus Septoria lycopersici — it is one of the most common tomato diseases worldwide.\n\nHow does it happen?\n• Fungus survives in infected plant debris and soil\n• Warm, wet weather (20-25°C) triggers spore release\n• Spores spread through rain splash from soil to leaves\n• Small circular spots with dark borders appear on lower leaves\n• Spots have tan or gray centers with small black dots inside\n• Disease progresses rapidly upward through the plant\n• Infected leaves turn yellow then brown and fall off\n• Defoliation weakens plant and exposes fruit to sunscald\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Septoria Leaf Spot?\n2️⃣ How to TREAT Tomato Septoria Leaf Spot?",
            "prevent": "🛡️ How to PREVENT Tomato Septoria Leaf Spot\n\n• Rotate tomatoes to new location every 2-3 years\n• Apply thick mulch layer to prevent soil splash\n• Stake or cage plants to keep them off ground\n• Water at base only — never wet the foliage\n• Remove lower leaves before they touch the soil\n• Apply preventive copper spray early in season\n• Clean up all plant debris after harvest\n• Choose resistant varieties when available\n\nWould you like to know how to TREAT Tomato Septoria? Type 'treat tomato septoria'",
            "treat": "💊 How to TREAT Tomato Septoria Leaf Spot\n\n• Apply fungicide (chlorothalonil or copper) immediately\n• Spray every 7-10 days throughout growing season\n• Remove all infected leaves starting from bottom of plant\n• Dispose of infected material — never compost it\n• Apply organic neem oil as alternative treatment\n• Maintain consistent treatment schedule even in dry weather\n• Ensure plants are well-watered and fertilized\n• Remove plant debris completely at season end\n\nRecovery time: 3-4 weeks — disease rarely kills plant but reduces yield"
        },
        "spider mites": {
            "about": "🕷️ Spider Mites on Tomato\n\nWhat is it?\nSpider Mites (Two-Spotted Spider Mite) are tiny arachnids, not insects. They are one of the most destructive tomato pests in hot, dry conditions.\n\nHow does it happen?\n• Mites thrive in hot, dry weather (above 27°C)\n• They reproduce extremely rapidly — new generation every 5-7 days!\n• Spread from plant to plant through wind, clothing and tools\n• Tiny yellow or white stippling appears on leaf surface\n• Fine webbing visible on underside of heavily infested leaves\n• Leaves turn bronze, then brown and drop off\n• Severe infestation can defoliate plant completely\n• Pesticide overuse kills natural predators making problem worse\n\nWould you like to know:\n1️⃣ How to PREVENT Spider Mites?\n2️⃣ How to TREAT Spider Mites?",
            "prevent": "🛡️ How to PREVENT Spider Mites\n\n• Keep plants well-watered — mites prefer stressed plants\n• Maintain humidity around plants with regular misting\n• Encourage natural predators like ladybugs and predatory mites\n• Avoid excessive nitrogen fertilizer which attracts mites\n• Avoid broad-spectrum pesticides that kill natural predators\n• Inspect plants regularly especially undersides of leaves\n• Remove heavily infested leaves early before spread\n• Keep garden area free of weeds that harbor mites\n\nWould you like to know how to TREAT Spider Mites? Type 'treat spider mites'",
            "treat": "💊 How to TREAT Spider Mites\n\n• Spray plants forcefully with water to knock mites off\n• Apply insecticidal soap solution to all leaf surfaces\n• Focus spray on undersides of leaves where mites live\n• Apply neem oil every 5-7 days as organic treatment\n• Use miticide (abamectin or bifenazate) for severe infestations\n• Introduce predatory mites (Phytoseiidae) as biological control\n• Rotate between different miticides to prevent resistance\n• Remove and destroy heavily infested plant parts\n\nRecovery time: 1-2 weeks with consistent treatment — act fast as they multiply quickly!"
        },
        "tomato target spot": {
            "about": "🍅 Tomato Target Spot Disease\n\nWhat is it?\nTomato Target Spot is caused by the fungus Corynespora cassiicola — it creates distinctive circular lesions that look like a target or bullseye.\n\nHow does it happen?\n• Fungus survives in infected plant debris in soil\n• Warm temperatures (24-30°C) with high humidity activate it\n• Spores spread through wind, rain splash and infected tools\n• Circular spots with concentric ring pattern appear on leaves\n• Spots start yellow then develop dark brown centers\n• Target-like rings give the disease its distinctive name\n• Severely infected leaves drop off prematurely\n• Fruit can also be infected with sunken dark lesions\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Target Spot?\n2️⃣ How to TREAT Tomato Target Spot?",
            "prevent": "🛡️ How to PREVENT Tomato Target Spot\n\n• Rotate tomatoes to different location every 2-3 years\n• Ensure good air circulation by proper plant spacing\n• Avoid overhead watering — use drip irrigation\n• Apply preventive fungicide during humid weather\n• Remove plant debris thoroughly after harvest\n• Avoid working in garden when plants are wet\n• Stake plants to keep foliage off the ground\n• Monitor plants regularly especially in humid conditions\n\nWould you like to know how to TREAT Tomato Target Spot? Type 'treat tomato target spot'",
            "treat": "💊 How to TREAT Tomato Target Spot\n\n• Apply fungicide (chlorothalonil or azoxystrobin) immediately\n• Spray every 7-10 days during wet weather periods\n• Remove all infected leaves from plant carefully\n• Dispose of infected material away from garden\n• Apply copper-based fungicide as alternative\n• Maintain consistent treatment schedule\n• Ensure plants receive adequate nutrition for recovery\n• Clean and disinfect tools after working with infected plants\n\nRecovery time: 2-4 weeks with proper fungicide treatment"
        },
        "yellow leaf curl": {
            "about": "🍅 Tomato Yellow Leaf Curl Virus\n\nWhat is it?\nTomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease spread by whiteflies. It can destroy entire tomato crops.\n\nHow does it happen?\n• Virus is transmitted by Bemisia tabaci whitefly\n• Whiteflies pick up virus from infected plants and spread it\n• Virus enters plant through feeding wounds\n• Young leaves curl upward and turn yellow at edges\n• Plants become stunted with small crinkled leaves\n• Flowers drop off without setting fruit\n• Infected plants produce little or no harvestable fruit\n• Virus spreads rapidly through whitefly populations\n\nWould you like to know:\n1️⃣ How to PREVENT Yellow Leaf Curl Virus?\n2️⃣ How to TREAT Yellow Leaf Curl Virus?",
            "prevent": "🛡️ How to PREVENT Tomato Yellow Leaf Curl Virus\n\n• Control whitefly populations with insecticide immediately\n• Use yellow sticky traps to monitor and catch whiteflies\n• Plant resistant tomato varieties (TY varieties)\n• Use reflective mulch to repel whiteflies\n• Install insect-proof mesh in greenhouse growing\n• Remove infected plants immediately before whiteflies spread virus\n• Plant away from other solanaceous crops\n• Inspect transplants carefully before planting\n\nWould you like to know how to TREAT Yellow Leaf Curl? Type 'treat yellow leaf curl'",
            "treat": "💊 How to TREAT Tomato Yellow Leaf Curl Virus\n\n• IMPORTANT: There is NO cure for viral diseases!\n• Focus on controlling whitefly vector immediately\n• Apply insecticide (imidacloprid or thiamethoxam) for whiteflies\n• Remove and destroy all infected plants to prevent spread\n• Use insecticidal soap on remaining plants for whitefly control\n• Install yellow sticky traps throughout garden\n• Do not save seeds from infected plants\n• Plant resistant varieties in next growing season\n\nNote: Infected plants will not recover — remove them to protect healthy plants!"
        },
        "mosaic virus": {
            "about": "🍅 Tomato Mosaic Virus\n\nWhat is it?\nTomato Mosaic Virus (ToMV) is a highly contagious viral disease that can survive for years in soil and plant debris.\n\nHow does it happen?\n• Virus spreads easily through touch, tools and clothing\n• Transmitted through infected seeds and transplants\n• Spreads through sap when handling plants — even touching!\n• Light and dark green mosaic pattern appears on leaves\n• Leaves may be distorted, curled or fern-like in appearance\n• Plants become stunted with reduced fruit production\n• Fruit may show yellow spots or internal browning\n• Virus can survive in dried plant tissue for years\n\nWould you like to know:\n1️⃣ How to PREVENT Tomato Mosaic Virus?\n2️⃣ How to TREAT Tomato Mosaic Virus?",
            "prevent": "🛡️ How to PREVENT Tomato Mosaic Virus\n\n• Use certified virus-free seeds and transplants only\n• Wash hands thoroughly with soap before handling plants\n• Disinfect all tools with 10% bleach or 70% alcohol\n• Do not use tobacco products near tomato plants\n• Remove and destroy infected plants immediately\n• Plant resistant varieties (TMV resistant)\n• Rotate crops regularly\n• Control aphids and other insects that may spread virus\n\nWould you like to know how to TREAT Mosaic Virus? Type 'treat mosaic virus'",
            "treat": "💊 How to TREAT Tomato Mosaic Virus\n\n• IMPORTANT: There is NO chemical cure for viral diseases!\n• Remove and destroy all infected plants immediately\n• Disinfect tools with bleach solution after every use\n• Wash hands thoroughly after handling infected plants\n• Do not compost infected material — burn or bag it\n• Control insect vectors with appropriate insecticides\n• Plant resistant varieties in following season\n• Do not save seeds from infected plants\n\nNote: Focus on prevention — once infected, plants cannot be cured!"
        },
        "powdery mildew": {
            "about": "🌿 Powdery Mildew Disease\n\nWhat is it?\nPowdery Mildew is caused by various fungal species — it appears as white powdery coating on leaves and is one of the most common plant diseases worldwide.\n\nHow does it happen?\n• Fungus thrives in warm days (20-27°C) with cool nights\n• Unlike most fungi, it does NOT need rain to spread\n• High humidity WITHOUT wet conditions is ideal\n• White powdery spots appear on upper leaf surface\n• Spots spread to cover entire leaf with white powder\n• Infected leaves turn yellow then brown and drop\n• Severely infected plants become stunted and weak\n• Spores spread easily through air to nearby plants\n\nWould you like to know:\n1️⃣ How to PREVENT Powdery Mildew?\n2️⃣ How to TREAT Powdery Mildew?",
            "prevent": "🛡️ How to PREVENT Powdery Mildew\n\n• Plant resistant varieties of your crop\n• Ensure good air circulation between plants\n• Avoid excessive nitrogen fertilization\n• Water at base — avoid wetting leaves\n• Apply preventive sulfur spray in susceptible conditions\n• Remove and destroy infected plant material\n• Plant in sunny locations — shade promotes mildew\n• Maintain proper plant spacing for air flow\n\nWould you like to know how to TREAT Powdery Mildew? Type 'treat powdery mildew'",
            "treat": "💊 How to TREAT Powdery Mildew\n\n• Apply potassium bicarbonate or sulfur fungicide immediately\n• Spray every 7-14 days until symptoms disappear\n• Apply neem oil as effective organic treatment\n• Mix 1 tablespoon baking soda + 1 teaspoon soap in 1L water as home remedy\n• Remove severely infected leaves\n• Improve air circulation around plants\n• Apply in evening to avoid burning leaves with sulfur\n• Continue treatment for 2-3 weeks after symptoms clear\n\nRecovery time: 2-3 weeks with consistent treatment"
        },
    }
    
    def find_disease(msg):
        if "apple scab" in msg or "scab" in msg and "apple" in msg:
            return "apple scab"
        elif "apple black rot" in msg or ("black rot" in msg and "apple" in msg):
            return "apple black rot"
        elif "potato early blight" in msg or ("early blight" in msg and "potato" in msg):
            return "potato early blight"
        elif "potato late blight" in msg or ("late blight" in msg and "potato" in msg):
            return "potato late blight"
        elif "tomato early blight" in msg or ("early blight" in msg and "tomato" in msg):
            return "tomato early blight"
        elif "tomato late blight" in msg or ("late blight" in msg and "tomato" in msg):
            return "tomato late blight"
        elif "tomato bacterial spot" in msg or ("bacterial spot" in msg and "tomato" in msg):
            return "tomato bacterial spot"
        elif "grape black rot" in msg or ("black rot" in msg and "grape" in msg):
            return "grape black rot"
        elif "corn common rust" in msg or ("common rust" in msg and "corn" in msg) or ("rust" in msg and "corn" in msg):
            return "corn common rust"
        elif "strawberry leaf scorch" in msg or ("leaf scorch" in msg and "strawberry" in msg):
            return "strawberry leaf scorch"
        elif "tomato leaf mold" in msg or ("leaf mold" in msg and "tomato" in msg):
            return "tomato leaf mold"
        elif "septoria" in msg or ("septoria" in msg and "tomato" in msg):
            return "tomato septoria"
        elif "spider mite" in msg or "spider mites" in msg:
            return "spider mites"
        elif "target spot" in msg or ("target" in msg and "tomato" in msg):
            return "tomato target spot"
        elif "yellow leaf curl" in msg or "tylcv" in msg:
            return "yellow leaf curl"
        elif "mosaic virus" in msg or ("mosaic" in msg and "tomato" in msg):
            return "mosaic virus"
        elif "powdery mildew" in msg or "mildew" in msg:
            return "powdery mildew"
        return None

    disease = find_disease(message)
    
    if any(word in message for word in ["hello", "hi", "hey", "namaste", "helo"]):
        return {"reply": "🌿 Hello! Welcome to PlantDoc AI Assistant!\n\nI can help you with:\n• Information about 38 plant diseases\n• How diseases happen and spread\n• Prevention tips\n• Treatment advice\n\nJust type the name of a disease like:\n• 'potato early blight'\n• 'tomato late blight'\n• 'powdery mildew'\n\nHow can I help you today?"}
    
    elif any(word in message for word in ["prevent", "prevention"]) and disease:
        return {"reply": DISEASE_DB[disease]["prevent"]}
    
    elif any(word in message for word in ["treat", "treatment", "cure", "fix"]) and disease:
        return {"reply": DISEASE_DB[disease]["treat"]}
    
    elif disease:
        return {"reply": DISEASE_DB[disease]["about"]}
    
    elif any(word in message for word in ["prevent", "prevention"]):
        return {"reply": "🛡️ General Plant Disease Prevention Tips\n\n• Rotate crops every 2-3 years\n• Use certified disease-free seeds and plants\n• Water at the base — avoid wetting leaves\n• Ensure good air circulation between plants\n• Apply preventive fungicide during wet weather\n• Remove infected plant material immediately\n• Clean and disinfect tools regularly\n• Monitor plants weekly for early signs of disease\n• Maintain proper nutrition for strong plant immunity\n• Remove weeds that harbor pests and diseases\n\nFor specific disease prevention, type the disease name!"}
    
    elif any(word in message for word in ["treat", "treatment", "cure"]):
        return {"reply": "💊 General Plant Disease Treatment Tips\n\n• Identify the disease correctly before treating\n• Apply appropriate fungicide or bactericide\n• Remove and destroy all infected plant parts\n• Never compost diseased material\n• Maintain treatment schedule consistently\n• Improve growing conditions (air flow, drainage)\n• Ensure plants are well-nourished for recovery\n• Monitor plants closely after treatment\n\nFor specific disease treatment, type the disease name!"}
    
    elif any(word in message for word in ["list", "diseases", "what diseases", "which diseases"]):
        return {"reply": "🌿 PlantDoc can detect these 38 conditions:\n\n🍎 Apple: Apple Scab, Black Rot, Cedar Rust, Healthy\n🫐 Blueberry: Healthy\n🍒 Cherry: Powdery Mildew, Healthy\n🌽 Corn: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy\n🍇 Grape: Black Rot, Esca, Leaf Blight, Healthy\n🍊 Orange: Haunglongbing (Citrus Greening)\n🍑 Peach: Bacterial Spot, Healthy\n🫑 Pepper: Bacterial Spot, Healthy\n🥔 Potato: Early Blight, Late Blight, Healthy\n🫐 Raspberry: Healthy\n🌱 Soybean: Healthy\n🥒 Squash: Powdery Mildew\n🍓 Strawberry: Leaf Scorch, Healthy\n🍅 Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy\n\nType any disease name to learn more!"}
    
    else:
        return {"reply": f"🌿 I'm PlantDoc AI Assistant!\n\nI didn't understand '{request.message}'. Try asking about:\n\n• A specific disease: 'potato early blight'\n• Prevention: 'how to prevent tomato late blight'\n• Treatment: 'how to treat powdery mildew'\n• List all diseases: 'list diseases'\n\nI know about all 38 plant diseases your app can detect!"}