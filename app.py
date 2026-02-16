# import uvicorn
# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ELU
# from tensorflow.keras import Model

# app = FastAPI()

# # إعداد الملفات الثابتة والقوالب
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# # تعريف بنية النموذج المحدثة بـ ELU
# class MyModel(Model):
#     def __init__(self):
#         super().__init__()
#         self.c1 = Conv2D(32, (3,3), padding="same")
#         self.a1 = ELU()
#         self.p1 = MaxPooling2D((2,2))
        
#         self.c2 = Conv2D(64, (3,3), padding="same")
#         self.b2 = BatchNormalization()
#         self.a2 = ELU()
#         self.p2 = MaxPooling2D((2,2))
        
#         self.c3 = Conv2D(128, (3,3), padding="same")
#         self.a3 = ELU()
#         self.p3 = MaxPooling2D((2,2))

#         self.f = Flatten()

#         self.d1 = Dense(128)
#         self.ad1 = ELU()
        
#         self.d2 = Dense(64)
#         self.ad2 = ELU()
        
#         self.out = Dense(5, activation='softmax')

#     def call(self, inputs):
#         x = self.c1(inputs); x = self.a1(x); x = self.p1(x)
#         x = self.c2(x); x = self.b2(x); x = self.a2(x); x = self.p2(x)
#         x = self.c3(x); x = self.a3(x); x = self.p3(x)
#         x = self.f(x)
#         x = self.d1(x); x = self.ad1(x)
#         x = self.d2(x); x = self.ad2(x)
#         return self.out(x)

# # بناء وتحميل النموذج
# model = MyModel()
# model.build(input_shape=(None, 244, 224, 1))
# # تأكد من اسم ملف النموذج الخاص بك هنا
# try:
#     model.load_weights(r'C:\Users\abdal\OneDrive\Desktop\nti\project\best_knee_model.keras (1).h5')
#     print("Model loaded successfully!")
# except:
#     print("Warning: Could not load weights. Check if the file exists.")

# CLASSES = [
#     "Grade 0: سليم (Healthy)", 
#     "Grade 1: مشكوك فيه (Doubtful)", 
#     "Grade 2: طفيف (Minimal)", 
#     "Grade 3: متوسط (Moderate)", 
#     "Grade 4: حاد (Severe)"
# ]

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
#     # المعالجة: تغيير الحجم إلى (244x224) كما في التدريب
#     img_resized = cv2.resize(img, (224, 244)) 
#     img_input = img_resized.reshape(1, 244, 224, 1).astype('float32') / 255.0
    
#     prediction = model.predict(img_input)
#     class_idx = np.argmax(prediction)
#     confidence = float(np.max(prediction))
    
#     return {
#         "class_id": int(class_idx),
#         "label": CLASSES[class_idx],
#         "confidence": f"{confidence * 100:.2f}%"
#     }




import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ELU
from tensorflow.keras import Model

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class KneeAnalysisModel(Model):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2D(32, (3,3), padding="same")
        self.a1 = ELU()
        self.p1 = MaxPooling2D((2,2))
        
        self.c2 = Conv2D(64, (3,3), padding="same")
        self.b2 = BatchNormalization()
        self.a2 = ELU()
        self.p2 = MaxPooling2D((2,2))
        
        self.c3 = Conv2D(128, (3,3), padding="same")
        self.a3 = ELU()
        self.p3 = MaxPooling2D((2,2))

        self.f = Flatten()
        self.d1 = Dense(128); self.ad1 = ELU()
        self.d2 = Dense(64); self.ad2 = ELU()
        self.out = Dense(5, activation='softmax')

    def call(self, inputs):
        x = self.c1(inputs); x = self.a1(x); x = self.p1(x)
        x = self.c2(x); x = self.b2(x); x = self.a2(x); x = self.p2(x)
        x = self.c3(x); x = self.a3(x); x = self.p3(x)
        x = self.f(x)
        x = self.d1(x); x = self.ad1(x)
        x = self.d2(x); x = self.ad2(x)
        return self.out(x)


model = KneeAnalysisModel()
model.build(input_shape=(None, 224, 224, 1))


try:
    model.load_weights('best_knee_model.keras (1).h5')
    print("✓ AI Diagnostic Engine Online")
except:
    print("! Warning: Weights not found. Running in demo mode.")

CLASSES = [
    {"grade": "Grade 0", "desc": "Healthy - No signs of osteoarthritis."},
    {"grade": "Grade 1", "desc": "Doubtful - Possible joint space narrowing."},
    {"grade": "Grade 2", "desc": "Minimal - Definite osteophytes present."},
    {"grade": "Grade 3", "desc": "Moderate - Multiple osteophytes, narrowing."},
    {"grade": "Grade 4", "desc": "Severe - Large osteophytes, severe narrowing."}
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    
    img_resized = cv2.resize(img, (224, 224)) 
    img_input = img_resized.reshape(1, 224, 224, 1).astype('float32') / 255.0
    
    prediction = model.predict(img_input)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    return {
        "class_id": int(class_idx),
        "label": CLASSES[class_idx]["grade"],
        "description": CLASSES[class_idx]["desc"],
        "confidence": f"{confidence * 100:.1f}%",
        "severity_score": int(class_idx) 
    }