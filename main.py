"""
SMS Phishing Detection — FastAPI Backend
==========================================
يستقبل رسالة نصية ويرجع هل هي phishing أم لا.

تشغيل:
    pip install fastapi uvicorn scikit-learn numpy pandas
    uvicorn main:app --reload --port 8000

API Docs (Swagger):
    http://localhost:8000/docs
"""

import os
import re
import pickle
import logging
import numpy as np
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator
import uvicorn

# ─── إعداد الـ Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── إعداد التطبيق ───────────────────────────────────────────────────────────
app = FastAPI(
    title="SMS Phishing Detector API",
    description="كشف رسائل التصيد الاحتيالي باستخدام ML/DL",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ في الإنتاج: استبدل بـ ["https://yourdomain.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── نماذج Pydantic ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      example="WINNER!! You've been selected for a £1000 prize!")
    source: Optional[str] = Field(None, example="telegram", description="مصدر الرسالة: sms/whatsapp/telegram/email")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("النص لا يمكن أن يكون فارغاً")
        return v

class PredictResponse(BaseModel):
    label: str  # "phishing" أو "safe"
    confidence: float  # 0.0 → 1.0
    is_phishing: bool
    risk_level: str  # "high" / "medium" / "low"
    features: dict  # ميزات مساعدة للشرح
    model_used: str  # نوع الموديل المستخدم
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_type: str
    version: str
    models_loaded: bool

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)
    
    @validator('texts')
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError("قائمة الرسائل لا يمكن أن تكون فارغة")
        return [t for t in v if t.strip()]  # إزالة النصوص الفاضية

# ─── تحميل النموذج ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "1_model", "models")
ML_MODEL = None
DL_MODEL = None
TOKENIZER = None
MODELS_LOADED = False

def load_models():
    """تحميل الموديلات مع معالجة الأخطاء"""
    global ML_MODEL, DL_MODEL, TOKENIZER, MODELS_LOADED
    
    try:
        # تحميل موديل الـ ML
        ml_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
        if os.path.exists(ml_path):
            with open(ml_path, "rb") as f:
                ML_MODEL = pickle.load(f)
            logger.info(f"✓ ML model loaded from {ml_path}")
        else:
            logger.warning(f"✗ ML model not found at {ml_path}")

        # تحميل الـ Tokenizer
        tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
        if os.path.exists(tok_path):
            with open(tok_path, "rb") as f:
                TOKENIZER = pickle.load(f)
            logger.info(f"✓ Tokenizer loaded from {tok_path}")

        # تحميل موديل الـ DL (اختياري)
        try:
            import tensorflow as tf
            import glob
            dl_files = glob.glob(os.path.join(MODELS_DIR, "best_dl_model_*.keras"))
            if dl_files:
                DL_MODEL = tf.keras.models.load_model(dl_files[0])
                logger.info(f"✓ DL model loaded: {dl_files[0]}")
        except ImportError:
            logger.info("⊙ TensorFlow not installed - DL model skipped")
        except Exception as e:
            logger.warning(f"⊙ Failed to load DL model: {e}")

        MODELS_LOADED = ML_MODEL is not None or DL_MODEL is not None
        logger.info(f"✓ Models initialization complete. Loaded: {MODELS_LOADED}")
        
    except Exception as e:
        logger.error(f"✗ Critical error loading models: {e}", exc_info=True)

# تحميل الموديلات عند بدء التشغيل
load_models()

# ─── ثوابت ودوال مساعدة ─────────────────────────────────────────────────────
MAX_LEN = 150

PHISHING_KEYWORDS = [
    "winner", "prize", "congratulations", "free", "urgent", "claim",
    "verify", "account", "suspended", "click", "limited", "offer",
    "reward", "selected", "cash", "bank", "password", "confirm",
    "مبروك", "فاز", "جائزة", "عاجل", "حسابك", "تحقق", "خصم", "اضغط",
]

URL_PATTERN = re.compile(r"http[s]?://\S+|www\.\S+", re.I)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s\-]{7,}\d)\b")
CAPS_RATIO_THR = 0.4

def preprocess(text: str) -> str:
    """معالجة النص بنفس الطريقة المستخدمة في التدريب"""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"\b\d+\b", " num ", t)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)  # الحفاظ على الحروف العربية
    return re.sub(r"\s+", " ", t).strip()

def extract_features(text: str) -> dict:
    """يستخرج ميزات يدوية لتعزيز الشفافية (Explainable AI)"""
    lower = text.lower()
    return {
        "has_url": bool(URL_PATTERN.search(text)),
        "has_phone": bool(PHONE_PATTERN.search(text)),
        "keyword_count": sum(1 for k in PHISHING_KEYWORDS if k in lower),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "text_length": len(text),
        "exclamation_count": text.count("!"),
        "has_numbers": bool(re.search(r"\d+", text)),
    }

def rule_based_score(features: dict) -> float:
    """سكور بسيط يُستخدم لو النماذج مش متاحة (Fallback)"""
    score = 0.0
    if features["has_url"]: score += 0.3
    if features["keyword_count"] > 2: score += 0.25
    if features["caps_ratio"] > CAPS_RATIO_THR: score += 0.2
    if features["exclamation_count"] > 1: score += 0.15
    if features["has_phone"]: score += 0.1
    return min(score, 0.99)

def get_risk_level(confidence: float, is_phishing: bool) -> str:
    """تحديد مستوى الخطر بناءً على الثقة"""
    if not is_phishing:
        return "low"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"

def predict_sync(text: str) -> dict:
    """
    الدالة الأساسية للتنبؤ - تعمل بشكل متزامن
    """
    features = extract_features(text)
    clean = preprocess(text)
    result = {}

    # 1) محاولة استخدام الـ DL Model أولاً (لو متاح)
    if DL_MODEL and TOKENIZER:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = TOKENIZER.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            conf = float(DL_MODEL.predict(padded, verbose=0)[0][0])
            is_p = conf >= 0.5
            
            result = {
                "label": "phishing" if is_p else "safe",
                "confidence": round(conf if is_p else 1 - conf, 4),
                "is_phishing": is_p,
                "risk_level": get_risk_level(conf, is_p),
                "features": features,
                "model_used": "DL (LSTM/CNN)",
            }
            logger.debug(f"✓ DL prediction: {result['label']} (conf: {result['confidence']})")
            return result
        except Exception as e:
            logger.warning(f"⊙ DL prediction failed: {e}")
            # الاستمرار للـ ML model

    # 2) محاولة استخدام الـ ML Model
    if ML_MODEL:
        try:
            # ✅ الإصلاح: التحقق من وجود predict_proba قبل الاستخدام
            if hasattr(ML_MODEL, "predict_proba"):
                prob = ML_MODEL.predict_proba([clean])[0][1]
            else:
                # Fallback لو الموديل مفيهوش predict_proba (زي SVM بدون probability=True)
                pred = ML_MODEL.predict([clean])[0]
                prob = float(pred) if pred in [0, 1] else 0.5
                logger.debug(f"⊙ Using predict() fallback, prob set to: {prob}")
            
            is_p = prob >= 0.5
            result = {
                "label": "phishing" if is_p else "safe",
                "confidence": round(prob if is_p else 1 - prob, 4),
                "is_phishing": is_p,
                "risk_level": get_risk_level(prob, is_p),
                "features": features,
                "model_used": "ML (TF-IDF + Classifier)",
            }
            logger.debug(f"✓ ML prediction: {result['label']} (conf: {result['confidence']})")
            return result
        except Exception as e:
            logger.error(f"✗ ML prediction failed: {e}", exc_info=True)

    # 3) Fallback: Rule-based system
    logger.info("⊙ Using rule-based fallback")
    score = rule_based_score(features)
    is_p = score >= 0.5
    return {
        "label": "phishing" if is_p else "safe",
        "confidence": round(score if is_p else 1 - score, 4),
        "is_phishing": is_p,
        "risk_level": get_risk_level(score, is_p),
        "features": features,
        "model_used": "Rule-based (fallback)",
    }

async def predict_async(text: str) -> dict:
    """Wrapper async للتنبؤ عشان يدعم FastAPI بشكل أفضل"""
    return await run_in_threadpool(predict_sync, text)

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """Endpoint رئيسي للتحقق من تشغيل الـ API"""
    return {
        "message": "SMS Phishing Detector API 🛡️",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """التحقق من صحة النظام وحالة الموديلات"""
    model_type = "DL" if DL_MODEL else ("ML" if ML_MODEL else "rule-based")
    return HealthResponse(
        status="ok" if MODELS_LOADED else "degraded",
        model_type=model_type,
        version="1.0.0",
        models_loaded=MODELS_LOADED,
    )

@app.post("/predict", response_model=PredictResponse, tags=["Detection"])
async def predict_endpoint(req: PredictRequest, request: Request):
    """
    يحلل رسالة نصية ويرجع النتيجة.
    
    - **label**: "phishing" أو "safe"
    - **confidence**: نسبة الثقة (0-1)
    - **risk_level**: high / medium / low
    - **features**: ميزات الرسالة للشفافية
    """
    logger.info(f"🔍 New prediction request from {request.client.host}")
    
    if not req.text.strip():
        logger.warning("✗ Empty text in request")
        raise HTTPException(status_code=400, detail="النص لا يمكن أن يكون فارغاً!")

    try:
        result = await predict_async(req.text)
        
        response = PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            is_phishing=result["is_phishing"],
            risk_level=result["risk_level"],
            features=result["features"],
            model_used=result["model_used"],
            timestamp=datetime.utcnow().isoformat(),
        )
        logger.info(f"✓ Prediction complete: {result['label']} (conf: {result['confidence']})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ داخلي في المعالجة: {str(e)}")

@app.post("/batch-predict", response_model=List[PredictResponse], tags=["Detection"])
async def batch_predict_endpoint(req: BatchPredictRequest, request: Request):
    """
    يحلل مجموعة رسائل في طلب واحد (الحد الأقصى: 100 رسالة).
    
    مفيد للتحليلات المجمعة أو الـ Bulk Processing.
    """
    logger.info(f"📦 Batch prediction request: {len(req.texts)} messages from {request.client.host}")
    
    if len(req.texts) > 100:
        logger.warning(f"✗ Batch too large: {len(req.texts)} > 100")
        raise HTTPException(status_code=400, detail="الحد الأقصى 100 رسالة في كل طلب")
    
    try:
        results = []
        for i, text in enumerate(req.texts):
            if not text.strip():
                continue  # تخطي النصوص الفاضية
            result = await predict_async(text)
            results.append(PredictResponse(
                label=result["label"],
                confidence=result["confidence"],
                is_phishing=result["is_phishing"],
                risk_level=result["risk_level"],
                features=result["features"],
                model_used=result["model_used"],
                timestamp=datetime.utcnow().isoformat(),
            ))
        
        logger.info(f"✓ Batch complete: {len(results)} predictions")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة المجمعة: {str(e)}")

# ─── تشغيل التطبيق ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # ❌ مهم: False في الإنتاج
        log_level="info",
    )