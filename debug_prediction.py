"""
debug_prediction.py — لفهم ليه الرسالة دي اتصنفت غلط
"""
import pickle
import re
import os

# ─── نفس دالة المعالجة من train.py ─────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
    text = re.sub(r'\b\d+\b', ' num ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── تحميل الموديل ────────────────────────────────────────────
MODEL_PATH = "1_model/models/best_ml_model.pkl"
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

# ─── الرسالة المشبوهة ─────────────────────────────────────────
test_message = "Your bank account will be closed. Update info: http://fake-bank.com"

print(f"\n🔍 الرسالة الأصلية:\n   {test_message}")
print(f"\n🔧 بعد المعالجة:\n   {preprocess_text(test_message)}")

# ─── التنبؤ بدون معالجة (غلط) ─────────────────────────────────
pred_raw = pipeline.predict([test_message])[0]
proba_raw = pipeline.predict_proba([test_message])[0][1] if hasattr(pipeline, 'predict_proba') else 0.5
print(f"\n❌ بدون معالجة:")
print(f"   النتيجة: {'🔴 احتيالية' if pred_raw == 1 else '🟢 آمنة'}")
print(f"   الثقة: {proba_raw*100:.1f}%")

# ─── التنبؤ بعد المعالجة (صح) ─────────────────────────────────
processed = preprocess_text(test_message)
pred_proc = pipeline.predict([processed])[0]
proba_proc = pipeline.predict_proba([processed])[0][1] if hasattr(pipeline, 'predict_proba') else 0.5
print(f"\n✅ بعد المعالجة:")
print(f"   النتيجة: {'🔴 احتيالية' if pred_proc == 1 else '🟢 آمنة'}")
print(f"   الثقة: {proba_proc*100:.1f}%")

# ─── تحليل الـ Features ───────────────────────────────────────
print(f"\n📊 تحليل الـ TF-IDF Features:")
vectorizer = pipeline.named_steps['tfidf']
clf = pipeline.named_steps['clf']

vec = vectorizer.transform([processed])
feature_names = vectorizer.get_feature_names_out()
coeffs = clf.coef_[0] if hasattr(clf, 'coef_') else None

if coeffs is not None:
    # أهم الكلمات اللي أثرت في القرار
    indices = vec.nonzero()[1]
    scores = [(feature_names[i], vec[0, i], coeffs[i]) for i in indices]
    scores.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\n🔑 أهم الكلمات المؤثرة:")
    for word, tfidf_val, coef in scores[:10]:
        impact = "➕ احتيالية" if coef > 0 else "➖ آمنة"
        print(f"   {word:15s} | TF-IDF: {tfidf_val:.3f} |_coef: {coef:+.3f} | {impact}")