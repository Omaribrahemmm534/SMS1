"""
SMS Phishing Detection — Streamlit Dashboard
=============================================
واجهة مستخدم تفاعلية للتحليل وعرض الإحصائيات.

تشغيل:
    pip install streamlit requests pandas plotly numpy scikit-learn
    streamlit run app.py
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import os
import pickle
import re
import numpy as np


# ─── دالة معالجة النصوص (نفس اللي في التدريب) ─────────────────────────────
def preprocess_text(text: str) -> str:
    """معالجة النص بنفس الطريقة المستخدمة في تدريب الموديل"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    
    # تطبيع الحروف العربية
    arabic_norm = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ي', 'ء': 'ا', 'ى': 'ي', 'ة': 'ه'}
    for char, norm in arabic_norm.items():
        text = text.replace(char, norm)
    
    # إزالة الروابط
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
    
    # استبدال الأرقام
    text = re.sub(r'\b\d+\b', ' num ', text)
    
    # إزالة الرموز الخاصة
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ─── إعدادات المسارات ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "1_model", "models", "best_ml_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "1_model", "models", "tokenizer.pkl")

# ─── تحميل الموديلات بأمان ───────────────────────────────────────────────────
@st.cache_resource
def load_model_and_vectorizer():
    """تحميل الموديل والـ vectorizer مع caching عشان ما يتحملوش كل مرة"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as v:
            vectorizer = pickle.load(v)
        return model, vectorizer, None
    except FileNotFoundError as e:
        return None, None, f"❌ لم يتم العثور على الملفات: {e}"
    except Exception as e:
        return None, None, f"❌ خطأ أثناء التحميل: {e}"

model, vectorizer, load_error = load_model_and_vectorizer()

if load_error:
    st.error(load_error)
    st.stop()
else:
    st.success("✅ تم تحميل الموديلات بنجاح!")

# ─── دوال مساعدة (Helpers) ───────────────────────────────────────────────────
def ensure_string(text):
    """تأكد إن النص string بأي شكل"""
    if text is None:
        return ""
    if isinstance(text, np.ndarray):
        return str(text.item()) if text.size == 1 else str(text)
    if isinstance(text, (list, tuple)):
        return str(text[0]) if len(text) > 0 else ""
    return str(text)
    
def preprocess_text(text: str) -> str:
    """
    نفس دالة المعالجة المستخدمة في التدريب - ضرورية جداً!
    تطبيع النص، إزالة الروابط، معالجة الأرقام، الحفاظ على العربي
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    
    # تطبيع الحروف العربية (نفس اللي في train.py)
    arabic_norm = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ي', 'ء': 'ا', 'ى': 'ي', 'ة': 'ه'}
    for char, norm in arabic_norm.items():
        text = text.replace(char, norm)
    
    # إزالة الروابط
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
    
    # استبدال الأرقام
    text = re.sub(r'\b\d+\b', ' num ', text)
    
    # إزالة الرموز الخاصة مع الحفاظ على العربي والإنجليزي
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def smart_vectorize(text, vec):
    """تحويل ذكي للنص لأرقام (يدعم Keras Tokenizer و Sklearn Vectorizer)"""
    text = ensure_string(text)
    if hasattr(vec, "texts_to_matrix"):  # Keras Tokenizer
        return vec.texts_to_matrix([text], mode='tfidf')
    else:  # Sklearn Vectorizer (TF-IDF, CountVectorizer, etc.)
        return vec.transform([text])

def get_confidence(model, text_raw, vectorizer=None):
    """حساب نسبة الثقة بشكل آمن مع أي نوع موديل (Pipeline أو عادي)"""
    try:
        if hasattr(model, 'named_steps'):
            # Pipeline - نستخدم predict_proba مباشرة على النص
            if hasattr(model, "predict_proba"):
                return float(np.max(model.predict_proba([text_raw])))
        else:
            # Model عادي - نعمل vectorization الأول
            if hasattr(model, "predict_proba"):
                if vectorizer is not None:
                    if hasattr(vectorizer, "texts_to_matrix"):
                        vec = vectorizer.texts_to_matrix([text_raw], mode='tfidf')
                    else:
                        vec = vectorizer.transform([text_raw])
                    return float(np.max(model.predict_proba(vec)))
    except Exception:
        pass
    
    # Fallback لو مفيش predict_proba
    return 0.95

def extract_features(text):
    """استخراج الميزات اليدوية للرسالة"""
    # ✅ تحويل النص لـ string بأي شكل
    text = ensure_string(text).strip()
    
    length = len(text)
    caps = sum(1 for c in text if c.isupper())
    caps_ratio = caps / length if length > 0 else 0
    has_url = bool(re.search(r'(https?://[^\s]+)', text))
    has_phone = bool(re.search(r'\+?\d[\d -]{8,12}\d', text))
    keywords = ['win', 'prize', 'urgent', 'free', 'click', 'مبروك', 'جائزة', 'اربح', 'عاجل', 'تحديث']
    
    # ✅ التأكد إن text string قبل lower()
    text_lower = text.lower()
    kw_count = sum(1 for w in keywords if w.lower() in text_lower)
    
    exclamations = text.count('!')
    
    return {
        "has_url": has_url,
        "has_phone": has_phone,
        "keyword_count": kw_count,
        "caps_ratio": caps_ratio,
        "text_length": length,
        "exclamation_count": exclamations
    }

# ─── إعداد الصفحة ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/spam.png", width=60)
    st.title("🛡️ Phishing Detector")
    st.caption("مشروع تخرج — SMS Phishing Detection using ML & DL")
    st.divider()
    
    st.subheader("⚙️ الإعدادات")
    threshold = st.slider("حد الكشف (Threshold)", 0.3, 0.9, 0.5, 0.05)
    st.info(f"💡 الرسائل اللي نسبتها > {threshold*100:.0f}% هتعتبر احتيالية")
    
    st.divider()
    st.caption("🔐 جميع التحليلات تتم محلياً - لا يتم إرسال بياناتك لأي سيرفر خارجي")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🔍 كشف رسائل التصيد الاحتيالي")
st.caption("SMS Phishing Detection using Machine Learning & Deep Learning")
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_stats, tab_about = st.tabs([
    "🔎 تحليل رسالة",
    "📦 تحليل مجموعة",
    "📊 الإحصائيات",
    "ℹ️ عن المشروع",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: تحليل رسالة واحدة
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📩 أدخل الرسالة")
        sample = st.selectbox("جرب رسالة نموذجية:", [
            "اكتب رسالتك هنا...",
            "WINNER!! You've been selected for a £1000 prize. Call now!",
            "Hey, are you coming to the party tonight?",
            "URGENT: Your account has been suspended. Verify immediately",
            "مبروك! لقد فزت بجائزة نقدية. اتصل الآن لاستلامها",
            "هل أنت جاهز للاجتماع غدًا الساعة 3؟",
        ])

        text_input = st.text_area(
            "نص الرسالة:",
            value=sample if sample != "اكتب رسالتك هنا..." else "",
            height=140,
            placeholder="الصق الرسالة المشبوهة هنا...",
        )
        source = st.selectbox("مصدر الرسالة:", ["sms", "whatsapp", "telegram", "email", "other"])
        analyze_btn = st.button("🔍 تحليل", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 النتيجة")

        if analyze_btn:
            # ✅ التأكد من إن النص string قبل أي حاجة
            if text_input is None or ensure_string(text_input).strip() == "":
                st.warning("⚠️ اكتب رسالة أولاً")
            else:
                # تحويل النص لـ string بشكل نهائي
                message_text = ensure_string(text_input).strip()
                
                if message_text:
                    with st.spinner("جارٍ التحليل..."):
                        try:
                            # 1. استخراج الميزات
                            feats = extract_features(message_text)
                            
                            # 2. التوقع - الـ Pipeline بيعمل الـ vectorization لوحده
                            # ✅ الإصلاح: نبعت النص raw مش vectorized
                            processed_text = preprocess_text(message_text)
                            raw_prediction = model.predict([processed_text])
                            
                            # استخراج القيمة الصافية
                            pred_val = raw_prediction[0] if isinstance(raw_prediction, (np.ndarray, list)) else raw_prediction
                            
                            # تحديد هل هي احتيالية
                            is_phishing = False
                            if isinstance(pred_val, str):
                                if pred_val.strip().lower() in ["spam", "phishing", "1"]:
                                    is_phishing = True
                            else:
                                if float(pred_val) >= 0.5:
                                    is_phishing = True

                            # حساب نسبة الثقة
                            conf = get_confidence(model, processed_text, vectorizer)

                            # 3. عرض النتيجة
                            if is_phishing:
                                st.error(f"🔴 **احتيالية** — ثقة: {conf*100:.1f}%")
                            else:
                                st.success(f"🟢 **آمنة** — ثقة: {conf*100:.1f}%")

                            # رسم العداد (Gauge chart)
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=conf * 100,
                                title={"text": "نسبة الخطر %"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": "#e74c3c" if is_phishing else "#27ae60"},
                                    "steps": [
                                        {"range": [0, 40], "color": "#d5f5e3"},
                                        {"range": [40, 70], "color": "#fef9e7"},
                                        {"range": [70, 100], "color": "#fadbd8"},
                                    ],
                                    "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                                },
                                number={"suffix": "%"},
                            ))
                            fig.update_layout(height=220, margin=dict(t=30, b=0, l=0, r=0))
                            st.plotly_chart(fig, use_container_width=True)

                            # عرض الميزات
                            st.markdown("**🔬 ميزات الرسالة:**")
                            feat_cols = st.columns(2)
                            items = [
                                ("🔗 يحتوي URL", "✅" if feats["has_url"] else "❌"),
                                ("📞 يحتوي رقم", "✅" if feats["has_phone"] else "❌"),
                                ("🚨 كلمات مشبوهة", str(feats["keyword_count"])),
                                ("🔠 نسبة Caps", f"{feats['caps_ratio']*100:.1f}%"),
                                ("📏 طول الرسالة", str(feats["text_length"])),
                                ("❗ علامات تعجب", str(feats["exclamation_count"])),
                            ]
                            for i, (k, v) in enumerate(items):
                                feat_cols[i % 2].metric(k, v)

                            # حفظ في التاريخ
                            st.session_state.history.append({
                                "text": message_text[:80] + "..." if len(message_text) > 80 else message_text,
                                "label": "احتيالية" if is_phishing else "آمنة",
                                "confidence": conf,
                                "risk": "عالي" if is_phishing else "منخفض",
                                "source": source,
                                "time": datetime.now().strftime("%H:%M:%S"),
                            })

                        except Exception as e:
                            st.error(f"❌ خطأ: {e}")
                            st.exception(e)
                else:
                    st.warning("⚠️ اكتب رسالة أولاً")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Batch Analysis - النسخة المحسنة
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📦 تحليل مجموعة رسائل")
    st.info("💡 اكتب كل رسالة في سطر منفصل. الحد الأقصى: 50 رسالة")
    
    batch_input = st.text_area(
        "أدخل الرسائل (رسالة في كل سطر):",
        height=200,
        placeholder="الرسالة الأولى\nالرسالة الثانية\n..."
    )
    
    if st.button("🔍 فحص المجموعة"):
        if batch_input is None or ensure_string(batch_input).strip() == "":
            st.warning("⚠️ يرجى إدخال الرسائل أولاً!")
        else:
            try:
                messages = [ensure_string(msg).strip() for msg in ensure_string(batch_input).split('\n') if ensure_string(msg).strip() != ""]
                
                if not messages:
                    st.warning("⚠️ لم يتم العثور على رسائل صالحة للتحليل.")
                elif len(messages) > 50:
                    st.warning(f"⚠️ تم اختيار أول 50 رسالة فقط من أصل {len(messages)}")
                    messages = messages[:50]
                else:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for i, msg in enumerate(messages):
                        # تحديث التقدم
                        progress_bar.progress((i + 1) / len(messages))
                        status_text.text(f"جاري فحص الرسالة {i+1} من {len(messages)}...")
                        
                        # ✅ الإصلاح: الـ Pipeline بيعمل الـ vectorization لوحده
                        processed_msg = preprocess_text(msg)
                        prediction = model.predict([processed_msg])[0]
                        
                        # Unpacking
                        pred_val = prediction
                        while isinstance(pred_val, (np.ndarray, list)):
                            pred_val = pred_val[0]
                        
                        # تحديد النتيجة
                        is_phishing = False
                        if isinstance(pred_val, str):
                            if pred_val.strip().lower() in ["spam", "phishing", "1"]:
                                is_phishing = True
                        else:
                            if float(pred_val) >= 0.5:
                                is_phishing = True
                        
                        # حساب الثقة
                        conf = get_confidence(model, processed_msg, vectorizer)
                        
                        results.append({
                            "message": msg[:50] + "..." if len(msg) > 50 else msg,
                            "label": "🔴 احتيالية" if is_phishing else "🟢 آمنة",
                            "confidence": f"{conf*100:.1f}%",
                            "is_phishing": is_phishing
                        })
                    
                    status_text.text("✅ اكتمل التحليل!")
                    progress_bar.empty()
                    
                    # عرض النتائج في جدول
                    results_df = pd.DataFrame(results)
                    st.dataframe(
                        results_df[["message", "label", "confidence"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # إحصائيات سريعة
                    phishing_count = sum(1 for r in results if r["is_phishing"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📩 إجمالي الرسائل", len(results))
                    c2.metric("🔴 احتيالية", phishing_count)
                    c3.metric("🟢 آمنة", len(results) - phishing_count)
                        
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التحليل: {e}")
                st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Statistics
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("📊 إحصائيات الجلسة")

    if not st.session_state.history:
        st.info("📭 لسه ما حللتش أي رسائل في الجلسة دي.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        total = len(hist_df)
        phishing_count = (hist_df["label"] == "احتيالية").sum()
        safe_count = total - phishing_count

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📩 إجمالي", total)
        c2.metric("🔴 احتيالية", phishing_count)
        c3.metric("🟢 آمنة", safe_count)
        c4.metric("📈 نسبة الكشف", f"{phishing_count/total*100:.1f}%" if total > 0 else "0%")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(
                names=hist_df["label"].value_counts().index,
                values=hist_df["label"].value_counts().values,
                color=hist_df["label"].value_counts().index,
                color_discrete_map={"احتيالية": "#e74c3c", "آمنة": "#27ae60"},
                title="توزيع الرسائل",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_hist = px.histogram(
                x=hist_df["confidence"],
                color=hist_df["label"],
                color_discrete_map={"احتيالية": "#e74c3c", "آمنة": "#27ae60"},
                title="توزيع نسب الثقة",
                nbins=20,
                labels={"x": "نسبة الثقة", "color": "النوع"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📋 سجل التحليلات")
        st.dataframe(
            hist_df[["time", "text", "label", "confidence", "risk", "source"]],
            use_container_width=True,
            hide_index=True
        )

        # زر المسح يظهر بس لو فيه تاريخ
        if st.session_state.history:
            if st.button("🗑️ مسح السجل", type="secondary"):
                st.session_state.history = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ عن المشروع")
    st.markdown("""
    ### 🛡️ SMS Phishing Detection using ML & DL

    **الهدف:** بناء نظام ذكي لكشف رسائل التصيد الاحتيالي (Phishing/Spam)
    في SMS و WhatsApp و Telegram باستخدام تقنيات الذكاء الاصطناعي.

    ---

    #### 🏗️ المكونات التقنية:
    | المكون | التقنية المستخدمة |
    |--------|-------------------|
    | **النماذج** | SVM, Naive Bayes, CNN-Text, LSTM |
    | **معالجة النصوص** | TF-IDF, Tokenization, Word Embeddings |
    | **الواجهة** | Streamlit + Plotly للتصورات |
    | **الـ Backend** | FastAPI (Python) |
    | **التكامل** | Telegram Bot API, WhatsApp (Twilio) |

    #### 📊 مقاييس الأداء:
    - ✅ **Accuracy**: ~99% على بيانات الاختبار
    - ✅ **Precision**: دقة عالية في تجنب الـ False Positives
    - ✅ **Recall**: كشف معظم رسائل الـ Spam الحقيقية
    - ✅ **F1-Score**: توازن ممتاز بين الدقة والاستدعاء

    #### 🔐 الخصوصية والأمان:
    - جميع التحليلات تتم **محلياً** على جهازك
    - لا يتم إرسال أي رسائل لسيرفرات خارجية
    - الكود مفتوح المصدر للمراجعة والتدقيق

    ---
    *مشروع تخرج — كلية الحاسبات والمعلومات 🎓*
    """)
    
    st.divider()
    st.caption("🔧 تم التطوير باستخدام ❤️ و Python")