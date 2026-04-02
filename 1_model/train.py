"""
SMS Phishing Detection — Model Training (Improved for Arabic)
================================================================
يدرّب عدة نماذج ML و DL ويحفظ أفضلهم مع دعم محسّن للغة العربية.

تشغيل:
    pip install -r requirements.txt
    python train.py
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score, f1_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── TensorFlow / Keras (DL) ───────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, LSTM, Dense, Dropout,
        GlobalMaxPooling1D, Conv1D
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("[!] TensorFlow مش متاح — هيتدرب ML فقط")

# ─── إعدادات المسارات (مصححة) ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# البحث عن ملف البيانات في مسارات متعددة
def find_data_file():
    """البحث عن ملف spam.csv في مسارات مختلفة"""
    possible_paths = [
        os.path.join(SCRIPT_DIR, "spam.csv"),                    # نفس فولدر السكربت
        os.path.join(SCRIPT_DIR, "1_model", "spam.csv"),         # داخل 1_model
        os.path.join(SCRIPT_DIR, "..", "1_model", "spam.csv"),   # خارجي
        os.path.join(SCRIPT_DIR, "data", "spam.csv"),            # داخل data/
        "spam.csv",                                               # المسار الحالي
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

DATA_PATH = find_data_file()
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Config ────────────────────────────────────────────────────────────────
MAX_FEATURES   = 15_000   # حجم المفردات للـ TF-IDF (زادناه لدعم العربية)
MAX_LEN        = 150      # أقصى طول للرسالة (في الـ DL)
EMBEDDING_DIM  = 64       # حجم الـ embedding layer
LSTM_UNITS     = 64
EPOCHS         = 15       # زادنا الـ epochs لتحسين التعلم
BATCH_SIZE     = 32
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# ─── قاموس تطبيع الحروف العربية ───────────────────────────────────────────
ARABIC_NORMALIZATION = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ي', 'ء': 'ا',
    'ى': 'ي', 'ة': 'ه', 'گ': 'ك', 'پ': 'ب', 'چ': 'ج', 'ڤ': 'ف',
    'ۀ': 'ه', 'ك': 'ك', '،': ' ', '؛': ' ', '؟': ' ', '！': ' ',
}

# ─── كلمات مفتاحية للكشف عن التصيد (عربي + إنجليزي) ───────────────────────
PHISHING_KEYWORDS_AR = [
    'مبروك', 'فزت', 'جائزة', 'اربح', 'ربح', 'عاجل', 'مهم', 'تحقق', 'رابط',
    'اضغط', 'هنا', 'عرض', 'خصم', 'مجاني', 'هدية', 'نقدية', 'حسابك', 'معلق',
    'مغلق', 'كلمة', 'سر', 'بنك', 'بطاقة', 'رصيد', 'تحديث', 'تأكيد', 'فوري',
    'محدود', 'فرصة', 'اختر', 'شارك', 'ارسل', 'بياناتك', 'معلومات', 'خاص',
    'حصري', 'مفاجأة', 'تهانينا', 'مبارك', 'استلم', 'جائزتك', 'الآن', 'فوراً'
]

PHISHING_KEYWORDS_EN = [
    'winner', 'prize', 'congratulations', 'urgent', 'important', 'verify',
    'click', 'here', 'offer', 'discount', 'free', 'gift', 'win', 'cash',
    'account', 'suspended', 'closed', 'password', 'bank', 'card', 'balance',
    'reward', 'selected', 'claim', 'limited', 'exclusive', 'surprise',
    'instant', 'immediate', 'act now', 'hurry', 'don\'t miss', 'final'
]

SAFE_KEYWORDS_AR = [
    'مرحباً', 'أهلاً', 'شكراً', 'مع السلامة', 'إلى اللقاء', 'صباح', 'مساء',
    'كيف', 'حالك', 'صحتك', 'العافية', 'اجتماع', 'موعد', 'عمل', 'دراسة',
    'عائلة', 'أصدقاء', 'طعام', 'نوم', 'صباح الخير', 'مساء الخير'
]

SAFE_KEYWORDS_EN = [
    'hello', 'hi', 'thanks', 'thank you', 'goodbye', 'morning', 'evening',
    'how are you', 'meeting', 'appointment', 'work', 'study', 'family',
    'friends', 'food', 'sleep', 'good morning', 'good evening', 'see you'
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. معالجة النصوص (مع دعم عربي محسّن)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_arabic(text: str) -> str:
    """تطبيع الحروف العربية (توحيد الأشكال المختلفة لنفس الحرف)"""
    for char, norm in ARABIC_NORMALIZATION.items():
        text = text.replace(char, norm)
    return text

def preprocess_text(text: str) -> str:
    """
    معالجة شاملة للنص: عربي + إنجليزي
    - تطبيع العربي
    - إزالة الروابط
    - معالجة الأرقام
    - الحفاظ على الكلمات المفتاحية
    """
    if not isinstance(text, str):
        text = str(text)
    
    # حفظ الكلمات المفتاحية قبل المعالجة (لإضافتها كـ features)
    text_lower = text.lower()
    kw_ar_count = sum(1 for kw in PHISHING_KEYWORDS_AR if kw in text_lower)
    kw_en_count = sum(1 for kw in PHISHING_KEYWORDS_EN if kw in text_lower)
    safe_ar_count = sum(1 for kw in SAFE_KEYWORDS_AR if kw in text_lower)
    safe_en_count = sum(1 for kw in SAFE_KEYWORDS_EN if kw in text_lower)
    
    # التطبيع والمعالجة
    text = text.lower()
    text = normalize_arabic(text)
    
    # إزالة الروابط مع الحفاظ على كلمة url كـ feature
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
    
    # استبدال الأرقام بكلمة num (بدلاً من إزالتها تماماً)
    text = re.sub(r'\b\d+\b', ' num ', text)
    
    # إزالة الرموز الخاصة مع الحفاظ على الحروف العربية والإنجليزية والمسافات
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    
    # إضافة عدّاد الكلمات المفتاحية كـ features إضافية
    if kw_ar_count > 0 or kw_en_count > 0:
        text += f" kw_phish_{kw_ar_count + kw_en_count}"
    if safe_ar_count > 0 or safe_en_count > 0:
        text += f" kw_safe_{safe_ar_count + safe_en_count}"
    
    return text


def augment_data(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label', 
                 augmentation_factor: float = 0.3) -> pd.DataFrame:
    """
    زيادة البيانات (Data Augmentation) للرسائل الاحتيالية العربية
    لتحسين أداء الموديل على النصوص العربية
    """
    augmented = []
    
    # تصفية الرسائل الاحتيالية التي تحتوي على نص عربي
    spam_ar = df[(df[label_col] == 1) & (df[text_col].str.contains(r'[\u0600-\u06FF]', regex=True))]
    
    for idx, row in spam_ar.iterrows():
        text = row[text_col]
        label = row[label_col]
        
        # تقنية 1: إضافة علامات تعجب
        augmented.append({'text': text + '!', label: label})
        augmented.append({'text': text + '!!', label: label})
        
        # تقنية 2: إضافة بادئة "عاجل:"
        augmented.append({'text': 'عاجل: ' + text, label: label})
        
        # تقنية 3: تبديل مرادفات بسيطة
        text_swapped = text.replace('جائزة', 'هدية').replace('اربح', 'فزت')
        if text_swapped != text:
            augmented.append({'text': text_swapped, label: label})
    
    if augmented:
        aug_df = pd.DataFrame(augmented)
        print(f"[+] تم زيادة {len(aug_df)} رسالة عربية إضافية")
        return pd.concat([df, aug_df], ignore_index=True)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. تحميل البيانات
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> pd.DataFrame:
    """يحمّل dataset من مسار تم تحديده مسبقاً"""
    
    if DATA_PATH is None:
        print("[!] ⚠️ ملف spam.csv غير موجود!")
        print("[i] 💡 يمكنك تحميله من: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        print("[i] 💡 أو استخدم البيانات التجريبية المؤقتة...\n")
        return _generate_demo_data()
    
    try:
        # محاولة التحميل مع ترميز latin-1 (المستخدم في ملف UCI الأصلي)
        df = pd.read_csv(DATA_PATH, encoding='latin-1', on_bad_lines='skip')
        
        # التعامل مع أسماء الأعمدة المختلفة
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        elif 'Label' in df.columns and 'Message' in df.columns:
            df = df[["Label", "Message"]].rename(columns={"Label": "label", "Message": "text"})
        elif 'label' not in df.columns or 'text' not in df.columns:
            print(f"[!] ⚠️ أعمدة غير متوقعة: {list(df.columns)}")
            return _generate_demo_data()
        
        # تحويل التصنيفات
        df["label"] = df["label"].astype(str).str.lower().map({
            'spam': 1, 'phishing': 1, '1': 1, 'true': 1,
            'ham': 0, 'safe': 0, 'legitimate': 0, '0': 0, 'false': 0
        })
        
        # إزالة الصفوف التي لم يتم تصنيفها
        df = df.dropna(subset=['label', 'text'])
        df['label'] = df['label'].astype(int)
        
        print(f"[+] تم تحميل {len(df)} رسالة من: {os.path.basename(DATA_PATH)}")
        return df
        
    except Exception as e:
        print(f"[!] خطأ أثناء تحميل البيانات: {e}")
        print("[i] جاري استخدام البيانات التجريبية كبديل...")
        return _generate_demo_data()


def _generate_demo_data(n: int = 3000) -> pd.DataFrame:
    """يولّد بيانات تجريبية غنية بالعربية والإنجليزية لاختبار الكود"""
    
    spam_templates_ar = [
        "مبروك! لقد فزت بجائزة نقدية بقيمة 1000 دولار. اتصل الآن على الرقم 123456789",
        "عاجل: حسابك البنكي سيتم إغلاقه خلال 24 ساعة. تحقق من بياناتك فوراً على الرابط",
        "تهانينا! ربحت آيفون 15 برو. أرسل بياناتك الشخصية لاستلام الهدية الآن",
        "عرض حصري لك فقط! خصم 90% على جميع المنتجات. اضغط هنا قبل نفاذ الكمية",
        "تحذير أمني: تم اكتشاف نشاط مشبوه في حسابك. قم بتغيير كلمة السر فوراً",
        "مبارك! اسمك ظهر في قائمة الفائزين. اتصل برقم الخدمة لاستلام جائزتك",
        "فرصة محدودة! اربح رحلة مجانية لدبي. سجل بياناتك الآن للمشاركة",
        "عاجل جداً: رصيدك سينتهي خلال ساعة. اشحن الآن لتجنب الإيقاف",
        "تهانينا! تم اختيارك للحصول على بطاقة هدايا بقيمة 500 ريال. اضغط للتحقق",
        "تحذير: حسابك في واتساب سيتم حظره. تحقق من هويتك الآن على الرابط",
    ]
    
    spam_templates_en = [
        "WINNER!! You have been selected to receive a £1000 prize. Call now!",
        "FREE entry in 2 a wkly comp to win FA Cup. Text FA to 87121",
        "Congratulations! You've won a free iPhone. Click here to claim",
        "URGENT: Your account has been compromised. Verify immediately",
        "You have won $5000 cash prize. Send your details to collect",
        "Limited time offer! Get 90% off. Reply YES to claim your discount",
        "Your bank account will be suspended. Update your info now",
        "Click the link to verify your identity or your account will be closed",
        "Free ringtones! Reply WIN to 8007 to get yours now",
        "You are selected for a cash reward. Call 0800 to claim now",
    ]
    
    ham_templates_ar = [
        "مرحباً، كيف حالك اليوم؟ أتمنى أن تكون بخير",
        "هل أنت جاهز للاجتماع غدًا الساعة 3pp؟ لا تنسى إحضار التقرير",
        "شكراً لك على مساعدتك أمس، كنت رائعاً جداً",
        "صباح الخير! أتمنى لك يوماً سعيداً وموفقاً",
        "تذكر أن تحضر الحليب عندما تعود من العمل",
        "العائلة كلها بخير، ننتظرك على الغد",
        "موعد الدكتور يوم الثلاثاء الساعة 10 صباحاً",
        "شكراً على الدعوة، سأحاول الحضور إن شاء الله",
        "الامتحانات قربت، نتمنى لك التوفيق والنجاح",
        "تصبح على خير، نراكم غداً إن شاء الله",
    ]
    
    ham_templates_en = [
        "Hey, are you coming to the party tonight?",
        "Can you pick up some milk on the way home?",
        "Meeting at 3pm tomorrow, don't forget!",
        "Happy birthday! Hope you have a great day",
        "Just finished the report, sending it now",
        "Are you free this weekend for a coffee?",
        "I'll be there in 10 minutes",
        "Thanks for your help yesterday, really appreciated",
        "The match starts at 8, want to watch together?",
        "Call me when you get a chance",
    ]
    
    rng = np.random.default_rng(RANDOM_STATE)
    
    # توليد رسائل احتيالية (40% من البيانات)
    spam_count = int(n * 0.4)
    spam_ar = [rng.choice(spam_templates_ar) for _ in range(spam_count // 2)]
    spam_en = [rng.choice(spam_templates_en) for _ in range(spam_count - spam_count // 2)]
    
    # توليد رسائل آمنة (60% من البيانات)
    ham_count = n - spam_count
    ham_ar = [rng.choice(ham_templates_ar) for _ in range(ham_count // 2)]
    ham_en = [rng.choice(ham_templates_en) for _ in range(ham_count - ham_count // 2)]
    
    # تجميع البيانات
    texts = spam_ar + spam_en + ham_ar + ham_en
    labels = [1] * len(spam_ar + spam_en) + [0] * len(ham_ar + ham_en)
    
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. بناء وتدريب نماذج ML
# ══════════════════════════════════════════════════════════════════════════════

def build_ml_pipelines() -> dict:
    """يبني pipelines جاهزة: TF-IDF + classifier مع إعدادات محسّنة للعربية"""
    
    # إعدادات TF-IDF المحسّنة للعربية
    tfidf_config = {
        'max_features': MAX_FEATURES,
        'ngram_range': (1, 3),        # uni + bi + tri-grams لفهم السياق العربي
        'sublinear_tf': True,          # تقليل تأثير الكلمات المتكررة
        'min_df': 2,                   # تجاهل الكلمات النادرة جداً
        'max_df': 0.85,                # تجاهل الكلمات الشائعة جداً
        'strip_accents': 'unicode',    # دعم أفضل للأحرف العربية
    }
    
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf", LinearSVC(C=10, max_iter=3000, class_weight='balanced')),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf", LogisticRegression(C=10, max_iter=2000, class_weight='balanced')),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=15, 
                                          class_weight='balanced', random_state=RANDOM_STATE)),
        ]),
    }


def train_ml_models(X_train, X_test, y_train, y_test) -> tuple:
    """يدرّب ويقيّم كل نماذج ML ويعيد أفضل موديل"""
    pipelines = build_ml_pipelines()
    results = {}

    print("\n" + "="*60)
    print("  📊 نتائج نماذج Machine Learning")
    print("="*60)

    best_model = None
    best_f1 = 0.0

    for name, pipeline in pipelines.items():
        print(f"\n[🔄 تدريب: {name}]")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # حساب المقاييس
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        f1 = report["1"]["f1-score"]
        
        # حساب AUC لو الموديل يدعم predict_proba
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except AttributeError:
            auc = None

        results[name] = {
            "accuracy": acc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": f1,
            "auc": auc,
            "model": pipeline,
        }

        print(f"  ✅ Accuracy  : {acc:.4f}")
        print(f"  ✅ Precision : {report['1']['precision']:.4f}")
        print(f"  ✅ Recall    : {report['1']['recall']:.4f}")
        print(f"  ✅ F1-Score  : {f1:.4f}")
        if auc:
            print(f"  ✅ AUC-ROC   : {auc:.4f}")

        # تحديث أفضل موديل
        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, pipeline)

    print(f"\n[🏆] أفضل نموذج ML: {best_model[0]} (F1={best_f1:.4f})")
    return results, best_model


# ══════════════════════════════════════════════════════════════════════════════
# 4. بناء وتدريب نماذج DL (لو TensorFlow متاح)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dl_data(X_train, X_test):
    """يحوّل النصوص لـ sequences للـ LSTM/CNN مع دعم عربي"""
    tokenizer = Tokenizer(
        num_words=MAX_FEATURES, 
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'  # الحفاظ على الأحرف العربية
    )
    tokenizer.fit_on_texts(X_train)

    X_tr = pad_sequences(
        tokenizer.texts_to_sequences(X_train), 
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )
    X_te = pad_sequences(
        tokenizer.texts_to_sequences(X_test), 
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    return X_tr, X_te, tokenizer


def build_lstm_model(vocab_size: int) -> "tf.keras.Model":
    """نموذج LSTM محسّن للتعامل مع النصوص العربية"""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2),
        GlobalMaxPooling1D(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy", 
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


def build_cnn_model(vocab_size: int) -> "tf.keras.Model":
    """نموذج CNN-Text أسرع وأخف من LSTM"""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation="relu", padding='same'),
        GlobalMaxPooling1D(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


def train_dl_models(X_train_seq, X_test_seq, y_train, y_test, vocab_size: int) -> dict:
    """يدرّب نماذج Deep Learning"""
    dl_models = {
        "LSTM": build_lstm_model(vocab_size),
        "CNN-Text": build_cnn_model(vocab_size),
    }
    results = {}
    early_stop = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')

    print("\n" + "="*60)
    print("  🧠 نتائج نماذج Deep Learning")
    print("="*60)

    for name, model in dl_models.items():
        print(f"\n[🔄 تدريب: {name}]")
        history = model.fit(
            X_train_seq, y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )
        
        # التقييم
        y_prob = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "accuracy": acc,
            "f1": report["1"]["f1-score"],
            "auc": auc,
            "model": model,
            "history": history,
        }

        print(f"  ✅ Accuracy : {acc:.4f}  |  F1: {report['1']['f1-score']:.4f}  |  AUC: {auc:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. حفظ النماذج
# ══════════════════════════════════════════════════════════════════════════════

def save_models(best_ml, tokenizer=None, best_dl_model=None, best_dl_name=None):
    """يحفظ الـ pipeline الأفضل والـ tokenizer"""
    name, pipeline = best_ml

    # حفظ ML pipeline (يشمل الـ Vectorizer + Classifier)
    ml_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    with open(ml_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[💾] تم حفظ نموذج ML في: {ml_path}  ({name})")

    # حفظ Tokenizer (مهم للـ DL وللـ app)
    if tokenizer:
        tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
        with open(tok_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"[💾] تم حفظ Tokenizer في: {tok_path}")

    # حفظ DL model
    if best_dl_model:
        dl_path = os.path.join(MODELS_DIR, f"best_dl_model_{best_dl_name}.keras")
        best_dl_model.save(dl_path)
        print(f"[💾] تم حفظ نموذج DL في: {dl_path}  ({best_dl_name})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════╗")
    print("║  SMS Phishing Detection — Training (Arabic+)  ║")
    print("╚════════════════════════════════════════════════╝\n")

    # --- التحقق من ملف البيانات ---
    if DATA_PATH:
        print(f"[✓] ملف البيانات: {os.path.basename(DATA_PATH)}")
    else:
        print("[⚠] لم يتم العثور على ملف بيانات — باستخدام البيانات التجريبية")

    # --- تحميل البيانات ---
    df = load_dataset()
    
    # --- زيادة البيانات (اختياري) ---
    augment = input("\n[؟] هل تريد زيادة البيانات العربية؟ (y/n): ").strip().lower()
    if augment == 'y':
        df = augment_data(df)

    # --- معالجة النصوص ---
    print("\n[⚙️] جاري معالجة النصوص...")
    df["text"] = df["text"].apply(preprocess_text)

    # --- عرض الإحصائيات ---
    print(f"\n[📊] توزيع البيانات بعد المعالجة:")
    print(df["label"].value_counts().rename({0: "🟢 Ham (Safe)", 1: "🔴 Spam (Phishing)"}).to_string())

    # --- تقسيم البيانات ---
    X = df["text"].values
    y = df["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n[📈] بيانات التدريب: {len(X_train)} رسالة")
    print(f"[📉] بيانات الاختبار: {len(X_test)} رسالة")

    # --- تدريب نماذج ML ---
    ml_results, best_ml = train_ml_models(X_train, X_test, y_train, y_test)

    # --- تدريب نماذج DL (لو متاح) ---
    tokenizer = None
    best_dl_model, best_dl_name = None, None

    if DL_AVAILABLE:
        print(f"\n[🧠] جاري تجهيز البيانات لـ Deep Learning...")
        X_tr_seq, X_te_seq, tokenizer = prepare_dl_data(X_train, X_test)
        dl_results = train_dl_models(X_tr_seq, X_te_seq, y_train, y_test, vocab_size=MAX_FEATURES)

        # اختيار أفضل نموذج DL
        best_dl_name = max(dl_results, key=lambda k: dl_results[k]["f1"])
        best_dl_model = dl_results[best_dl_name]["model"]

        # --- مقارنة النتائج ---
        print("\n" + "="*60)
        print("  🏆 المقارنة النهائية بين جميع النماذج")
        print("="*60)
        
        all_results = {**{k: v for k, v in ml_results.items()},
                       **{k: v for k, v in dl_results.items()}}
        
        comparison = pd.DataFrame([
            {
              "Model": k, 
              "Accuracy": f"{v['accuracy']:.4f}", 
              "Precision": f"{v.get('precision', 'N/A'):.4f}" if isinstance(v.get('precision'), (int, float)) else "N/A", 
              "Recall": f"{v.get('recall', 'N/A'):.4f}" if isinstance(v.get('recall'), (int, float)) else "N/A",
              "F1": f"{v['f1']:.4f}",
              "AUC": f"{v.get('auc', 'N/A'):.4f}" if isinstance(v.get('auc'), (int, float)) else "N/A"
           }
            for k, v in all_results.items()
        ]).sort_values("F1", ascending=False)
        
        print(comparison.to_string(index=False))

    # --- حفظ النماذج ---
    print(f"\n[💾] جاري حفظ النماذج في: {MODELS_DIR}")
    save_models(best_ml, tokenizer, best_dl_model, best_dl_name)

    # --- اختبار سريع ---
    print(f"\n[🧪] اختبار سريع على رسائل عربية:")
    test_msgs = [
        "مبروك! لقد فزت بجائزة نقدية. اتصل الآن",
        "هل أنت جاهز للاجتماع غدًا؟",
        "عاجل: حسابك البنكي سيتم إغلاقه",
        "شكراً لك على المساعدة",
    ]
    
    _, best_pipeline = best_ml
    for msg in test_msgs:
        processed = preprocess_text(msg)
        pred = best_pipeline.predict([processed])[0]
        proba = best_pipeline.predict_proba([processed])[0][1] if hasattr(best_pipeline, 'predict_proba') else 0.5
        label = "🔴 احتيالية" if pred == 1 else "🟢 آمنة"
        print(f"   {label} ({proba*100:.1f}%): {msg[:40]}...")

    print(f"\n[✅] انتهى التدريب بنجاح! 🎉")
    print(f"[i] لتشغيل التطبيق: streamlit run app.py")


if __name__ == "__main__":
    main()