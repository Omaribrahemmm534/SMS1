"""
SMS Phishing Detection — Telegram Bot
======================================
بوت تيليجرام يحلل أي رسالة بيبعتها المستخدم.
متوافق مع Railway و Streamlit Cloud.

المتطلبات:
    pip install python-telegram-bot requests

إعداد:
    1. ابعت رسالة لـ @BotFather على تيليجرام وعمل بوت جديد
    2. خد الـ Token وحطه في متغير البيئة TELEGRAM_TOKEN
    3. ارفع الـ Backend على Railway أو شغّله محلياً
    4. شغّل: python telegram_bot.py
"""

import os
import time
import logging
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

# ─── إعداد الـ Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ─── الإعدادات (من بيئة التشغيل) ─────────────────────────────────────────────
# ⚠️ ملاحظة: التوكن هنا للتجربة فقط — في الإنتاج استخدم متغير البيئة فقط!
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8668299543:AAEwlM-iGq_xD0cZSqnHMEtlxnfCfWiKWzQ")

# ✅ مهم جداً: API_URL يكون الرابط الأساسي بدون /predict
# Railway: https://your-app.railway.app
# Local: http://localhost:8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = "/predict"  # endpoint ثابت
API_URL = f"{API_BASE_URL.rstrip('/')}{API_ENDPOINT}"

# إعدادات الـ Retry
MAX_RETRIES = 3
RETRY_DELAY = 2  # ثواني

RISK_EMOJI = {
    "high":   "🔴",
    "medium": "🟡", 
    "low":    "🟢",
}

# ─── دوال مساعدة ────────────────────────────────────────────────────────────

def check_api_health(base_url: str, timeout: int = 5) -> bool:
    """التحقق من إن الـ API شغّال"""
    try:
        health_url = f"{base_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False

def call_api_with_retry(text: str, source: str = "telegram") -> dict:
    """إرسال الطلب للـ API مع retry في حال الفشل"""
    payload = {"text": text, "source": source}
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"API Request (attempt {attempt+1}): {text[:50]}...")
            
            response = requests.post(
                API_URL,
                json=payload,
                timeout=15,  # زيادة الـ timeout للرسائل الطويلة
                headers={"User-Agent": "SMS-Phishing-Bot/1.0"}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"API Response: {result.get('label')} (conf: {result.get('confidence')})")
            return result
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                return {"error": "تعذر الاتصال بالخادم. تأكد من تشغيل الـ Backend."}
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                return {"error": "انتهت مهلة الطلب. يرجى المحاولة لاحقاً."}
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e} - Response: {response.text if 'response' in locals() else 'N/A'}")
            return {"error": f"خطأ في الخادم: {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"error": f"خطأ غير متوقع: {str(e)}"}
        
        # انتظار قبل المحاولة التالية
        time.sleep(RETRY_DELAY)
    
    return {"error": "فشل الاتصال بعد عدة محاولات"}

# ─── Handlers ───────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رسالة الترحيب."""
    # التحقق من اتصال الـ API
    api_status = "✅ متصل" if check_api_health(API_BASE_URL) else "❌ غير متصل"
    
    welcome = (
        f"👋 *مرحباً بك في بوت كشف التصيد الاحتيالي* 🛡️\n\n"
        f"📩 *طريقة الاستخدام:*\n"
        f"أرسل لي أي رسالة مشبوهة وسأخبرك فوراً إذا كانت:\n"
        f"  🟢 *آمنة* — أو — 🔴 *احتيالية (Phishing)*\n\n"
        f"🔧 *حالة النظام:* {api_status}\n\n"
        f"📋 *الأوامر المتاحة:*\n"
        f"/start — رسالة الترحيب هذه\n"
        f"/help  — شرح مفصل للاستخدام\n"
        f"/stats — إحصائيات تحليلاتك\n"
        f"/ping  — فحص اتصال البوت بالخادم\n\n"
        f"⚠️ *تنبيه:* النتائج إرشادية ولا تغني عن الحذر الشخصي."
    )
    
    await update.message.reply_text(welcome, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """شرح طريقة الاستخدام."""
    help_text = (
        "📖 *كيفية استخدام البوت:*\n\n"
        "1️⃣ *انسخ* الرسالة المشبوهة من أي تطبيق (SMS، WhatsApp، إلخ)\n"
        "2️⃣ *ألصقها* هنا في محادثة البوت\n"
        "3️⃣ *انتظر* ثوانٍ قليلة...\n"
        "4️⃣ *استلم* التقرير الفوري!\n\n"
        "📊 *مكونات التقرير:*\n"
        "• التصنيف: آمنة ✅ أو احتيالية 🔴\n"
        "• نسبة الثقة: %\n"
        "• مستوى الخطر: منخفض/متوسط/عالي\n"
        "• الميزات المكتشفة: روابط، أرقام، كلمات مشبوهة...\n\n"
        "💡 *نصائح للأمان:*\n"
        "• لا تضغط على روابط في رسائل مشبوهة\n"
        "• لا ترسل بياناتك الشخصية لأي جهة غير موثوقة\n"
        "• تحقق دائماً من عنوان الرابط قبل النقر عليه\n\n"
        "⚠️ *إخلاء مسؤولية:* البوت تجريبي والنتائج قد لا تكون دقيقة 100%."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """فحص اتصال البوت بالخادم."""
    status_msg = await update.message.reply_text("🔄 جاري فحص الاتصال...")
    
    if check_api_health(API_BASE_URL):
        try:
            response = requests.get(f"{API_BASE_URL.rstrip('/')}/health", timeout=5)
            data = response.json()
            reply = (
                f"✅ *البوت متصل بالخادم بنجاح!*\n\n"
                f"📊 *معلومات الخادم:*\n"
                f"• الحالة: {data.get('status', 'unknown')}\n"
                f"• نوع الموديل: {data.get('model_type', 'unknown')}\n"
                f"• الإصدار: {data.get('version', 'unknown')}\n"
                f"• URL: `{API_BASE_URL}`"
            )
        except Exception as e:
            reply = f"✅ متصل لكن حدث خطأ في قراءة البيانات: {e}"
    else:
        reply = (
            f"❌ *لا يمكن الاتصال بالخادم*\n\n"
            f"🔧 *الحلول المقترحة:*\n"
            f"• تأكد من تشغيل الـ Backend على المنفذ الصحيح\n"
            f"• تحقق من قيمة API_BASE_URL في المتغيرات البيئية\n"
            f"• جرب الأمر: `/start` للتحقق من الحالة العامة"
        )
    
    await status_msg.edit_text(reply, parse_mode="Markdown")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إحصائيات المستخدم في الجلسة الحالية."""
    data = context.user_data
    total = data.get("total", 0)
    phish = data.get("phishing", 0)
    safe = data.get("safe", 0)
    
    if total == 0:
        await update.message.reply_text("📭 لم تحلل أي رسائل بعد في هذه الجلسة.")
        return
    
    phishing_rate = (phish / total * 100) if total > 0 else 0
    
    stats_text = (
        f"📊 *إحصائيات تحليلاتك:*\n"
        f"━━━━━━━━━━\n"
        f"📨 إجمالي الرسائل: *{total}*\n"
        f"🔴 احتيالية: *{phish}*\n"
        f"🟢 آمنة: *{safe}*\n"
        f"📈 نسبة التصيد: *{phishing_rate:.1f}%*\n"
        f"━━━━━━━━━━\n"
        f"💡 *ملاحظة:* الإحصائيات تحفظ في الجلسة الحالية فقط."
    )
    await update.message.reply_text(stats_text, parse_mode="Markdown")

async def analyze_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يحلل الرسالة المرسلة من المستخدم."""
    text = update.message.text
    
    if not text or not text.strip():
        await update.message.reply_text("⚠️ يرجى إرسال نص للتحليل.")
        return
    
    # عرض رسالة "جاري التحليل..."
    thinking = await update.message.reply_text("🔍 جاري تحليل الرسالة...")
    
    # استدعاء الـ API
    result = call_api_with_retry(text, source="telegram")
    
    # معالجة الأخطاء
    if "error" in result:
        error_msg = (
            f"❌ *حدث خطأ أثناء التحليل*\n\n"
            f"{result['error']}\n\n"
            f"💡 *جرب:*\n"
            f"• الأمر `/ping` للتحقق من الاتصال\n"
            f"• إعادة إرسال الرسالة لاحقاً"
        )
        await thinking.edit_text(error_msg, parse_mode="Markdown")
        return
    
    # استخراج البيانات من النتيجة
    try:
        is_phishing = result.get("is_phishing", False)
        confidence = result.get("confidence", 0)
        risk_level = result.get("risk_level", "unknown")
        label = "احتيالية ⚠️" if is_phishing else "آمنة ✅"
        emoji = RISK_EMOJI.get(risk_level, "⚪")
        conf_pct = int(confidence * 100)
        
        # الميزات
        features = result.get("features", {})
        has_url = "✅ نعم" if features.get("has_url") else "❌ لا"
        has_phone = "✅ نعم" if features.get("has_phone") else "❌ لا"
        keyword_count = features.get("keyword_count", 0)
        caps_ratio = features.get("caps_ratio", 0) * 100
        
        # بناء رسالة النتيجة
        reply = (
            f"{emoji} *نتيجة التحليل*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *التصنيف:* {label}\n"
            f"📊 *نسبة الثقة:* {conf_pct}%\n"
            f"⚠️ *مستوى الخطر:* {risk_level.upper()}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 *الميزات المكتشفة:*\n"
            f"• يحتوي رابط URL: {has_url}\n"
            f"• يحتوي رقم هاتف: {has_phone}\n"
            f"• كلمات مشبوهة: {keyword_count}\n"
            f"• نسبة الأحرف الكبيرة: {caps_ratio:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ *تنبيه:* النتائج إرشادية — استخدم حكمك دائماً."
        )
        
        # أزرار التفاعل (Feedback)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ النتيجة صحيحة", callback_data="fb_correct"),
                InlineKeyboardButton("❌ النتيجة خاطئة", callback_data="fb_wrong"),
            ]
        ])
        
        await thinking.edit_text(reply, parse_mode="Markdown", reply_markup=keyboard)
        
        # تحديث إحصائيات المستخدم
        context.user_data["total"] = context.user_data.get("total", 0) + 1
        if is_phishing:
            context.user_data["phishing"] = context.user_data.get("phishing", 0) + 1
        else:
            context.user_data["safe"] = context.user_data.get("safe", 0) + 1
            
    except KeyError as e:
        logger.error(f"Missing key in API response: {e}")
        await thinking.edit_text(
            "❌ خطأ في معالجة نتيجة التحليل.\n"
            "يرجى إبلاغ المطور إذا استمرت المشكلة."
        )

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يستقبل ردود فعل المستخدم على النتائج."""
    query = update.callback_query
    await query.answer()  # إزالة علامة التحميل من الزر
    
    if query.data == "fb_correct":
        await query.edit_message_reply_markup(None)  # إخفاء الأزرار
        await query.message.reply_text(
            "✅ *شكراً لتأكيدك!*\n"
            "ملاحظاتك تساعدنا في تحسين دقة النموذج. 🙏"
        )
    elif query.data == "fb_wrong":
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(
            "🙏 *شكراً لتصحيحك!*\n\n"
            "إذا كان لديك الوقت، يمكنك إرسال:\n"
            "• النص الصحيح للرسالة (إن أمكن)\n"
            "• سبب اعتقادك أن النتيجة خاطئة\n\n"
            "فريق التطوير يراجع جميع الملاحظات لتحسين البوت. 💪"
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الأخطاء العامة في البوت."""
    logger.error(f"Update {update} caused error: {context.error}", exc_info=True)
    
    if update and update.message:
        try:
            await update.message.reply_text(
                "⚠️ حدث خطأ غير متوقع في البوت.\n"
                "يرجى المحاولة لاحقاً أو استخدام /help للمساعدة."
            )
        except Exception:
            pass  # تجنب أخطاء متتالية

# ─── التشغيل الرئيسي ────────────────────────────────────────────────────────

def main():
    """الدالة الرئيسية لتشغيل البوت."""
    print("╔════════════════════════════════════════╗")
    print("║  SMS Phishing Detection — Telegram Bot ║")
    print("╚════════════════════════════════════════╝\n")
    
    # التحقق من التوكن
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ خطأ: لم يتم تعيين Telegram Bot Token!")
        print("\n💡 للحصول على التوكن:")
        print("   1. افتح تيليجرام وابحث عن @BotFather")
        print("   2. أرسل الأمر /newbot")
        print("   3. اتبع التعليمات لاختيار اسم البوت")
        print("   4. انسخ التوكن وضعه في:")
        print("      • ملف .env: TELEGRAM_TOKEN=your_token")
        print("      • أو في Railway Variables")
        return
    
    # التحقق من اتصال الـ API
    print(f"🔌 جاري التحقق من الاتصال بالخادم: {API_BASE_URL}")
    if check_api_health(API_BASE_URL):
        print("✅ متصل بالخادم بنجاح!")
    else:
        print("⚠️ تحذير: لا يمكن الاتصال بالخادم")
        print(f"   تأكد من: {API_BASE_URL}/health")
        print("   أو عدّل API_BASE_URL في المتغيرات البيئية")
    
    print(f"\n🤖 جاري تشغيل البوت: @{context.bot.username if 'context' in locals() else 'your_bot'}")
    print("   اضغط Ctrl+C للإيقاف\n")
    
    # إنشاء تطبيق البوت
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # تسجيل الـ Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("ping", ping_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_message))
    application.add_handler(CallbackQueryHandler(feedback_callback))
    application.add_error_handler(error_handler)
    
    # تشغيل البوت (Polling)
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البوت بواسطة المستخدم.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        print(f"\n❌ توقف البوت بسبب خطأ: {e}")

if __name__ == "__main__":
    main()