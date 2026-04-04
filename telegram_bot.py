"""
SMS Phishing Detection — Telegram Bot
======================================
بوت تيليجرام يحلل أي رسالة بيبعتها المستخدم.
معدل للعمل كـ Web Service مجانية على منصة Render.
"""

import os
import time
import logging
import requests
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

# ─── خدعة السيرفر الوهمي (عشان Render المجاني) ─────────────────────────
class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Telegram Bot is running smoothly on Render!")
        
    def do_HEAD(self):  # ضفنا السطرين دول عشان UptimeRobot
        self.send_response(200)
        self.end_headers()

def keep_alive():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_ENDPOINT = "/predict"
API_URL = f"{API_BASE_URL.rstrip('/')}{API_ENDPOINT}"

MAX_RETRIES = 3
RETRY_DELAY = 2

RISK_EMOJI = {
    "high":   "🔴",
    "medium": "🟡", 
    "low":    "🟢",
}

# ─── دوال مساعدة ────────────────────────────────────────────────────────────
def check_api_health(base_url: str, timeout: int = 5) -> bool:
    try:
        health_url = f"{base_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False

def call_api_with_retry(text: str, source: str = "telegram") -> dict:
    payload = {"text": text, "source": source}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL, json=payload, timeout=15,
                headers={"User-Agent": "SMS-Phishing-Bot/1.0"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            if attempt == MAX_RETRIES - 1:
                return {"error": "تعذر الاتصال بالخادم. تأكد من تشغيل الـ Backend."}
        except requests.exceptions.Timeout as e:
            if attempt == MAX_RETRIES - 1:
                return {"error": "انتهت مهلة الطلب. يرجى المحاولة لاحقاً."}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"error": f"خطأ غير متوقع: {str(e)}"}
        time.sleep(RETRY_DELAY)
    return {"error": "فشل الاتصال بعد عدة محاولات"}

# ─── Handlers ───────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    api_status = "✅ متصل بالخادم" if check_api_health(API_BASE_URL) else "❌ غير متصل"
    welcome = (
        f"👋 *مرحباً بك في بوت كشف التصيد الاحتيالي* 🛡️\n\n"
        f"📩 *طريقة الاستخدام:*\nأرسل لي أي رسالة مشبوهة وسأخبرك فوراً نتيجتها.\n\n"
        f"🔧 *حالة النظام:* {api_status}\n\n"
        f"📋 *الأوامر المتاحة:*\n"
        f"/start — رسالة الترحيب\n/help  — شرح مفصل للاستخدام\n"
        f"/stats — إحصائيات تحليلاتك\n/ping  — فحص اتصال البوت بالخادم"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 *كيفية استخدام البوت:*\n"
        "انسخ الرسالة المشبوهة، ألصقها هنا، وانتظر التقرير الفوري.\n\n"
        "💡 *نصائح للأمان:* لا تضغط على روابط مجهولة ولا تشارك بياناتك."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("🔄 جاري فحص الاتصال...")
    if check_api_health(API_BASE_URL):
        reply = f"✅ *البوت متصل بالخادم بنجاح!*\nURL: `{API_BASE_URL}`"
    else:
        reply = "❌ *لا يمكن الاتصال بالخادم*"
    await status_msg.edit_text(reply, parse_mode="Markdown")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = context.user_data
    total = data.get("total", 0)
    if total == 0:
        await update.message.reply_text("📭 لم تحلل أي رسائل بعد في هذه الجلسة.")
        return
    phish = data.get("phishing", 0)
    safe = data.get("safe", 0)
    phishing_rate = (phish / total * 100)
    stats_text = (f"📊 *إحصائيات الجلسة:*\nإجمالي: *{total}*\n🔴 احتيالية: *{phish}*\n"
                  f"🟢 آمنة: *{safe}*\n📈 نسبة التصيد: *{phishing_rate:.1f}%*")
    await update.message.reply_text(stats_text, parse_mode="Markdown")

async def analyze_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text or not text.strip(): return
    thinking = await update.message.reply_text("🔍 جاري تحليل الرسالة...")
    
    result = call_api_with_retry(text)
    if "error" in result:
        await thinking.edit_text(f"❌ *خطأ:* {result['error']}", parse_mode="Markdown")
        return
    
    try:
        is_phishing = result.get("is_phishing", False)
        conf_pct = int(result.get("confidence", 0) * 100)
        risk_level = result.get("risk_level", "unknown")
        label = "احتيالية ⚠️" if is_phishing else "آمنة ✅"
        emoji = RISK_EMOJI.get(risk_level, "⚪")
        features = result.get("features", {})
        
        reply = (
            f"{emoji} *النتيجة:* {label}\n📊 *الثقة:* {conf_pct}%\n"
            f"⚠️ *الخطر:* {risk_level.upper()}\n━━━━━━━━━━\n"
            f"• رابط URL: {'✅' if features.get('has_url') else '❌'}\n"
            f"• رقم هاتف: {'✅' if features.get('has_phone') else '❌'}\n"
            f"• كلمات مشبوهة: {features.get('keyword_count', 0)}\n"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ صحيحة", callback_data="fb_correct"),
             InlineKeyboardButton("❌ خاطئة", callback_data="fb_wrong")]
        ])
        await thinking.edit_text(reply, parse_mode="Markdown", reply_markup=keyboard)
        
        context.user_data["total"] = context.user_data.get("total", 0) + 1
        if is_phishing: context.user_data["phishing"] = context.user_data.get("phishing", 0) + 1
        else: context.user_data["safe"] = context.user_data.get("safe", 0) + 1
            
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        await thinking.edit_text("❌ خطأ في عرض النتيجة.")

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(None)
    msg = "✅ شكراً لتأكيدك!" if query.data == "fb_correct" else "🙏 شكراً لملاحظتك، سنقوم بتحسين الموديل."
    await query.message.reply_text(msg)

# ─── التشغيل الرئيسي ────────────────────────────────────────────────────────
def main():
    # 1. تشغيل السيرفر الوهمي في الخلفية (عشان Render ميفصلش البوت)
    threading.Thread(target=keep_alive, daemon=True).start()
    print("🌐 السيرفر الوهمي يعمل الآن للحفاظ على استقرار Render...")
    
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ خطأ: تأكد من إضافة TELEGRAM_TOKEN في المتغيرات البيئية!")
        return
        
    print("🤖 جاري تشغيل بوت التليجرام...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("ping", ping_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_message))
    application.add_handler(CallbackQueryHandler(feedback_callback))
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
