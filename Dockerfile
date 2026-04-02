# اختيار بيئة بايثون خفيفة
FROM python:3.10-slim

# تحديد مسار العمل داخل السيرفر
WORKDIR /app

# نسخ ملف المكتبات وتسطيبها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كل ملفات المشروع (بما فيها الموديل)
COPY . .

# تشغيل السيرفر على بورت 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]