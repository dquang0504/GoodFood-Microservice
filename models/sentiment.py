from transformers import pipeline
# Cấu hình cho Sentiment Analysis model
model_path = "5CD-AI/Vietnamese-Sentiment-visobert"
sent_pipeline = pipeline("text-classification", model=model_path)