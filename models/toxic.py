from transformers import pipeline
# Toxic text detection pipeline
toxic_pipeline = pipeline(
    "text-classification",
    model="unitary/toxic-bert"  # model tiếng Anh chuyên detect toxicity
)