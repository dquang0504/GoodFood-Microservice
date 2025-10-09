from sentence_splitter import SentenceSplitter
from models.sentiment import sent_pipeline

# ️️ Dùng sentence splitter đa ngôn ngữ
splitter = SentenceSplitter(language='en')  # hoặc 'en', nhưng 'vi' vẫn tách được cả 2 tốt

# Hàm phân tách câu
def split_into_clauses(sentence):
    """
    Chia văn bản thành các cụm câu dựa trên mô hình xử lý tiếng Việt.
    """
    clauses = splitter.split(text=sentence)
    print(clauses)
    return clauses


# Hàm phân tích cảm xúc của từng câu
def analyze_clauses(clauses):
    result = []
    for clause in clauses:
        sentiment = sent_pipeline(clause)[0]
        result.append({"clause": clause, "sentiment": sentiment["label"]})
    return result


# Hàm sinh nhận xét từ phân tích cảm xúc
def generate_summary(analysis):
    summary = []
    for item in analysis:
        clause = item['clause']
        sentiment = item['sentiment']
        if sentiment == "POS":
            summary.append(f"Khen {clause.strip()}.")
        elif sentiment == "NEG":
            summary.append(f"Chê {clause.strip()}.")
        else:
            summary.append(f"Ý kiến trung lập về {clause.strip()}.")
    return " ".join(summary)