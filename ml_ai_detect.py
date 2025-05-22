import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 数据加载函数
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 保存结果为JSONL格式
def save_jsonl(file_path, datas):
    assert isinstance(datas, list), "datas should be a list"
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for data in datas:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

# 特征提取类
class TextFeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=5
        )
    
    def extract_statistical_features(self, texts):
        features = []
        for text in tqdm(texts, desc="提取统计特征"):
            # 文本长度
            text_length = len(text)
            
            # 句子数量和平均长度
            sentences = re.split(r'[.!?]', text)
            sentence_count = len(sentences)
            sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
            avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
            sentence_length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            # 标点符号比例
            punctuation_count = len(re.findall(r'[,.;:!?()]', text))
            punctuation_ratio = punctuation_count / text_length if text_length > 0 else 0
            
            # 大写字母比例
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_count / text_length if text_length > 0 else 0
            
            # 词汇多样性
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = set(words)
            unique_word_ratio = len(unique_words) / len(words) if len(words) > 0 else 0
            
            # 平均词长
            word_lengths = [len(w) for w in words if w]
            avg_word_length = np.mean(word_lengths) if word_lengths else 0
            
            # 重复词语比例
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            repeated_word_ratio = repeated_words / len(word_counts) if word_counts else 0
            
            # Markdown格式特征
            has_markdown_title = 1 if re.search(r'^#\s+', text, re.MULTILINE) else 0
            has_markdown_subtitle = 1 if re.search(r'^##\s+', text, re.MULTILINE) else 0
            
            # 数字编号特征
            has_number_prefix = 1 if re.search(r'^\d+\s+', text, re.MULTILINE) else 0
            
            # 行数
            line_count = text.count('\n') + 1
            
            # 特殊字符比例
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s,.;:!?()]', text))
            special_char_ratio = special_chars / text_length if text_length > 0 else 0
            
            features.append({
                'text_length': text_length,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'sentence_length_std': sentence_length_std,
                'punctuation_ratio': punctuation_ratio,
                'uppercase_ratio': uppercase_ratio,
                'unique_word_ratio': unique_word_ratio,
                'avg_word_length': avg_word_length,
                'repeated_word_ratio': repeated_word_ratio,
                'has_markdown_title': has_markdown_title,
                'has_markdown_subtitle': has_markdown_subtitle,
                'has_number_prefix': has_number_prefix,
                'line_count': line_count,
                'special_char_ratio': special_char_ratio
            })
        
        return pd.DataFrame(features)
    
    def fit_transform(self, texts):
        print("提取TF-IDF特征...")
        tfidf_features = self.tfidf.fit_transform(texts)
        
        print("提取统计特征...")
        statistical_features = self.extract_statistical_features(texts)
        
        return {
            'tfidf': tfidf_features,
            'statistical': statistical_features
        }
    
    def transform(self, texts):
        tfidf_features = self.tfidf.transform(texts)
        statistical_features = self.extract_statistical_features(texts)
        
        return {
            'tfidf': tfidf_features,
            'statistical': statistical_features
        }

# 模型训练与评估
def train_and_evaluate(X_train, y_train, X_val, y_val):
    print("训练模型...")
    # 梯度提升模型
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # 训练TF-IDF特征上的模型
    print("训练TF-IDF特征模型...")
    gb_model.fit(X_train['tfidf'], y_train)
    tfidf_preds = gb_model.predict_proba(X_val['tfidf'])[:, 1]
    tfidf_auc = roc_auc_score(y_val, tfidf_preds)
    print(f"TF-IDF特征AUC: {tfidf_auc:.4f}")
    
    # 训练统计特征上的模型
    print("训练统计特征模型...")
    statistical_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    statistical_model.fit(X_train['statistical'], y_train)
    statistical_preds = statistical_model.predict_proba(X_val['statistical'])[:, 1]
    statistical_auc = roc_auc_score(y_val, statistical_preds)
    print(f"统计特征AUC: {statistical_auc:.4f}")
    
    # 融合预测结果
    print("训练融合模型...")
    X_blend = np.column_stack((tfidf_preds, statistical_preds))
    blend_model = LogisticRegression(random_state=42)
    blend_model.fit(X_blend, y_val)
    
    # 在验证集上评估融合模型
    blend_preds = blend_model.predict_proba(X_blend)[:, 1]
    blend_auc = roc_auc_score(y_val, blend_preds)
    print(f"融合模型AUC: {blend_auc:.4f}")
    
    return {
        'tfidf_model': gb_model,
        'statistical_model': statistical_model,
        'blend_model': blend_model
    }

# 预测函数
def predict(models, X_test):
    # 使用TF-IDF模型预测
    tfidf_preds = models['tfidf_model'].predict_proba(X_test['tfidf'])[:, 1]
    
    # 使用统计特征模型预测
    statistical_preds = models['statistical_model'].predict_proba(X_test['statistical'])[:, 1]
    
    # 融合预测结果
    X_blend = np.column_stack((tfidf_preds, statistical_preds))
    final_preds = models['blend_model'].predict_proba(X_blend)[:, 1]
    
    return final_preds

def main():
    np.random.seed(42)
    
    print("加载数据...")
    train_data = load_data('data/train.jsonl')
    test_data = load_data('data/test.jsonl')
    
    # 提取文本和标签
    train_texts = [item['text'] for item in train_data]
    train_labels = [item['generated'] for item in train_data]
    test_texts = [item['text'] for item in test_data]
    test_ids = [item['id'] for item in test_data]
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 特征提取
    feature_extractor = TextFeatureExtractor()
    X_train = feature_extractor.fit_transform(train_texts)
    X_val = feature_extractor.transform(val_texts)
    X_test = feature_extractor.transform(test_texts)
    
    # 训练和评估模型
    models = train_and_evaluate(X_train, train_labels, X_val, val_labels)
    
    # 预测测试集
    print("预测测试集...")
    test_preds = predict(models, X_test)
    
    # 保存结果
    results = [{"id": id_, "generated": float(pred)} for id_, pred in zip(test_ids, test_preds)]
    save_jsonl('test_pred_ml.jsonl', results)
    print("预测结果已保存到 test_pred_ml.jsonl")

if __name__ == "__main__":
    main() 