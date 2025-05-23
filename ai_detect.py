import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

# 文本特征提取
class TextFeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, text):
        features = {}
        
        # 文本长度
        features['text_length'] = len(text)
        
        # 句子数量
        sentences = re.split(r'[.!?]', text)
        features['sentence_count'] = len(sentences)
        
        # 平均句子长度
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # 标点符号比例
        punctuation_count = len(re.findall(r'[,.;:!?()]', text))
        features['punctuation_ratio'] = punctuation_count / len(text) if len(text) > 0 else 0
        
        # 大写字母比例
        uppercase_count = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = uppercase_count / len(text) if len(text) > 0 else 0
        
        # 重复词语检测
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
        
        # Markdown格式特征
        features['has_markdown_title'] = 1 if re.search(r'^#\s+', text, re.MULTILINE) else 0
        features['has_markdown_subtitle'] = 1 if re.search(r'^##\s+', text, re.MULTILINE) else 0
        
        # 数字编号特征
        features['has_number_prefix'] = 1 if re.search(r'^\d+\s+', text, re.MULTILINE) else 0
        
        return features

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, feature_extractor=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 提取手工特征
        if self.feature_extractor:
            features = self.feature_extractor.extract_features(text)
            feature_vector = torch.tensor([
                features['text_length'] / 5000,  # 归一化
                features['sentence_count'] / 200,
                features['avg_sentence_length'] / 100,
                features['punctuation_ratio'],
                features['uppercase_ratio'],
                features['unique_word_ratio'],
                features['has_markdown_title'],
                features['has_markdown_subtitle'],
                features['has_number_prefix']
            ], dtype=torch.float)
        else:
            feature_vector = None
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': feature_vector
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

# 模型定义
class BertClassifier(nn.Module):
    def __init__(self, bert_path, hidden_size=768, mid_size=256, feature_size=9):
        super(BertClassifier, self).__init__()
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_path)
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        # 特征融合层
        self.feature_layer = nn.Linear(feature_size, mid_size)
        # 结合BERT输出和手工特征
        self.combined_layer = nn.Linear(hidden_size + mid_size, mid_size)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(mid_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mid_size, 1),
        )

    def forward(self, input_ids, attention_mask, features=None):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 处理手工特征
        if features is not None:
            feature_output = F.relu(self.feature_layer(features))
            # 合并BERT输出和手工特征
            combined = torch.cat((pooled_output, feature_output), dim=1)
            combined = self.combined_layer(combined)
        else:
            combined = pooled_output
        
        # 分类
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

# 训练函数
def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5, model_save_path='models', resume_from=None):
    # 创建模型保存目录
    os.makedirs(model_save_path, exist_ok=True)
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_auc = 0.0
    best_model = None
    start_epoch = 0
    
    # 如果指定了恢复训练的模型文件
    if resume_from and os.path.exists(resume_from):
        print(f"从检查点恢复训练: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['val_auc']
        print(f"从epoch {start_epoch}继续训练，之前最佳验证AUC: {best_auc:.4f}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device) if 'features' in batch else None
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device) if 'features' in batch else None
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, features)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict().copy()
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_auc,
            }, os.path.join(model_save_path, 'best_model.pth'))
            print(f"保存最佳模型，验证AUC: {best_auc:.4f}")
    
    # 加载最佳模型
    if best_model:
        model.load_state_dict(best_model)
    
    return model, best_auc

# 预测函数
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device) if 'features' in batch else None
            
            outputs = model(input_ids, attention_mask, features)
            predictions.extend(outputs.squeeze().cpu().numpy())
    
    return predictions

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='AI生成文本检测模型训练和预测')
    parser.add_argument('--resume', type=str, default=None, help='从指定的模型文件恢复训练')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='评估批次大小')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)
    
    # 设置模型保存路径
    MODEL_SAVE_PATH = 'models'

    # 设置结果保存路径
    RESULTS_SAVE_PATH = 'results'
    TEST_PRED_RESULT_SAVE_PATH = f'{RESULTS_SAVE_PATH}/test_pred.jsonl'
    
    # 加载数据
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
    
    # 初始化tokenizer和特征提取器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    feature_extractor = TextFeatureExtractor()
    
    # 准备数据集
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, feature_extractor=feature_extractor)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, feature_extractor=feature_extractor)
    test_dataset = TextDataset(test_texts, tokenizer=tokenizer, feature_extractor=feature_extractor)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = BertClassifier('bert-base-uncased')
    model.to(device)
    
    # 训练模型
    model, best_auc = train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=args.epochs, 
        lr=args.lr, 
        model_save_path=MODEL_SAVE_PATH,
        resume_from=args.resume
    )
    print(f"最佳验证AUC: {best_auc:.4f}")
    
    # 预测测试集
    test_preds = predict(model, test_loader, device)
    
    # 保存结果
    results = [{"id": id_, "generated": float(pred)} for id_, pred in zip(test_ids, test_preds)]
    save_jsonl(TEST_PRED_RESULT_SAVE_PATH, results)
    print(f"预测结果已保存到 {TEST_PRED_RESULT_SAVE_PATH}")

if __name__ == "__main__":
    main() 