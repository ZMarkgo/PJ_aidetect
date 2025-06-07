import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置全局绘图样式
plt.rcParams['figure.figsize'] = (12, 8)  # 设置默认图片大小
plt.rcParams['savefig.dpi'] = 300  # 设置保存图片的DPI
plt.rcParams['font.size'] = 12  # 设置默认字体大小

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_text_length(texts, labels=None):
    """分析文本长度分布"""
    lengths = [len(text) for text in texts]
    
    if labels is not None:
        human_lengths = [len(text) for text, label in zip(texts, labels) if label < 0.5]
        ai_lengths = [len(text) for text, label in zip(texts, labels) if label >= 0.5]
        
        plt.figure(figsize=(10, 6))
        plt.hist(human_lengths, bins=50, alpha=0.5, label='人类撰写')
        plt.hist(ai_lengths, bins=50, alpha=0.5, label='AI生成')
        plt.xlabel('文本长度')
        plt.ylabel('频率')
        plt.title('人类撰写与AI生成文本的长度分布')
        plt.legend()
        plt.savefig('results/text_length_distribution.png')
        
        print(f"人类撰写文本平均长度: {np.mean(human_lengths):.2f}")
        print(f"AI生成文本平均长度: {np.mean(ai_lengths):.2f}")
    
    print(f"整体文本平均长度: {np.mean(lengths):.2f}")
    print(f"文本长度中位数: {np.median(lengths):.2f}")
    print(f"文本长度最小值: {min(lengths)}")
    print(f"文本长度最大值: {max(lengths)}")

def analyze_sentence_length(texts, labels=None):
    """分析句子长度分布"""
    all_sentence_lengths = []
    human_sentence_lengths = []
    ai_sentence_lengths = []
    
    for i, text in enumerate(texts):
        sentences = re.split(r'[.!?]', text)
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        all_sentence_lengths.extend(sentence_lengths)
        
        if labels is not None:
            if labels[i] < 0.5:
                human_sentence_lengths.extend(sentence_lengths)
            else:
                ai_sentence_lengths.extend(sentence_lengths)
    
    if labels is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(human_sentence_lengths, bins=50, alpha=0.5, label='人类撰写')
        plt.hist(ai_sentence_lengths, bins=50, alpha=0.5, label='AI生成')
        plt.xlabel('句子长度')
        plt.ylabel('频率')
        plt.title('人类撰写与AI生成文本的句子长度分布')
        plt.legend()
        plt.savefig('results/sentence_length_distribution.png')
        
        print(f"人类撰写文本平均句子长度: {np.mean(human_sentence_lengths):.2f}")
        print(f"AI生成文本平均句子长度: {np.mean(ai_sentence_lengths):.2f}")
    
    print(f"整体平均句子长度: {np.mean(all_sentence_lengths):.2f}")
    print(f"句子长度中位数: {np.median(all_sentence_lengths):.2f}")
    print(f"句子长度最小值: {min(all_sentence_lengths)}")
    print(f"句子长度最大值: {max(all_sentence_lengths)}")

def analyze_punctuation(texts, labels=None):
    """分析标点符号使用情况"""
    all_punctuation_ratios = []
    human_punctuation_ratios = []
    ai_punctuation_ratios = []
    
    for i, text in enumerate(texts):
        punctuation_count = len(re.findall(r'[,.;:!?()]', text))
        ratio = punctuation_count / len(text) if len(text) > 0 else 0
        all_punctuation_ratios.append(ratio)
        
        if labels is not None:
            if labels[i] < 0.5:
                human_punctuation_ratios.append(ratio)
            else:
                ai_punctuation_ratios.append(ratio)
    
    if labels is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(human_punctuation_ratios, bins=50, alpha=0.5, label='人类撰写')
        plt.hist(ai_punctuation_ratios, bins=50, alpha=0.5, label='AI生成')
        plt.xlabel('标点符号比例')
        plt.ylabel('频率')
        plt.title('人类撰写与AI生成文本的标点符号使用比例')
        plt.legend()
        plt.savefig('results/punctuation_ratio_distribution.png')
        
        print(f"人类撰写文本标点符号比例: {np.mean(human_punctuation_ratios):.4f}")
        print(f"AI生成文本标点符号比例: {np.mean(ai_punctuation_ratios):.4f}")
    
    print(f"整体标点符号比例: {np.mean(all_punctuation_ratios):.4f}")

def analyze_word_uniqueness(texts, labels=None):
    """分析词汇多样性"""
    all_unique_ratios = []
    human_unique_ratios = []
    ai_unique_ratios = []
    
    for i, text in enumerate(texts):
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        ratio = len(unique_words) / len(words) if len(words) > 0 else 0
        all_unique_ratios.append(ratio)
        
        if labels is not None:
            if labels[i] < 0.5:
                human_unique_ratios.append(ratio)
            else:
                ai_unique_ratios.append(ratio)
    
    if labels is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(human_unique_ratios, bins=50, alpha=0.5, label='人类撰写')
        plt.hist(ai_unique_ratios, bins=50, alpha=0.5, label='AI生成')
        plt.xlabel('词汇多样性比例')
        plt.ylabel('频率')
        plt.title('人类撰写与AI生成文本的词汇多样性')
        plt.legend()
        plt.savefig('results/word_uniqueness_distribution.png')
        
        print(f"人类撰写文本词汇多样性: {np.mean(human_unique_ratios):.4f}")
        print(f"AI生成文本词汇多样性: {np.mean(ai_unique_ratios):.4f}")
    
    print(f"整体词汇多样性: {np.mean(all_unique_ratios):.4f}")

def analyze_markdown_usage(texts, labels=None):
    """分析Markdown格式使用情况"""
    title_count = sum(1 for text in texts if re.search(r'^#\s+', text, re.MULTILINE))
    subtitle_count = sum(1 for text in texts if re.search(r'^##\s+', text, re.MULTILINE))
    number_prefix_count = sum(1 for text in texts if re.search(r'^\d+\s+', text, re.MULTILINE))
    
    if labels is not None:
        human_texts = [text for text, label in zip(texts, labels) if label < 0.5]
        ai_texts = [text for text, label in zip(texts, labels) if label >= 0.5]
        
        human_title_count = sum(1 for text in human_texts if re.search(r'^#\s+', text, re.MULTILINE))
        human_subtitle_count = sum(1 for text in human_texts if re.search(r'^##\s+', text, re.MULTILINE))
        human_number_prefix_count = sum(1 for text in human_texts if re.search(r'^\d+\s+', text, re.MULTILINE))
        
        ai_title_count = sum(1 for text in ai_texts if re.search(r'^#\s+', text, re.MULTILINE))
        ai_subtitle_count = sum(1 for text in ai_texts if re.search(r'^##\s+', text, re.MULTILINE))
        ai_number_prefix_count = sum(1 for text in ai_texts if re.search(r'^\d+\s+', text, re.MULTILINE))
        
        print(f"人类撰写文本使用标题比例: {human_title_count/len(human_texts):.4f}")
        print(f"AI生成文本使用标题比例: {ai_title_count/len(ai_texts):.4f}")
        print(f"人类撰写文本使用子标题比例: {human_subtitle_count/len(human_texts):.4f}")
        print(f"AI生成文本使用子标题比例: {ai_subtitle_count/len(ai_texts):.4f}")
        print(f"人类撰写文本使用数字编号比例: {human_number_prefix_count/len(human_texts):.4f}")
        print(f"AI生成文本使用数字编号比例: {ai_number_prefix_count/len(ai_texts):.4f}")
    
    print(f"整体使用标题比例: {title_count/len(texts):.4f}")
    print(f"整体使用子标题比例: {subtitle_count/len(texts):.4f}")
    print(f"整体使用数字编号比例: {number_prefix_count/len(texts):.4f}")

def count_label_distribution(labels):
    """统计标签分布"""
    human_count = sum(1 for label in labels if label < 0.5)
    ai_count = sum(1 for label in labels if label >= 0.5)
    
    print(f"人类撰写文本数量: {human_count} ({human_count/len(labels):.2%})")
    print(f"AI生成文本数量: {ai_count} ({ai_count/len(labels):.2%})")

def analyze_prompt_distribution(data):
    """分析提示ID分布"""
    prompt_ids = [item['prompt_id'] for item in data if 'prompt_id' in item]
    prompt_counter = Counter(prompt_ids)
    
    print(f"不同提示ID数量: {len(prompt_counter)}")
    print("最常见的5个提示ID:")
    for prompt_id, count in prompt_counter.most_common(5):
        print(f"  提示ID {prompt_id}: {count} 次 ({count/len(prompt_ids):.2%})")

def main():
    print("加载训练数据...")
    train_data = load_data('data/train.jsonl')
    
    print("加载测试数据...")
    test_data = load_data('data/test.jsonl')
    
    # 提取文本和标签
    train_texts = [item['text'] for item in train_data]
    train_labels = [item['generated'] for item in train_data]
    test_texts = [item['text'] for item in test_data]
    
    print("\n数据集基本信息:")
    print(f"训练集样本数: {len(train_texts)}")
    print(f"测试集样本数: {len(test_texts)}")
    
    print("\n标签分布:")
    count_label_distribution(train_labels)
    
    print("\n提示ID分布:")
    print("训练集:")
    analyze_prompt_distribution(train_data)
    print("测试集:")
    analyze_prompt_distribution(test_data)
    
    print("\n文本长度分析:")
    analyze_text_length(train_texts, train_labels)
    
    print("\n句子长度分析:")
    analyze_sentence_length(train_texts[:1000], train_labels[:1000])  # 取部分样本分析以加快速度
    
    print("\n标点符号使用分析:")
    analyze_punctuation(train_texts[:1000], train_labels[:1000])
    
    print("\n词汇多样性分析:")
    analyze_word_uniqueness(train_texts[:1000], train_labels[:1000])
    
    print("\nMarkdown格式使用分析:")
    analyze_markdown_usage(train_texts, train_labels)

if __name__ == "__main__":
    main() 