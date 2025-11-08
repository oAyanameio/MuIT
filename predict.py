import torch
import cv2
import librosa
import numpy as np
import os
import pickle
import tempfile
import platform
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import whisper

# 全局模型缓存
whisper_model = None
bert_tokenizer = None
bert_model = None
pca_model = None

def init_models():
    """初始化语音转文字、BERT和PCA模型"""
    global whisper_model, bert_tokenizer, bert_model, pca_model
    # 加载Whisper（语音转文字）
    if whisper_model is None:
        whisper_model = whisper.load_model("small")
    # 加载BERT（文本特征提取）
    if bert_tokenizer is None or bert_model is None:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').eval()
    # 初始化PCA（文本特征降维至300维）
    if pca_model is None:
        pca_model = PCA(n_components=300, random_state=42)
    return True

def extract_audio_features(video_path):
    """提取音频特征（74维梅尔频谱，长度500）"""
    # 临时保存音频
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_audio = f.name
    # 调用ffmpeg提取音频
    null_device = 'NUL' if platform.system() == 'Windows' else '/dev/null'
    os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 "{temp_audio}" > {null_device} 2>&1')
    
    # 提取梅尔频谱
    y, sr = librosa.load(temp_audio, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=74).T  # (seq_len, 74)
    mel[mel == -np.inf] = 0  # 处理异常值
    
    # 统一长度为500
    if len(mel) < 500:
        mel = np.pad(mel, ((0, 500 - len(mel)), (0, 0)), mode="constant")
    else:
        mel = mel[:500]
    os.remove(temp_audio)  # 清理临时文件
    return mel.astype(np.float32)

def extract_visual_features(video_path):
    """提取视觉特征（35维，长度50）"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = max(cap.get(cv2.CAP_PROP_FPS), 30.0)  # 处理异常FPS
    sample_interval = max(1, int(fps // 2))  # 每2秒采样一帧
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            # 缩放为48x48灰度图，降维至35维
            frame = cv2.resize(frame, (48, 48))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (7, 7)).flatten()[:35]  # 7x7=49，取前35维
            frames.append(frame)
        frame_count += 1
    cap.release()
    
    # 统一长度为50
    if len(frames) < 50:
        frames += [np.zeros(35)] * (50 - len(frames))
    else:
        frames = frames[:50]
    return np.array(frames, dtype=np.float32)

def extract_text_features(video_path):
    """提取文本特征（300维，长度50）"""
    # 提取音频并转文字
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_audio = f.name
    os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 "{temp_audio}" > /dev/null 2>&1')
    
    # 语音转文字
    result = whisper_model.transcribe(temp_audio)
    text = result.get("text", "")
    os.remove(temp_audio)
    
    # BERT提取特征（768维）
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=50  # 固定长度50
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    text_feat = outputs.last_hidden_state.squeeze(0).numpy()  # (50, 768)
    
    # PCA降维至300维
    text_feat_300 = pca_model.fit_transform(text_feat)  # (50, 300)
    return text_feat_300.astype(np.float32)

def mp4_to_mosei_features(video_path, output_path):
    """将MP4转换为MOSEI格式的特征文件（.pkl）"""
    # 初始化模型
    init_models()
    
    # 提取三模态特征
    audio_feat = extract_audio_features(video_path)
    visual_feat = extract_visual_features(video_path)
    text_feat = extract_text_features(video_path)
    
    # 构造与MOSEI一致的字典结构
    features = {
        'text': text_feat,       # (50, 300)
        'audio': audio_feat,     # (500, 74)
        'vision': visual_feat,   # (50, 35)
        'labels': np.array([[0.0]])  # 推理时可忽略标签，占位用
    }
    
    # 保存为.pkl文件
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"特征已保存至 {output_path}")
    return features

# 使用示例
if __name__ == "__main__":
    video_path = "test.mp4"  # 输入MP4文件
    output_path = "test_mosei_features.pkl"  # 输出特征文件
    mp4_to_mosei_features(video_path, output_path)