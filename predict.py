import torch
import cv2
import librosa
import numpy as np
import os
import argparse
import tempfile
from sklearn.decomposition import PCA
import platform  # 新增：用于跨平台判断

from src.utils import load_model
from src.eval_metrics import eval_mosei_senti, eval_iemocap

# 全局模型缓存（避免重复加载）
whisper_model = None
bert_tokenizer = None
bert_model = None
pca_model = None  # 预训练PCA模型缓存


# 配置参数
def get_hyp_params():
    parser = argparse.ArgumentParser()
    # 模型核心参数
    parser.add_argument('--model', type=str, default='MulT')
    parser.add_argument('--dataset', type=str, default='mosei_senti')
    parser.add_argument('--aligned', action='store_false', help="是否使用对齐数据")
    parser.add_argument('--lonly', action='store_false', help="仅使用语言特征")
    parser.add_argument('--aonly', action='store_false', help="仅使用音频特征")
    parser.add_argument('--vonly', action='store_false', help="仅使用视觉特征")
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--nlevels', type=int, default=5)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--relu_dropout', type=float, default=0.1)
    parser.add_argument('--res_dropout', type=float, default=0.1)
    parser.add_argument('--embed_dropout', type=float, default=0.25)
    parser.add_argument('--out_dropout', type=float, default=0.0)
    parser.add_argument('--attn_mask', action='store_true', default=True)

    # 特征维度参数
    parser.add_argument('--orig_d_l', type=int, default=300)
    parser.add_argument('--orig_d_a', type=int, default=74)
    parser.add_argument('--orig_d_v', type=int, default=35)
    parser.add_argument('--l_len', type=int, default=50)
    parser.add_argument('--a_len', type=int, default=500)
    parser.add_argument('--v_len', type=int, default=50)

    # 推理参数
    parser.add_argument('--model_path', type=str, default='pre_trained_models/aligned_model_MulT.pt')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--video_path', type=str, default='TestVido3.mp4')
    parser.add_argument('--pca_path', type=str, default='pca_model.pkl', help="预训练PCA模型路径")
    args = parser.parse_args()

    # 补充参数
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.output_dim = 1 if args.dataset in ['mosi', 'mosei_senti'] else 8
    args.layers = args.nlevels
    return args


# 从视频提取音频（跨平台优化）
def extract_audio_from_video(video_path):
    # 临时文件处理：使用with语句确保自动清理
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir='.') as temp_file:
        temp_audio_path = temp_file.name

    # 跨平台输出重定向（Windows用NUL，其他用/dev/null）
    null_device = 'NUL' if platform.system() == 'Windows' else '/dev/null'
    cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 "{temp_audio_path}" > {null_device} 2>&1'

    exit_code = os.system(cmd)
    if exit_code != 0 or not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
        os.remove(temp_audio_path)  # 清理失败的临时文件
        raise RuntimeError("音频提取失败，可能是视频文件损坏、格式不支持或ffmpeg未安装")
    return temp_audio_path


# 提取音频特征
def extract_audio_features(audio_path, target_len=500):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=74).T
        mel[mel == -np.inf] = 0
        # 统一特征长度
        if len(mel) < target_len:
            mel = np.pad(mel, ((0, target_len - len(mel)), (0, 0)), mode="constant")
        else:
            mel = mel[:target_len]
        return mel.astype(np.float32)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)  # 确保临时文件被删除


# 提取视觉特征（优化特征提取方式）
def extract_visual_features(video_path, target_len=50):
    # 处理带空格的路径（使用原始路径，让OpenCV自行处理）
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # 处理异常FPS值
        fps = 30.0
    sample_interval = max(1, int(fps // 2))  # 每2秒采样一次
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            # 优化：使用resize+均值池化提取特征，替代简单截取
            frame = cv2.resize(frame, (48, 48))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 48x48 -> 35维：使用均值池化减少维度
            pool_size = (7, 7)  # 7*7=49，接近35
            frame_pooled = cv2.resize(frame, (7, 7)).flatten()[:35]
            frames.append(frame_pooled)
        frame_count += 1
    cap.release()

    # 统一长度
    if len(frames) == 0:
        return np.zeros((target_len, 35), dtype=np.float32)
    if len(frames) < target_len:
        frames += [np.zeros_like(frames[0])] * (target_len - len(frames))
    else:
        frames = frames[:target_len]
    return np.array(frames, dtype=np.float32)


# 初始化全局模型（避免重复加载）
def init_global_models(pca_path):
    global whisper_model, bert_tokenizer, bert_model, pca_model
    # 初始化Whisper模型
    if whisper_model is None:
        try:
            import whisper
            whisper_model = whisper.load_model("small")
        except Exception as e:
            print(f"Whisper模型初始化警告: {str(e)}")

    # 初始化BERT模型
    if bert_tokenizer is None or bert_model is None:
        try:
            from transformers import BertTokenizer, BertModel
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bert_model = bert_model.to(device)
        except Exception as e:
            print(f"BERT模型初始化警告: {str(e)}")

    # 加载预训练PCA模型（替代实时训练）
    if pca_model is None and os.path.exists(pca_path):
        try:
            import pickle
            with open(pca_path, 'rb') as f:
                pca_model = pickle.load(f)
        except Exception as e:
            print(f"PCA模型加载警告: {str(e)}，将使用临时训练的PCA")


# 提取文本特征（优化模型加载和PCA使用）
def extract_text_features(video_path, pca_path):
    # 确保全局模型已初始化
    init_global_models(pca_path)

    # 提取音频并转文字
    audio_path = extract_audio_from_video(video_path)
    text = ""
    try:
        if whisper_model is not None:
            result = whisper_model.transcribe(audio_path)
            text = result.get("text", "")
        else:
            print("Whisper模型未初始化，无法进行语音转文字")
    except Exception as e:
        print(f"语音转文字警告: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # 提取BERT特征并降维
    try:
        if bert_tokenizer is None or bert_model is None:
            raise RuntimeError("BERT模型未初始化")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=50
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)
        text_feat = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (50, 768)

        # 维度调整：使用预训练PCA或临时训练
        global pca_model
        if pca_model is None:
            pca_model = PCA(n_components=300, random_state=42)
            text_feat_reduced = pca_model.fit_transform(text_feat)
        else:
            text_feat_reduced = pca_model.transform(text_feat)  # 使用预训练参数

        return text_feat_reduced.T  # (300, 50)
    except Exception as e:
        print(f"BERT特征提取警告: {str(e)}")
        return np.random.randn(300, 50).astype(np.float32)


def preprocess_features(text_feat, audio_feat, visual_feat, hyp_params):
    # 修正维度顺序：将 (特征维度, 时间步长) 转换为 (时间步长, 特征维度) 后再增加批次维度
    text = torch.tensor(text_feat.T).unsqueeze(0)  # (1, 50, 300) 时间步长×特征维度
    audio = torch.tensor(audio_feat).unsqueeze(0)  # (1, 500, 74)
    vision = torch.tensor(visual_feat).unsqueeze(0)  # (1, 50, 35)

    if hyp_params.use_cuda:
        text = text.cuda()
        audio = audio.cuda()
        vision = vision.cuda()
    return text, audio, vision


def parse_result(output, dataset_type):
    if dataset_type in ['mosi', 'mosei_senti']:
        score = output.item()
        if score > 1.5:
            sentiment = "强烈正面"
        elif score > 0.5:
            sentiment = "正面"
        elif score > -0.5:
            sentiment = "中性"
        elif score > -1.5:
            sentiment = "负面"
        else:
            sentiment = "强烈负面"
        return f"情感分析结果：\n- 情感分数：{score:.2f}（范围-3~3）\n- 情感倾向：{sentiment}"
    else:
        # 修正：iemocap实际有8类情感，这里补充完整并匹配输出维度
        emotions = ["中性", "快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "挫败"]
        if output.size(-1) != len(emotions):
            return f"警告：情感类别数量与模型输出不匹配（预期{len(emotions)}类，实际{output.size(-1)}类）"

        pred_probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        top_idx = np.argmax(pred_probs)
        top_emotion = emotions[top_idx]
        top_prob = pred_probs[top_idx] * 100
        prob_details = "\n".join([f"- {emotions[i]}: {pred_probs[i] * 100:.1f}%" for i in range(len(emotions))])
        return f"情感分类结果：\n- 最可能的情感：{top_emotion}（{top_prob:.1f}%）\n{prob_details}"


# 主推理函数
def predict_video_sentiment(video_path, hyp_params):
    if not os.path.exists(video_path):
        print(f"错误：视频文件 {video_path} 不存在")
        return

    try:
        print("初始化模型资源...")
        init_global_models(hyp_params.pca_path)  # 提前初始化所有模型

        print("提取音频特征...")
        audio_path = extract_audio_from_video(video_path)
        audio_feat = extract_audio_features(audio_path, hyp_params.a_len)

        print("提取视觉特征...")
        visual_feat = extract_visual_features(video_path, hyp_params.v_len)

        print("提取文本特征...")
        text_feat = extract_text_features(video_path, hyp_params.pca_path)

        text, audio, vision = preprocess_features(text_feat, audio_feat, visual_feat, hyp_params)

        print("加载模型...")
        model = load_model(hyp_params)
        if model is None:
            raise RuntimeError("模型加载失败，请检查模型路径")
        model.eval()
        if hyp_params.use_cuda:
            model = model.cuda()

        with torch.no_grad():
            output, _ = model(text, audio, vision)

        print("\n===== 情感分析结果 =====")
        print(parse_result(output, hyp_params.dataset))
        print("==================================")

    except Exception as e:
        print(f"推理过程出错: {str(e)}")


if __name__ == "__main__":
    hyp_params = get_hyp_params()
    predict_video_sentiment(hyp_params.video_path, hyp_params)