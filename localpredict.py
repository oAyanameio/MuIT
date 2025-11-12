import os
import numpy as np
import torch
import torch.nn as nn
import subprocess
from gensim.models import KeyedVectors
import whisper
from pathlib import Path
import sys
from typing import Dict, Tuple
import librosa  # 仅用 librosa 实现 74 维音频特征

# -------------------------- 核心配置（Python3.10+附录D规范）--------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIMS = {"L": 300, "V": 35, "A": 74}  # 附录D特征维度（音频74维不变）
HIDDEN_DIM = 40
CONV_KERNELS = {"L": 3, "V": 3, "A": 3}
GLOVE_PATH = "glove.840B.300d.txt"
TEMP_DIR = "temp_appendixD_py310"
PRETRAINED_MODEL_PATH = "pre_trained_models/aligned_model_MulT.pt"  # 替换为你的模型路径
os.makedirs(TEMP_DIR, exist_ok=True)


# -------------------------- 依赖加载（删除 pycovarep）--------------------------
def load_glove() -> KeyedVectors:
    print("加载附录D指定GloVe词向量...")
    try:
        return KeyedVectors.load_word2vec_format(
            GLOVE_PATH, binary=False, no_header=True, encoding="utf-8"
        )
    except FileNotFoundError:
        print(f"错误：请下载GloVe文件：{GLOVE_PATH}")
        sys.exit(1)


def load_whisper() -> whisper.Whisper:
    print("加载语音转文字模型...")
    try:
        return whisper.load_model("base", device=DEVICE, in_memory=True)
    except Exception as e:
        print(f"错误：Whisper加载失败：{str(e)}")
        sys.exit(1)


GLOVE_MODEL = load_glove()
WHISPER_MODEL = load_whisper()


# -------------------------- 模态拆分（不变）--------------------------
def split_modalities(mp4_path: str) -> Dict[str, str]:
    mp4_name = Path(mp4_path).stem
    audio_path = os.path.join(TEMP_DIR, f"{mp4_name}_audio.wav")
    frames_dir = os.path.join(TEMP_DIR, f"{mp4_name}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 提取音频
    print("提取原始音频流...")
    result = subprocess.run([
        "ffmpeg", "-i", mp4_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path, "-y"
    ], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"ffmpeg音频提取失败：{result.stderr}")
        sys.exit(1)

    # 提取视频帧
    print("提取15Hz视频帧...")
    result = subprocess.run([
        "ffmpeg", "-i", mp4_path, "-r", "15", "-f", "image2",
        os.path.join(frames_dir, "frame_%04d.jpg"), "-y"
    ], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"ffmpeg帧提取失败：{result.stderr}")
        sys.exit(1)

    # 转录文本
    print("转录音频为文本...")
    result = WHISPER_MODEL.transcribe(
        audio_path, language="en", fp16=torch.cuda.is_available(), verbose=False
    )
    return {
        "text": result["text"].strip(),
        "audio_path": audio_path,
        "frames_dir": frames_dir
    }


# -------------------------- 特征提取（核心修改音频部分）--------------------------
def extract_language_feat(text: str) -> np.ndarray:
    """附录D：300维GloVe词嵌入"""
    words = text.lower().split()
    embeddings: list[np.ndarray] = []
    for word in words:
        emb = GLOVE_MODEL[word].astype(np.float32) if word in GLOVE_MODEL else np.zeros(300, dtype=np.float32)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def extract_vision_feat(frames_dir: str) -> np.ndarray:
    """附录D：35维AU特征"""
    print("提取35维AU特征...")
    openface_csv = os.path.join(TEMP_DIR, "openface_au.csv")
    cmd = [
        "FeatureExtraction.exe" if os.name == "nt" else "FeatureExtraction",
        "-fdir", frames_dir, "-out_dir", TEMP_DIR, "-au_static", "-no_gaze", "-no_3dface"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"OpenFace提取失败：{result.stderr}")
        return np.zeros((15, FEAT_DIMS["V"]), dtype=np.float32)

    try:
        au_data = np.loadtxt(
            openface_csv, delimiter=",", skiprows=1, usecols=range(22, 57),
            dtype=np.float32, encoding="utf-8"
        )
        return au_data
    except Exception as e:
        print(f"AU特征读取失败：{str(e)}")
        return np.zeros((15, FEAT_DIMS["V"]), dtype=np.float32)


def extract_audio_feat(audio_path: str) -> np.ndarray:
    """附录D：74维音频特征（用librosa实现，替代COVAREP）"""
    print("提取附录D要求的74维音频特征（librosa实现）...")
    y, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
    feat_len = len(y) // 800  # 20Hz采样率（与COVAREP一致）
    feat = np.zeros((feat_len, FEAT_DIMS["A"]), dtype=np.float32)  # (T_A, 74)

    # 1. MFCC特征（12维，附录D核心声学特征）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_fft=512, hop_length=800)
    feat[:, :12] = mfcc.T[:feat_len]  # 对齐长度

    # 2. 频谱特征（20维）
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=800).T[:feat_len]  # 1维
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=800).T[:feat_len]  # 1维
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=800).T[:feat_len]  # 1维
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=800).T[:feat_len]  # 1维
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=16, hop_length=800).T[:feat_len]  # 16维
    feat[:, 12:13] = spectral_centroid
    feat[:, 13:14] = spectral_bandwidth
    feat[:, 14:15] = spectral_rolloff
    feat[:, 15:16] = spectral_flatness
    feat[:, 16:32] = mel_spectrogram

    # 3. 音频能量特征（4维）
    rms = librosa.feature.rms(y=y, hop_length=800).T[:feat_len]  # 1维
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, hop_length=800).T[:feat_len]  # 1维
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=800).T[:feat_len, :2]  # 2维
    feat[:, 32:33] = rms
    feat[:, 33:34] = zero_crossing_rate
    feat[:, 34:36] = spectral_contrast

    # 4. 基音频率相关特征（8维，模拟COVAREP基音特征）
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=800)
    f0 = np.nan_to_num(f0).reshape(-1, 1)[:feat_len]  # 基音频率（1维）
    f0_deriv = np.gradient(f0, axis=0)  # 基音一阶导数（1维）
    f0_deriv2 = np.gradient(f0_deriv, axis=0)  # 基音二阶导数（1维）
    harmonicity = librosa.feature.mfcc(y=librosa.effects.harmonic(y), sr=sr, n_mfcc=5, hop_length=800).T[
        :feat_len]  # 5维
    feat[:, 36:37] = f0
    feat[:, 37:38] = f0_deriv
    feat[:, 38:39] = f0_deriv2
    feat[:, 39:44] = harmonicity

    # 5. 补充特征（30维，确保总维度74）
    # 频谱熵+MFCC一阶/二阶导数
    spectral_entropy = -np.sum(np.square(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=800).T[:feat_len]),
                               axis=1, keepdims=True)  # 1维
    mfcc_delta = librosa.feature.delta(mfcc).T[:feat_len]  # 12维（MFCC一阶导数）
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2).T[:feat_len]  # 12维（MFCC二阶导数）
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=800).T[:feat_len, :5]  # 5维（音高类特征）
    feat[:, 44:45] = spectral_entropy
    feat[:, 45:57] = mfcc_delta
    feat[:, 57:69] = mfcc_delta2
    feat[:, 69:74] = chroma

    return feat  # 最终输出 (T_A, 74)，严格匹配附录D维度


# -------------------------- 预处理模块（不变）--------------------------
class AppendixDPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_L: nn.Conv1d = nn.Conv1d(FEAT_DIMS["L"], HIDDEN_DIM, kernel_size=CONV_KERNELS["L"], padding=1)
        self.conv_V: nn.Conv1d = nn.Conv1d(FEAT_DIMS["V"], HIDDEN_DIM, kernel_size=CONV_KERNELS["V"], padding=1)
        self.conv_A: nn.Conv1d = nn.Conv1d(FEAT_DIMS["A"], HIDDEN_DIM, kernel_size=CONV_KERNELS["A"], padding=1)
        self.relu = nn.ReLU()

    def positional_embedding(self, seq_len: int) -> torch.Tensor:
        PE = torch.zeros(seq_len, HIDDEN_DIM, device=DEVICE, dtype=torch.float32)
        position = torch.arange(0, seq_len, dtype=torch.float32, device=DEVICE)[:, None]
        div_term = torch.exp(
            torch.arange(0, HIDDEN_DIM, 2, device=DEVICE, dtype=torch.float32)
            * (-np.log(10000.0) / HIDDEN_DIM)
        )
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        return PE

    def forward(
            self, L: torch.Tensor, V: torch.Tensor, A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = self.relu(self.conv_L(L.permute(0, 2, 1))).permute(0, 2, 1)
        V = self.relu(self.conv_V(V.permute(0, 2, 1))).permute(0, 2, 1)
        A = self.relu(self.conv_A(A.permute(0, 2, 1))).permute(0, 2, 1)
        L += self.positional_embedding(L.shape[1])
        V += self.positional_embedding(V.shape[1])
        A += self.positional_embedding(A.shape[1])
        return L, V, A


# -------------------------- 推理核心（不变）--------------------------
class AppendixDInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = AppendixDPreprocess().to(DEVICE)
        self.pretrained_model = self.load_pretrained()

    def load_pretrained(self) -> nn.Module:
        print(f"加载预训练模型：{PRETRAINED_MODEL_PATH}")
        if not os.path.exists(PRETRAINED_MODEL_PATH):
            print("错误：未找到预训练模型文件")
            sys.exit(1)

        try:
            model_data = torch.load(
                PRETRAINED_MODEL_PATH,
                map_location=DEVICE,
                weights_only=True
            )
        except Exception as e:
            print(f"模型加载失败：{str(e)}")
            sys.exit(1)

        if isinstance(model_data, nn.Module):
            return model_data.eval()
        elif isinstance(model_data, dict):
            try:
                from modules.mult import MulT  # 替换为你的模型定义路径
            except ImportError:
                print("错误：未找到MulT模型定义，请确保modules.mult模块存在")
                sys.exit(1)

            mult_model = MulT(
                hidden_dim=HIDDEN_DIM,
                cross_attn_heads=8,
                num_cross_attn_layers=4
            ).to(DEVICE)
            mult_model.load_state_dict(model_data, strict=False)
            return mult_model.eval()
        else:
            raise ValueError("不支持的模型格式（仅支持nn.Module或state_dict）")

    @torch.no_grad()
    def forward(self, mp4_path: str) -> Dict[str, any]:
        # 1. 模态拆分
        modalities = split_modalities(mp4_path)
        # 2. 特征提取
        L_feat = extract_language_feat(modalities["text"])
        V_feat = extract_vision_feat(modalities["frames_dir"])
        A_feat = extract_audio_feat(modalities["audio_path"])
        # 3. 张量转换
        L = torch.tensor(L_feat, device=DEVICE).unsqueeze(0)
        V = torch.tensor(V_feat, device=DEVICE).unsqueeze(0)
        A = torch.tensor(A_feat, device=DEVICE).unsqueeze(0)
        # 4. 预处理
        L_pre, V_pre, A_pre = self.preprocess(L, V, A)
        # 5. 预测
        with torch.inference_mode():
            pred = self.pretrained_model(L_pre, V_pre, A_pre).squeeze().item()
        # 6. 结果格式化
        return {
            "transcript": modalities["text"],
            "raw_prediction": round(pred, 4),
            "sentiment_score": np.clip(pred, -3.0, 3.0),
            "sentiment_label": "Positive" if pred > 0 else "Negative" if pred < 0 else "Neutral"
        }


# -------------------------- 推理入口（不变）--------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="附录D预处理推理（Python3.10+Windows）")
    parser.add_argument("--mp4-path", required=True, help="输入MP4视频路径")
    args = parser.parse_args()

    inference = AppendixDInference()
    input_mp4 = args.mp4_path

    if not (os.path.exists(input_mp4) or input_mp4.startswith(("http://", "https://"))):
        print(f"错误：视频文件不存在或URL无效：{input_mp4}")
        sys.exit(1)

    print(f"\n开始处理视频：{input_mp4}")
    result = inference(input_mp4)
    print("\n=== 附录D预处理规范推理结果 ===")
    print(f"转录文本：{result['transcript']}")
    print(f"原始预测值：{result['raw_prediction']}")
    print(f"情感得分（-3~3）：{result['sentiment_score']:.2f}")
    print(f"情感标签：{result['sentiment_label']}")