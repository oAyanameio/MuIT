import importlib
import sys

# 项目核心依赖库列表（根据代码和 README 整理）
required_libraries = [
    # 基础库
    "torch", "torchvision", "torchaudio",
    # 数据处理
    "numpy", "scipy", "sklearn", "pickle",
    # 多媒体处理
    "cv2", "librosa",
    # 模型相关
    "transformers",
    # 可选（CTC 模块需要）
    "warpctc_pytorch"
]

# 检查函数
def check_library(library):
    try:
        importlib.import_module(library)
        print(f"✅ 已安装: {library}")
        return True
    except ImportError:
        print(f"❌ 未安装: {library}")
        return False

# 批量检查
print("===== 依赖库检查结果 =====")
missing = [lib for lib in required_libraries if not check_library(lib)]

if not missing:
    print("\n🎉 所有必要库均已安装！")
else:
    print(f"\n❗ 缺少以下库，请安装后重试：{missing}")
    print("\n安装命令参考：")
    print("pip install " + " ".join(missing).replace("warpctc_pytorch", "git+https://github.com/SeanNaren/warp-ctc.git#subdirectory=pytorch_binding"))