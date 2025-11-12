import urllib.request
import zipfile
import os
import sys

# 论文附录D指定的GloVe词向量
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
ZIP_FILE_NAME = "glove.zip"
TARGET_TXT = "glove.840B.300d.txt"


def main():
    print("=== 开始下载并解压 GloVe 词向量（约1.7GB）===")

    # 1. 下载文件（用Python内置urllib，替代wget）
    try:
        print(f"正在下载：{GLOVE_URL}")
        urllib.request.urlretrieve(
            GLOVE_URL,
            ZIP_FILE_NAME,
            reporthook=lambda count, block_size, total_size: print(
                f"下载进度：{min(count * block_size // (1024 * 1024), total_size // (1024 * 1024))}/{total_size // (1024 * 1024)}MB",
                end='\r'
            )
        )
        print("\n下载完成！")
    except Exception as e:
        print(f"\n下载失败：{str(e)}")
        print("备选方案：手动下载后放在当前目录，运行脚本自动解压")
        print("手动下载链接：https://nlp.stanford.edu/data/glove.840B.300d.zip")
        sys.exit(1)

    # 2. 解压文件（用Python内置zipfile，替代unzip）
    try:
        print("正在解压...")
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            # 只解压目标txt文件（避免多余文件）
            zip_ref.extract(TARGET_TXT)
        print(f"解压完成！文件路径：{os.path.abspath(TARGET_TXT)}")
    except Exception as e:
        print(f"解压失败：{str(e)}")
        sys.exit(1)

    # 3. 清理压缩包（可选，节省空间）
    try:
        os.remove(ZIP_FILE_NAME)
        print("已删除压缩包，释放空间")
    except:
        print("压缩包删除失败，可手动删除")


if __name__ == "__main__":
    main()