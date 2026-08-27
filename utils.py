# Author:Hibari
# 2025年11月18日17时27分39秒
# utils.py
import os
import sys
import datetime
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')


class Tee:
    """把 stdout/stderr 同时写到屏幕和文件"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def init_logging(log_dir="logs"):
    """
    初始化日志系统：
    - 在 log_dir 下创建一个“开始运行时间.txt”的日志文件
    - 所有 print() 的内容会同时显示在屏幕和写入这个文件
    """
    os.makedirs(log_dir, exist_ok=True)

    # Windows consoles may default to GBK and fail on symbols such as ✓.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # 日志文件名：开始运行时间.txt，例如 2025-11-18_16-20-33.txt
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{start_time}.txt")

    log_file = open(log_path, "w", encoding="utf-8")

    # 把 stdout / stderr 都重定向成“屏幕 + 文件”
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    return log_path


# 工具函数，确保实验结果可复现
def set_seed(seed=42):
    """设置所有随机种子以确保实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 自动检测并配置GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_device_info():
    """把设备信息打印出来（此时已经开启日志的话，会一起写入日志）"""
    print("=" * 80)
    print(" 超图神经网络 - snoRNA-Disease关联预测")
    print("=" * 80)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
