# 导入用于计算哈希值的库
import hashlib

# 从ComfyUI导入命令行参数配置
from comfy.cli_args import args

# 从PIL库导入图像处理相关的模块
from PIL import ImageFile, UnidentifiedImageError


# 设置条件值的函数
def conditioning_set_values(conditioning, values={}):
    # 创建一个新的列表来存储修改后的条件
    c = []
    # 遍历原始条件列表中的每个条件
    for t in conditioning:
        # 创建一个新的条件副本，避免修改原始数据
        # t[0]是条件的第一部分，t[1]是条件的元数据字典
        n = [t[0], t[1].copy()]
        # 将values字典中的所有键值对更新到新条件的元数据中
        for k in values:
            n[1][k] = values[k]
        # 将修改后的条件添加到新列表中
        c.append(n)

    # 返回修改后的条件列表
    return c


# 安全调用PIL图像操作的辅助函数
def pillow(fn, arg):
    # 用于保存原始的LOAD_TRUNCATED_IMAGES设置
    prev_value = None
    try:
        # 尝试直接调用传入的PIL函数
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):  # 处理PIL已知的问题（#4472和#2445）以及ComfyUI的#3416问题
        # 保存原始的LOAD_TRUNCATED_IMAGES设置
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        # 临时允许加载截断的图像文件
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # 再次尝试调用PIL函数
        x = fn(arg)
    finally:
        # 如果修改过LOAD_TRUNCATED_IMAGES设置，恢复原始值
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    # 返回PIL函数的执行结果
    return x


# 获取默认哈希函数的函数
def hasher():
    # 定义支持的哈希函数字典
    hashfuncs = {
        "md5": hashlib.md5,      # MD5哈希算法
        "sha1": hashlib.sha1,    # SHA-1哈希算法
        "sha256": hashlib.sha256,  # SHA-256哈希算法
        "sha512": hashlib.sha512   # SHA-512哈希算法
    }
    # 根据命令行参数返回对应的哈希函数
    return hashfuncs[args.default_hashing_function]