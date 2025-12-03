# 导入所需的库和模块
import hashlib  # 用于计算文件哈希值，检测文件是否变化
import os  # 用于文件和目录操作
import folder_paths  # ComfyUI提供的工具，用于获取输入输出路径
import numpy as np  # 用于数值计算和数组操作
import torch  # PyTorch深度学习框架
import node_helpers  # ComfyUI提供的节点辅助函数
from PIL import Image, ImageOps, ImageSequence  # Python图像处理库
from comfy.comfy_types import IO, ComfyNodeABC  # ComfyUI的类型定义
from comfy_api.latest import InputImpl  # ComfyUI的API输入实现


# 图像加载节点类
class ImageLoader:
    # 定义节点的输入参数类型
    @classmethod
    def INPUT_TYPES(s):
        # 获取ComfyUI的输入目录
        input_dir = folder_paths.get_input_directory()
        # 获取输入目录中所有支持的图像文件
        files = [
            f
            for f in os.listdir(input_dir)  # 遍历输入目录中的所有文件
            if os.path.isfile(os.path.join(input_dir, f))  # 只处理文件（不是目录）
            and f.split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]  # 只接受这些图像格式
        ]
        # 返回输入参数定义：一个名为"image"的下拉选择框，包含所有支持的图像文件
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},  # sorted(files)按字母排序
        }

    # 节点所属的类别，在ComfyUI界面中显示
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    # 节点的输出类型：图像、掩码、路径
    RETURN_TYPES = ("IMAGE", "MASK", "PATH")
    # 节点的主要功能函数名称
    FUNCTION = "load_image"

    # 加载图像的主要函数
    def load_image(self, image):
        # 获取图像文件的完整路径
        image_path = folder_paths.get_annotated_filepath(image)

        # 使用ComfyUI的辅助函数打开图像文件
        img = node_helpers.pillow(Image.open, image_path)

        # 初始化输出图像和掩码列表
        output_images = []
        output_masks = []
        w, h = None, None  # 用于存储图像宽度和高度

        # 排除的图像格式（这些格式不支持多帧）
        excluded_formats = ["MPO"]

        # 遍历图像的每一帧（支持动图，如GIF）
        for i in ImageSequence.Iterator(img):
            # 修复图像的方向（根据EXIF信息）
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            # 如果图像是I模式（32位整数像素），转换为0-1范围的浮点
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            # 转换为RGB模式
            image = i.convert("RGB")

            # 记录第一帧的尺寸
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            # 如果后续帧尺寸与第一帧不同，跳过该帧
            if image.size[0] != w or image.size[1] != h:
                continue

            # 将图像转换为NumPy数组，缩放为0-1范围的浮点数
            image = np.array(image).astype(np.float32) / 255.0
            # 转换为PyTorch张量，并添加批次维度
            image = torch.from_numpy(image)[None,]
            
            # 处理掩码（如果图像有alpha通道）
            if "A" in i.getbands():
                # 获取alpha通道并转换为0-1范围的浮点数
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                # 反转掩码（ComfyUI使用黑色表示透明，白色表示不透明）
                mask = 1.0 - torch.from_numpy(mask)
            else:
                # 如果没有alpha通道，创建一个全黑的掩码
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            # 将处理后的图像和掩码添加到输出列表
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        # 如果是多帧图像且不是排除的格式，将所有帧合并
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            # 否则只返回第一帧
            output_image = output_images[0]
            output_mask = output_masks[0]

        # 返回处理后的图像、掩码和图像路径
        return (output_image, output_mask, image_path)

    # 检测图像文件是否发生变化（用于缓存机制）
    @classmethod
    def IS_CHANGED(s, image):
        # 获取图像文件路径
        image_path = folder_paths.get_annotated_filepath(image)
        # 创建SHA256哈希对象
        m = hashlib.sha256()
        # 读取文件内容并更新哈希值
        with open(image_path, "rb") as f:
            m.update(f.read())
        # 返回哈希值的十六进制表示
        return m.digest().hex()

    # 验证输入是否有效
    @classmethod
    def VALIDATE_INPUTS(s, image):
        # 检查图像文件是否存在
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)  # 返回错误信息

        return True  # 输入有效


# 视频加载节点类（从ComfyUI输入目录选择）
class VideoLoader(ComfyNodeABC):
    # 定义节点的输入参数类型
    @classmethod
    def INPUT_TYPES(cls):
        # 获取ComfyUI的输入目录
        input_dir = folder_paths.get_input_directory()
        # 获取输入目录中的所有文件
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        # 筛选出视频文件
        files = folder_paths.filter_files_content_types(files, ["video"])
        # 返回输入参数定义：一个名为"file"的下拉选择框，包含所有视频文件
        return {
            "required": {"file": (sorted(files), {"video_upload": True})},
        }

    # 节点所属的类别
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    # 节点的输出类型：视频对象、路径
    RETURN_TYPES = (IO.VIDEO, "PATH")
    # 节点的主要功能函数名称
    FUNCTION = "load_video"

    # 加载视频的主要函数
    def load_video(self, file):
        # 获取视频文件的完整路径
        video_path = folder_paths.get_annotated_filepath(file)
        # 返回视频对象和视频路径
        return (InputImpl.VideoFromFile(video_path), video_path)

    # 检测视频文件是否发生变化（使用修改时间，比哈希更高效）
    @classmethod
    def IS_CHANGED(cls, file):
        # 获取视频文件路径
        video_path = folder_paths.get_annotated_filepath(file)
        # 获取文件的最后修改时间
        mod_time = os.path.getmtime(video_path)
        # 注释：使用修改时间而不是哈希文件，避免对大文件进行哈希计算
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    # 验证输入是否有效
    @classmethod
    def VALIDATE_INPUTS(cls, file):
        # 检查视频文件是否存在
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)  # 返回错误信息

        return True  # 输入有效


# 文本显示节点类
class DisplayText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "display_text"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def display_text(self, text):
        return {"ui": {"text": text}, "result": (text,)}


# 字符串保存节点类
class SaveString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "forceInput": True}),
                "filename": ("STRING", {"default": "output.txt"}),
                "append": ("BOOLEAN", {"default": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = ()
    FUNCTION = "save_string"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def save_string(self, string, filename, append):
        # 确保输入是列表格式
        if not isinstance(string, list):
            string = [string]
        if not isinstance(filename, list):
            filename = [filename] * len(string)
        if not isinstance(append, list):
            append = [append] * len(string)
        
        for s, fname, app in zip(string, filename, append):
            # 获取输出目录
            output_dir = folder_paths.get_output_directory()
            file_path = os.path.join(output_dir, fname)
            
            # 确保s是字符串，如果是列表则转换
            if isinstance(s, list):
                s = '\n'.join(str(item) for item in s)
            elif not isinstance(s, str):
                s = str(s)
            
            # 写入或追加文件
            mode = "a" if app else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                if app and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    # 如果是追加模式且文件已存在并有内容，添加分隔线
                    f.write('\n' + '='*50 + '\n')
                f.write(s)
        
        return {"ui": {"text": ["String saved successfully!"]}, "result": ()}


# 视频加载节点类（通过手动输入路径）
class VideoLoaderPath(ComfyNodeABC):
    # 定义节点的输入参数类型
    @classmethod
    def INPUT_TYPES(s):
        # 返回输入参数定义：一个名为"file"的文本输入框，用于输入视频文件路径
        return {
            "required": {
                "file": ("STRING", {"placeholder": "X://insert/path/here.mp4"}),  # 路径占位符示例
            },
        }

    # 节点所属的类别
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    # 节点的输出类型：视频对象、路径
    RETURN_TYPES = (IO.VIDEO, "PATH")
    # 节点的主要功能函数名称
    FUNCTION = "load_video"

    # 加载视频的主要函数
    def load_video(self, file):
        # 获取视频文件的完整路径
        video_path = folder_paths.get_annotated_filepath(file)
        # 返回视频对象和视频路径
        return (InputImpl.VideoFromFile(video_path), video_path)

    # 检测视频文件是否发生变化
    @classmethod
    def IS_CHANGED(cls, file):
        # 获取视频文件路径
        video_path = folder_paths.get_annotated_filepath(file)
        # 获取文件的最后修改时间
        mod_time = os.path.getmtime(video_path)
        # 注释：使用修改时间而不是哈希文件，避免对大文件进行哈希计算
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    # 验证输入是否有效
    @classmethod
    def VALIDATE_INPUTS(cls, file):
        # 检查视频文件是否存在
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)  # 返回错误信息

        return True  # 输入有效