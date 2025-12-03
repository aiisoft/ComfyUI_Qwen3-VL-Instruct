# 导入必要的Python模块
import os  # 用于文件路径操作
import torch  # PyTorch深度学习框架
import folder_paths  # ComfyUI的路径管理模块
from torchvision.transforms import ToPILImage  # 用于将张量转换为PIL图像
from transformers import (
    Qwen3VLForConditionalGeneration,  # Qwen3-VL模型类
    AutoProcessor,  # 自动加载模型处理器
    BitsAndBytesConfig,  # 量化配置类
)
import comfy.model_management  # ComfyUI的模型管理模块
from qwen_vl_utils import process_vision_info  # Qwen-VL工具函数，处理视觉信息
from pathlib import Path  # 用于路径操作的高级库


# Qwen3-VL视觉问答节点类
class Qwen3_VQA:
    def __init__(self):
        """初始化Qwen3-VL节点"""
        self.model_checkpoint = None  # 模型文件路径
        self.processor = None  # 模型处理器（用于文本和图像的预处理）
        self.model = None  # 加载的模型实例
        self.device = comfy.model_management.get_torch_device()  # 获取可用的计算设备（CPU/GPU）
        # 检查是否支持bfloat16精度（需要NVIDIA Ampere架构或更新的GPU）
        self.bf16_support = (
            torch.cuda.is_available()  # 检查是否有可用的GPU
            and torch.cuda.get_device_capability(self.device)[0] >= 8  # 检查GPU架构版本
        )
        self.current_model_id = None  # 跟踪当前使用的模型ID
        self.current_quantization = None  # 跟踪当前的量化设置

    @classmethod
    def INPUT_TYPES(s):
        """定义节点的输入参数类型和默认值"""
        return {
            "required": {  # 必需输入参数
                "text": ("STRING", {"default": "", "multiline": True}),  # 用户输入的文本（支持多行）
                "model": (  # 模型选择下拉菜单
                    [
                        "Qwen3-VL-4B-Instruct-FP8",
                        "Qwen3-VL-4B-Thinking-FP8",
                        "Qwen3-VL-8B-Instruct-FP8",
                        "Qwen3-VL-8B-Thinking-FP8",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                        "Huihui-Qwen3-VL-4B-Instruct-abliterated",
                        "Huihui-Qwen3-VL-8B-Instruct-abliterated",
                    ],
                    {"default": "Qwen3-VL-4B-Instruct-FP8"},  # 默认模型
                ),
                "quantization": (  # 量化设置（减少显存使用）
                    ["none", "4bit", "8bit"],  # 无量化、4位量化、8位量化
                    {"default": "none"},  # 默认无量化
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),  # 是否保持模型加载在内存中
                "temperature": (  # 生成文本的随机性（0-1，值越大越随机）
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (  # 最大生成的新token数量
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (  # 图像最小像素数（与模型处理能力相关）
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (  # 图像最大像素数（与模型处理能力相关）
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),  # 随机种子（-1表示不固定）
                "attention": (  # 注意力机制实现方式
                    [
                        "eager",  # 普通实现
                        "sdpa",  # 扩展点积注意力
                        "flash_attention_2",  # Flash Attention 2（更高效）
                    ],
                ),
            },
            "optional": {  # 可选输入参数
                "source_path": ("PATH",),  # 图像源路径
                "image": ("IMAGE",),  # 输入图像（从ComfyUI其他节点传入）
            },
        }

    RETURN_TYPES = ("STRING",)  # 节点返回类型：字符串（生成的文本）
    FUNCTION = "inference"  # 节点主要函数名
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"  # 节点在ComfyUI中的分类

    def inference(
        self,
        text,  # 用户输入文本
        model,  # 选择的模型
        keep_model_loaded,  # 是否保持模型加载
        temperature,  # 生成文本的随机性
        max_new_tokens,  # 最大生成token数
        min_pixels,  # 图像最小像素数
        max_pixels,  # 图像最大像素数
        seed,  # 随机种子
        quantization,  # 量化设置
        source_path=None,  # 图像源路径（可选）
        image=None,  # 输入图像（可选）
        attention="eager",  # 注意力机制实现
    ):
        """执行Qwen3-VL模型推理"""
        # 设置随机种子（如果seed不等于-1）
        if seed != -1:
            torch.manual_seed(seed)
        # 根据选择的模型构建模型ID
        if model.startswith("Huihui-"):
            # 对于Huihui模型，使用fireicewolf作为仓库名
            model_id = f"fireicewolf/{model}"
        else:
            # 对于标准Qwen模型，使用qwen作为仓库名
            model_id = f"qwen/{model}"
        
        # 构建模型目录（直接使用模型名称作为目录名）
        model_repo_dir = os.path.join(
            folder_paths.models_dir,  # ComfyUI模型目录
            "prompt_generator",  # 提示生成器模型子目录
            model  # 直接使用模型名称作为目录名
        )
        
        self.model_checkpoint = model_repo_dir
        
        # 如果模型不存在，从ModelScope下载
        if not os.path.exists(self.model_checkpoint):
            from modelscope import snapshot_download  # 导入ModelScope下载工具
            import traceback  # 导入错误跟踪模块
            import shutil  # 用于文件操作
            print(f"🚀 开始从 ModelScope 下载模型: {model_id}")
            print(f"📁 下载到: {self.model_checkpoint}")
            try:
                # 创建临时目录用于下载
                temp_dir = os.path.join(folder_paths.models_dir, "prompt_generator", ".temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                # 下载模型到临时目录
                snapshot_download(
                    repo_id=model_id,  # 模型仓库ID
                    cache_dir=temp_dir,  # 临时缓存目录
                )
                
                # 将下载的模型文件移动到正确的目录
                downloaded_dir = os.path.join(temp_dir, model_id.replace("/", os.sep))
                if os.path.exists(downloaded_dir):
                    # 如果目标目录已存在，先删除它
                    if os.path.exists(self.model_checkpoint):
                        shutil.rmtree(self.model_checkpoint)
                    # 移动下载的模型目录
                    shutil.move(downloaded_dir, self.model_checkpoint)
                    # 删除临时目录
                    shutil.rmtree(temp_dir)
                
                print(f"✅ 模型 {model_id} 下载完成!")
            except Exception as e:
                print(f"❌ 模型下载失败: {str(e)}")
                print(f"📋 完整错误信息:")
                traceback.print_exc()  # 打印详细错误信息
                # 清理临时目录
                temp_dir = os.path.join(folder_paths.models_dir, "prompt_generator", ".temp")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise Exception(f"无法下载模型 {model_id}，请检查网络连接或手动下载模型到 {self.model_checkpoint}") from e

        # 如果模型ID或量化设置改变，重新加载处理器和模型
        if (
            self.current_model_id != model_id  # 检查模型是否改变
            or self.current_quantization != quantization  # 检查量化设置是否改变
            or self.processor is None  # 检查处理器是否未加载
            or self.model is None  # 检查模型是否未加载
        ):
            # 更新当前模型信息
            self.current_model_id = model_id
            self.current_quantization = quantization
            
            # 释放之前加载的资源
            if self.processor is not None:
                del self.processor  # 删除处理器
                self.processor = None
            if self.model is not None:
                del self.model  # 删除模型
                self.model = None
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清空缓存
                torch.cuda.ipc_collect()  # 收集垃圾
            # 加载模型处理器（用于文本和图像预处理）
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,  # 模型路径
                min_pixels=min_pixels,  # 图像最小像素数
                max_pixels=max_pixels,  # 图像最大像素数
            )
            # 根据选择的量化设置创建量化配置
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # 启用4位量化
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,  # 启用8位量化
                )
            else:
                quantization_config = None  # 不使用量化

            # 加载Qwen3-VL模型
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,  # 模型路径
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,  # 数据类型（优先使用bfloat16）
                device_map="auto",  # 自动分配设备（CPU/GPU）
                attn_implementation=attention,  # 注意力机制实现
                quantization_config=quantization_config,  # 量化配置
            )

        # 处理输入图像
        temp_path = None
        if image is not None:
            # 将ComfyUI的图像张量转换为PIL图像
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            # 创建临时图像文件路径
            temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
            # 保存临时图像文件
            pil_image.save(temp_path)

        # 开始推理（torch.no_grad()表示不计算梯度，节省内存）
        with torch.no_grad():
            # 根据输入类型构建消息格式
            if source_path:
                # 如果提供了图像源路径
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": source_path
                        + [
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif temp_path:
                # 如果提供了临时图像文件
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},  # 图像路径
                            {"type": "text", "text": text},  # 用户输入文本
                        ],
                    },
                ]
            else:
                # 只有文本输入
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            # 推理前准备
            # 应用聊天模板格式化文本
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 处理消息中的视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            # 预处理输入（文本和图像）
            inputs = self.processor(
                text=[text],  # 输入文本
                images=image_inputs,  # 输入图像
                videos=video_inputs,  # 输入视频（如果有）
                padding=True,  # 填充到相同长度
                return_tensors="pt",  # 返回PyTorch张量
            )
            # 将输入移动到计算设备（CPU/GPU）
            inputs = inputs.to(self.device)
            # 执行推理：生成输出
            generated_ids = self.model.generate(
                **inputs,  # 输入数据
                max_new_tokens=max_new_tokens,  # 最大生成token数
                temperature=temperature,  # 随机性参数
            )
            # 裁剪生成的token（只保留新生成的部分）
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]  # 从输入长度之后开始截取
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # 将生成的token解码为文本
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,  # 跳过特殊token
                clean_up_tokenization_spaces=False,  # 不清理空格
                temperature=temperature,
            )

            # 如果不需要保持模型加载，释放资源
            if not keep_model_loaded:
                del self.processor  # 释放处理器内存
                del self.model  # 释放模型内存
                self.processor = None  # 将处理器设置为None
                self.model = None  # 将模型设置为None
                self.current_model_id = None  # 重置当前模型ID
                self.current_quantization = None  # 重置当前量化设置
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 清空缓存
                    torch.cuda.ipc_collect()  # 收集垃圾

            # 返回生成的文本结果
            return (result,)
