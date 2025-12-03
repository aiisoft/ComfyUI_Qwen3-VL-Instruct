# 导入OpenCV库，用于视频处理
import cv2


# 定义一个多路径输入节点类
class MultiplePathsInput:
    @classmethod
    def INPUT_TYPES(s):
        # 定义节点的输入参数配置
        return {
            "required": {  # 必填参数
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),  # 输入路径数量
                "path_1": ("PATH",),  # 第一个路径输入
            },
            "optional": {  # 可选参数
                "sample_fps": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),  # 采样帧率
                "max_frames": (  # 最大帧数
                    "INT",
                    {"default": 2, "min": 2, "max": (1 << 63) - 1, "step": 1},
                ),
                "use_total_frames": ("BOOLEAN", {"default": True}),  # 是否使用总帧数
                "use_original_fps_as_sample_fps": ("BOOLEAN", {"default": True}),  # 是否使用原始帧率作为采样帧率
            },
        }

    RETURN_TYPES = ("PATH",)  # 返回类型：路径
    RETURN_NAMES = ("paths",)  # 返回名称：paths
    FUNCTION = "combine"  # 执行的函数名
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"  # 节点分类
    DESCRIPTION = """  # 节点描述
从多个路径创建路径批处理。
您可以通过**inputcount**参数设置节点有多少个输入，
然后点击更新按钮。
"""

    @staticmethod
    def convert_path_to_json(
        file_path,  # 文件路径
        sample_fps=1,  # 采样帧率
        max_frames=1,  # 最大帧数
        use_total_frames=True,  # 是否使用总帧数
        use_original_fps_as_sample_fps=True,  # 是否使用原始帧率作为采样帧率
    ):
        # 获取文件扩展名（小写）
        ext = file_path.split(".")[-1].lower()

        # 如果是图像文件
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
            # 返回图像类型的JSON格式
            return {"type": "image", "image": f"{file_path}"}
        # 如果是视频文件
        elif ext in ["mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"]:
            print("source_video_path:", file_path)  # 打印视频源路径
            
            # 使用OpenCV打开视频文件
            vidObj = cv2.VideoCapture(file_path)
            vr = []  # 视频帧列表
            
            # 循环读取视频帧
            while vidObj.isOpened():
                ret, frame = vidObj.read()  # 读取一帧
                if not ret:  # 如果读取失败（到视频末尾）
                    break
                else:
                    vr.append(frame)  # 将帧添加到列表
            
            # 计算视频信息
            total_frames = len(vr) + 1  # 总帧数
            print("Total frames:", total_frames)  # 打印总帧数
            
            avg_fps = vidObj.get(cv2.CAP_PROP_FPS)  # 获取平均帧率
            print("Get average FPS(frame per second):", avg_fps)  # 打印帧率
            
            duration = len(vr) / avg_fps  # 计算视频时长（秒）
            print("Total duration:", duration, "seconds")  # 打印时长
            
            # 获取视频分辨率
            width = vr[0].shape[1]  # 宽度
            height = vr[0].shape[0]  # 高度
            print("Video resolution(width x height):", width, "x", height)  # 打印分辨率
            
            vidObj.release()  # 释放视频资源
            
            # 返回视频类型的JSON格式
            return {
                "type": "video",
                "video": f"{file_path}",  # 视频路径
                "fps": avg_fps if use_original_fps_as_sample_fps else sample_fps,  # 帧率
                "max_frames": total_frames if use_total_frames else max_frames,  # 最大帧数
            }
        else:
            return None  # 如果不是支持的文件类型，返回None

    def combine(self, inputcount, **kwargs):
        # 创建一个路径列表，用于存储处理后的路径
        path_list = []
        
        # 遍历所有输入路径
        for c in range(inputcount):
            path = kwargs[f"path_{c + 1}"]  # 获取第c+1个路径

            # 过滤掉所有以"path_"开头的参数，只保留其他参数
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if not k.startswith("path_")
            }

            # 将路径转换为JSON格式
            path = self.convert_path_to_json(path, **filtered_kwargs)
            print(path)  # 打印转换后的路径
            path_list.append(path)  # 将转换后的路径添加到列表
            
        print(path_list)  # 打印所有转换后的路径列表
        result = path_list  # 结果就是处理后的路径列表
        return (result,)  # 返回结果（注意：返回的是元组）