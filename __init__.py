from .nodes import Qwen3_VQA
from .util_nodes import ImageLoader, VideoLoader, VideoLoaderPath, DisplayText, SaveString
from .path_nodes import MultiplePathsInput

WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA": Qwen3_VQA,
    "ImageLoader": ImageLoader,
    "VideoLoader": VideoLoader,
    "VideoLoaderPath": VideoLoaderPath,
    "MultiplePathsInput": MultiplePathsInput,
    "DisplayText": DisplayText,
    "SaveString": SaveString,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA": "Qwen3 VQA",
    "ImageLoader": "Load Image Advanced",
    "VideoLoader": "Load Video Advanced",
    "VideoLoaderPath": "Load Video Advanced (Path)",
    "MultiplePathsInput": "Multiple Paths Input",
    "DisplayText": "Display Text",
    "SaveString": "Save String",
}
