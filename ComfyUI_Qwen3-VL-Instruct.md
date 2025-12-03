# ComfyUI_Qwen3-VL-Instruct 节点模型下载路径修复与ModelScope集成总结

## 问题描述

### 1. 模型下载路径问题
在使用 ComfyUI_Qwen3-VL-Instruct 节点时，模型下载会在 `prompt_generator` 目录下创建不必要的嵌套结构，例如：
```
E:\VideoComfyUI\ComfyUI\models\prompt_generator\fireicewolf\Huihui-Qwen3-VL-4B-Instruct-abliterated
```

用户期望模型直接下载到 `prompt_generator` 目录下，不包含仓库名前缀：
```
E:\VideoComfyUI\ComfyUI\models\prompt_generator\Huihui-Qwen3-VL-4B-Instruct-abliterated
```

### 2. 模型下载网络问题
原插件使用 `huggingface_hub` 进行模型下载，但在国内网络环境下经常遇到连接不稳定、下载速度慢或下载失败的问题，严重影响用户体验。

## 解决方案

### 1. 切换到ModelScope SDK进行模型下载

**为什么选择ModelScope SDK？**
- 国内网络环境下访问稳定，下载速度快
- 提供了与huggingface_hub类似的模型下载接口
- 支持断点续传和缓存机制
- 包含丰富的中文模型资源

**相关修改：**
1. **添加ModelScope依赖**：在 `requirements.txt` 中添加了 `modelscope` 依赖

2. **集成ModelScope SDK**：在 `nodes.py` 中导入并使用了 ModelScope 的 `snapshot_download` 函数

### 2. 修改模型路径构建逻辑

修改了 `nodes.py` 文件中的模型路径构建代码，将：
```python
# 构建模型仓库目录（包含repo_id结构的路径）
model_repo_dir = os.path.join(
    folder_paths.models_dir,  # ComfyUI模型目录
    "prompt_generator",  # 提示生成器模型子目录
    model_id.replace("/", os.sep)  # 替换为系统特定的路径分隔符
)
```

改为：
```python
# 构建模型目录（直接使用模型名称作为目录名）
model_repo_dir = os.path.join(
    folder_paths.models_dir,  # ComfyUI模型目录
    "prompt_generator",  # 提示生成器模型子目录
    model  # 直接使用模型名称作为目录名
)
```

### 3. 优化模型下载流程

修改了模型下载逻辑，使用临时目录下载模型，避免路径问题：

```python
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
```

### 4. 清理目录结构

执行了以下目录清理操作：
1. 将 `fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated` 移动到 `prompt_generator/Huihui-Qwen3-VL-4B-Instruct-abliterated`
2. 删除了不必要的 `fireicewolf` 目录
3. 删除了临时目录 `._____temp`

## ModelScope模型下载详细指南

### 1. 模型下载默认存放地址
无论是使用命令行还是ModelScope SDK，模型会下载到 `~/.cache/modelscope/hub` 默认路径下。如果需要修改 cache 目录，可以手动设置环境变量：`MODELSCOPE_CACHE`，完成设置后，模型将下载到该环境变量指定的目录中。

### 2. 使用命令行工具下载

**查看帮助信息：**
```bash
modelscope download --help
```

**命令参数说明：**
```
usage: modelscope <command> [<args>] download [-h] --model MODEL [--revision REVISION] [--cache_dir CACHE_DIR] [--local_dir LOCAL_DIR] [--include [INCLUDE ...]] [--exclude [EXCLUDE ...]] [files ...]

positional arguments:
  files                 Specify relative path to the repository file(s) to download.(e.g 'tokenizer.json', 'onnx/decoder_model.onnx').

options:
  -h, --help            show this help message and exit
  --model MODEL         The model id to be downloaded.
  --revision REVISION   Revision of the model.
  --cache_dir CACHE_DIR
                        Cache directory to save model.
  --local_dir LOCAL_DIR
                        File will be downloaded to local location specified bylocal_dir, in this case, cache_dir parameter will be ignored.
  --include [INCLUDE ...]
                        Glob patterns to match files to download.Ignored if file is specified
  --exclude [EXCLUDE ...]
                        Glob patterns to exclude from files to download.Ignored if file is specified
```

**使用示例：**

- 下载整个模型repo（到默认cache地址）：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b'
  ```

- 下载整个模型repo到指定目录：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' --local_dir 'path/to/dir'
  ```

- 指定下载单个文件（以'tokenizer.json'文件为例）：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' tokenizer.json
  ```

- 指定下载多个文件：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' tokenizer.json config.json
  ```

- 指定下载某些文件（例如所有.safetensors文件）：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' --include '*.safetensors'
  ```

- 过滤指定文件（例如排除所有.safetensors文件）：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' --exclude '*.safetensors'
  ```

- 指定下载cache_dir：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' --include '*.json' --cache_dir './cache_dir'
  ```
  模型文件将被下载到'cache_dir/Qwen/Qwen2-7b'。

- 指定下载local_dir：
  ```bash
  modelscope download --model 'Qwen/Qwen2-7b' --include '*.json' --local_dir './local_dir'
  ```
  模型文件将被下载到'./local_dir'。

**参数优先级：**
如果 `cache_dir` 和 `local_dir` 参数同时被指定，`local_dir` 优先级高，`cache_dir` 将被忽略。

### 3. 使用 ModelScope SDK 下载

**下载整个模型仓库：**
```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('iic/nlp_xlmr_named-entity-recognition_viet-ecommerce-title')
```

**参数说明：**
| 字段名 | 必填 | 类型 | 描述 |
|--------|------|------|------|
| model_id | 是 | str | 模型ID |
| revision | 否 | str | 模型的Git版本，分支名或tag |
| cache_dir | 否 | str,Path | 指定模型本次下载缓存目录，给定后下载的具体模型文件将会被存储在cache_dir/model_id/THE_MODEL_FILES |
| allow_patterns | 否 | str,List | 指定要下载的文件模式，如文件名或文件扩展名 |
| ignore_patterns | 否 | str,List | 指定要忽略下载的文件模式，如文件名或文件扩展名 |
| local_dir | 否 | str | 指定模型的下载存放目录，给定后本次下载的模型文件将会被存储在local_dir/THE_MODEL_FILES |

**参数优先级：**
如果 `cache_dir` 和 `local_dir` 参数同时被指定，`local_dir` 优先级高，`cache_dir` 将被忽略。

**指定下载某些文件：**
以指定下载Qwen/QwQ-32B-GGUF中q4_k_m量化版本到path/to/local/dir目录下为例：
```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('Qwen/QwQ-32B-GGUF', allow_patterns='qwq-32b-q4_k_m.gguf', local_dir='path/to/local/dir')
```

**过滤指定文件：**
以将deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B仓库除figures子目录外的所有文件下载到指定的path/to/local/dir目录为例：
```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', ignore_patterns='figures/', local_dir='path/to/local/dir')
```

**下载模型指定文件：**
您也可以使用model_file_download下载模型指定文件：
```python
from modelscope.hub.file_download import model_file_download

model_dir = model_file_download(model_id='Qwen/QwQ-32B-GGUF', file_path='qwq-32b-q4_k_m.gguf')
```

### 4. 通过Git下载
通过Git下载模型的方式类似于从GitHub或GitLab克隆仓库，需要使用Git命令行工具。具体步骤如下：
1. 获取模型的Git仓库URL
2. 使用 `git clone` 命令克隆仓库到本地
3. 可选：使用 `git checkout` 命令切换到特定版本

### 5. 下载私有模型需要登录

**通过login命令：**
当下载私有模型时，您需要先登陆。通过 CLI 方式登陆的命令为 `modelscope login`：

```bash
modelscope login --help
```

**命令参数：**
```
usage: modelscope <command> [<args>] login [-h] --token TOKEN

options:
  -h, --help     show this help message and exit
  --token TOKEN  The Access Token for modelscope.
```

**登录示例：**
```bash
modelscope login --token YOUR_MODELSCOPE_ACCESS_TOKEN
```

您可以在 [我的访问令牌](https://modelscope.cn/my/accessToken) 页面获取访问令牌。

## 最终目录结构

```
E:\VideoComfyUI\ComfyUI\models\prompt_generator\
├── Huihui-Qwen3-VL-4B-Instruct-abliterated\
├── Huihui-Qwen3-VL-8B-Instruct-abliterated\
└── Qwen3-VL-8B-Instruct\
```

## 修改的文件

1. **`e:\VideoComfyUI\ComfyUI\custom_nodes\ComfyUI_Qwen3-VL-Instruct\nodes.py`**：
   - 集成ModelScope SDK进行模型下载
   - 修改模型路径构建逻辑
   - 优化下载流程和错误处理

2. **`e:\VideoComfyUI\ComfyUI\custom_nodes\ComfyUI_Qwen3-VL-Instruct\requirements.txt`**：
   - 添加了 `modelscope` 依赖

## 验证

修复后：
1. 新下载的模型将直接存储在 `prompt_generator` 目录下，不再包含仓库名前缀
2. 模型下载更加稳定，解决了国内网络环境下huggingface_hub下载失败的问题
3. 下载流程更加健壮，包含了完善的错误处理和临时目录清理机制

## 使用说明

对于已有模型，插件会自动识别并加载；对于新模型，插件会从ModelScope自动下载。用户可以像之前一样使用插件，无需额外配置即可享受更稳定的模型下载体验。

如果需要手动下载模型，可以参考上述的ModelScope下载指南，将模型文件放置到 `E:\VideoComfyUI\ComfyUI\models\prompt_generator\` 目录下即可。

## 辅助节点说明

本扩展还提供了两个实用的辅助节点，用于文本处理和输出：

### 1. DisplayText 节点

**功能**：在ComfyUI界面中显示文本内容，方便用户查看模型输出或其他文本信息。

**输入参数**：
- `text`：字符串类型，多行输入，强制输入参数。需要显示的文本内容。

**输出**：
- `STRING`：字符串类型（列表格式），可用于连接到其他节点作为输入。
- UI显示：在节点界面上直接显示输入的文本内容。

**特性**：
- 支持多行文本显示
- 支持列表输入（可同时处理多个文本条目）
- 既提供UI显示功能，也提供节点连接用的字符串输出

**使用场景**：
- 查看Qwen3-VL模型的文本输出结果
- 调试工作流中的文本数据
- 展示生成的提示词或描述文本

### 2. SaveString 节点

**功能**：将文本内容保存到文件中，便于持久化存储模型输出或其他重要信息。

**输入参数**：
- `string`：字符串类型，多行输入，强制输入参数。需要保存的文本内容。
- `filename`：字符串类型，默认值为"output.txt"。保存文件的名称。
- `append`：布尔类型，默认值为True。是否以追加模式写入文件（True为追加，False为覆盖）。

**输出**：
- 无（OUTPUT_NODE=True，只在UI显示操作结果）
- UI显示：保存成功的提示信息。

**特性**：
- 支持多行文本保存
- 支持列表输入（可同时处理多个文本条目）
- 自动处理不同类型的输入：
  - 如果输入是列表，会转换为换行分隔的字符串
  - 如果输入不是字符串，会自动转换为字符串格式
- 在追加模式下，会自动在新内容前添加分隔线，避免内容混乱
- 文件默认保存到ComfyUI的输出目录（`E:\VideoComfyUI\ComfyUI\output\`）

**使用场景**：
- 记录Qwen3-VL模型的文本输出结果
- 保存生成的提示词或描述文本
- 持久化存储工作流中的重要文本信息
- 累积记录多次运行的结果

**使用示例**：
1. 将Qwen3-VL模型的输出连接到DisplayText节点，实时查看结果
2. 再将DisplayText节点的输出连接到SaveString节点，将结果保存到文件
3. 设置合适的文件名和追加模式，方便后续分析和使用