### 视频关键字遮挡处理程序


程序将使用OpenCV处理视频，使用pytesseract进行图像文字识别，并记录已处理的文件以避免重复处理。

### 程序说明

这个视频处理程序主要功能如下：

1. **配置项**：通过修改主函数中的`INPUT_DIRECTORY`、`OUTPUT_DIRECTORY`和`KEYWORDS`来设置输入目录、输出目录和需要遮挡的关键字。

2. **视频处理流程**：
   - 遍历输入目录及其子目录中的所有视频文件
   - 对每个视频逐帧进行文字识别
   - 检测到包含关键字的区域时，用黑色方块覆盖
   - 保持原有的目录结构，将处理后的视频保存到输出目录

3. **处理记录**：
   - 已处理的视频路径会记录在`processed_files.txt`中
   - 程序再次运行时会跳过已处理的文件，避免重复工作

### 准备

运行此程序前，需要安装以下依赖库：

```bash
pip install opencv-python pytesseract pillow numpy
```

此外，还需要安装Tesseract OCR引擎：
- Windows：从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装
- Ubuntu/Debian：`sudo apt install tesseract-ocr`
- macOS：`brew install tesseract`

如果Tesseract不在系统PATH中，需要在代码中指定其路径：
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows示例
```

### Windows 系统 GPU 加速配置
要在 Windows 上启用 GPU 加速，需要：
安装支持 CUDA 的 OpenCV 版本：
pip install opencv-python opencv-contrib-python -i https://mirrors.aliyun.com/pypi/simple/
pip install ffmpeg-python -i https://mirrors.aliyun.com/pypi/simple/

或专门的 CUDA 版本（需匹配您的 CUDA 驱动）
安装 NVIDIA CUDA 工具包：
从 NVIDIA 官网下载并安装与您显卡兼容的 CUDA 工具包
确保您的 NVIDIA 显卡支持 CUDA（计算能力 3.0 及以上）
如果 GPU 加速配置正确，程序启动时会显示 "GPU 加速：已启用" 的提示。

### 注意事项

- 视频处理可能需要较长时间，取决于视频数量和长度
- 文字识别准确率受图像质量、字体、背景等因素影响
- 程序默认使用MP4格式输出，如需其他格式可修改编码器参数

可以根据实际需求调整代码中的参数，如识别置信度阈值、输出视频格式等。




# CUDA支持验证
运行文件 gpu_check可以检查是否启用GPU

### 一、检查 CUDA 工具链安装
1. **验证 CUDA Toolkit 是否安装成功**  
   打开命令提示符（CMD）或 PowerShell，输入：  
   ```bash
   nvcc --version
   ```  
   - 若显示版本信息（如 `Cuda compilation tools, release 11.7, V11.7.99`），说明 CUDA 编译器正常安装。  
   - 若提示“不是内部或外部命令”，说明 CUDA 未添加到系统环境变量，需手动添加：  
     环境变量 `PATH` 中添加 CUDA 安装路径（默认路径类似）：  
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin`  
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\libnvvp`


2. **检查 NVIDIA 驱动是否匹配**  
   CUDA Toolkit 需要对应版本的显卡驱动支持，版本不匹配会导致设备无法识别：  
   - 打开“NVIDIA 控制面板”→“帮助”→“系统信息”→“显示”，查看“驱动程序版本”。  
   - 对照 [CUDA 与驱动版本对应表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id1)，确认驱动版本 ≥ CUDA 要求的最低版本（例如 CUDA 11.7 需驱动 ≥ 471.41）。  
   - 若驱动过旧，建议通过 [GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/) 或 [NVIDIA 官网](https://www.nvidia.com/download/index.aspx) 更新至最新适配版本。


### 二、验证 CUDA 运行时是否正常
1. **运行官方检测工具**  
   CUDA 安装目录下有 `deviceQuery` 工具，用于检测显卡是否被 CUDA 识别：  
   - 路径：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\demo_suite\deviceQuery.exe`  
   - 双击运行，若输出中包含 `Result = PASS` 且显示 GTX 1060 信息，说明 CUDA 运行时正常：  
     ```
     Detected 1 CUDA Capable device(s)
     Device 0: "GeForce GTX 1060 6GB"
         CUDA Driver Version / Runtime Version          12.0 / 11.7
         CUDA Capability Major/Minor version number:    6.1
     ```  
   - 若显示 `Result = FAIL` 或无设备，说明 CUDA 驱动或工具链安装有问题，建议卸载后重新安装对应版本。


### 三、检查 OpenCV 的 CUDA 支持
1. **确认 OpenCV 编译了 CUDA 模块**  
https://blog.csdn.net/weixin_47466670/article/details/142209951
   常规 `pip install opencv-python` 安装的 OpenCV 不包含 CUDA 支持，需安装**带 CUDA 编译的版本**：  
   需要码编译 OpenCV 并启用 CUDA 选项。

2. **验证 OpenCV 的 CUDA 支持**  
   运行 Python 代码检查：  
   ```python
   import cv2
   print(cv2.cuda.getCudaEnabledDeviceCount())  # 应输出 1（若识别到 GTX 1060）
   print(cv2.getBuildInformation())  # 查看编译信息，确认 "CUDA: YES"
   ```  
   - 若 `getCudaEnabledDeviceCount()` 仍返回 0，且编译信息中 `CUDA: NO`，说明当前 OpenCV 版本未启用 CUDA，需重新安装带 CUDA 的版本。  


### 四、常见问题解决
1. **多版本 CUDA 冲突**  
   若安装过多个 CUDA 版本，需确保环境变量指向当前使用的版本，且驱动版本兼容最高版本的 CUDA。

2. **显卡未被系统识别**  
   打开“设备管理器”→“显示适配器”，若 GTX 1060 显示黄色感叹号，说明驱动安装失败，需卸载驱动后重新安装（建议使用 DDU 工具彻底清理旧驱动）。

3. **Python 与 CUDA 位数不匹配**  
   确保安装的 Python（32位/64位）与 CUDA 版本（通常为 64位）一致，64位系统建议使用 64位 Python。


