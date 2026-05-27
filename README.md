# Simple Gesture Recognition

基于手势识别的智能图片浏览器，使用 MediaPipe 进行手部关键点检测，结合 ONNX 模型实现手势分类，通过手势控制图片浏览。

## 演示视频

![演示视频](demo.mp4)

## 项目简介

本项目实现了一个实时手势识别系统，用户可以通过手势控制图片的浏览、缩放和滚动。系统使用摄像头捕获手部动作，通过 MediaPipe 提取 21 个手部关键点，再利用训练好的 ONNX 模型进行手势分类，最终映射为对应的 UI 操作。

## 手势类别

系统支持以下 8 种手势识别：

| 手势 | 标签 | 功能 |
|------|------|------|
| 下挥 | `down` | 向下滚动/下一张图片 |
| 上挥 | `up` | 向上滚动/上一张图片 |
| 缩放 | `zoom` | 进入缩放模式 |
| 左右挥 | `left_right` | 左右切换图片 |
| 五指张开 | `five` | 放大图片 |
| 握拳 | `fist` | 缩小图片 |
| 移动 | `move` | 平移图片 |
| 无手势 | `None` | 无操作 |

## 原理实现

### 1. 手部关键点检测

使用 **MediaPipe Hands** 进行实时手部检测，提取 21 个关键点的三维坐标（x, y, z）：

```
关键点包括：手腕、拇指、食指、中指、无名指、小指的各个关节
```

### 2. 数据预处理

- **相对坐标转换**：以手腕（关键点 0）为原点，计算相对坐标
- **标准化处理**：对坐标进行标准化，消除手部位置和大小的影响

```python
relative_coordinate = landmarks - wrist_point
standardization = (data - mean) / std
```

### 3. 手势分类模型

使用 ONNX 模型进行手势分类：
- 输入：预处理后的 21 个关键点坐标（63 维向量）
- 输出：8 种手势类别的概率分布
- 模型文件：`example_model.onnx`

### 4. 手势状态机

系统采用状态机机制，通过连续帧检测确认手势意图：

1. **触发阶段**：连续检测到相同手势 8 次以上，进入对应模式
2. **执行阶段**：根据手势类型执行相应操作
3. **退出阶段**：检测到其他手势或超时后退出当前模式

### 5. 操作映射

```python
手势操作映射：
- magnify (放大) → 图片放大
- shrink (缩小) → 图片缩小
- up_swipe/left_swipe → 上一张图片
- down_swipe/right_swipe → 下一张图片
- 拖拽手势 → 滚动视图
```

## 项目结构

```
Simple_gesture_recognition/
├── app_gesture.py      # 主应用程序（GUI界面和手势处理逻辑）
├── Infer.py            # ONNX 模型推理类
├── AisInfer.py         # 昇腾 AI 推理类（用于华为昇腾平台）
├── example_model.onnx  # 预训练手势识别模型
├── test.txt            # 示例文本内容
├── img01.jpg           # 示例图片1
├── img02.jpg           # 示例图片2
├── img03.jpg           # 示例图片3
├── img04.jpg           # 示例图片4
└── README.md           # 项目说明文档
```

## 环境要求

- Python 3.8+
- OpenCV
- MediaPipe
- ONNX Runtime
- PySide6
- NumPy

## 安装依赖

```bash
pip install opencv-python mediapipe onnxruntime PySide6 numpy
```

## 使用方法

### 1. 直接运行

```bash
python app_gesture.py
```

### 2. 仅运行推理模块

```bash
python Infer.py
```

### 3. 使用昇腾 AI 平台

```bash
python AisInfer.py
```

## 使用说明

1. 运行程序后，会打开两个窗口：
   - **MediaPipe Hands**：显示手部关键点检测结果
   - **手势操作界面**：图片浏览器和状态显示

2. 将左手放在摄像头前，系统会自动检测手部关键点

3. 使用以下手势控制图片：
   - **握拳 → 张开五指**：进入缩放模式，持续张开放大，握拳缩小
   - **向上挥动**：切换到上一张图片
   - **向下挥动**：切换到下一张图片
   - **向左/右挥动**：切换图片
   - **移动手势**：平移图片视图

4. 界面顶部显示当前模式和状态信息

5. 按 **ESC** 键退出程序

## 摄像头配置

默认使用摄像头 ID 为 2，如需修改，请编辑 `app_gesture.py` 第 136 行：

```python
self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 修改为你的摄像头ID
```

## 自定义图片

将你的图片放入项目目录，并修改 `app_gesture.py` 第 171 行：

```python
images = ['img01.jpg', 'img02.jpg', 'img03.jpg', 'img04.jpg']  # 修改为你的图片列表
```

## 模型训练

如需训练自己的手势识别模型，需要：

1. 收集手部关键点数据集
2. 标注 8 种手势类别
3. 使用深度学习框架（如 PyTorch、TensorFlow）训练模型
4. 导出为 ONNX 格式

## 常见问题

### Q: 手势识别不准确？

A: 请确保：
- 光线充足
- 手部完整出现在画面中
- 背景简洁，避免干扰

### Q: 摄像头无法打开？

A: 检查摄像头 ID 是否正确，尝试修改 `cv2.VideoCapture()` 的参数

### Q: 如何添加新的手势？

A: 需要：
1. 收集新手势的数据
2. 重新训练模型
3. 在 `IndexPredicted` 字典中添加新类别
4. 在 `GestureHandler` 中添加对应操作

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有任何问题，欢迎联系。
