# OCR-Extract-Subtitles
使用OCR技术提取视频字幕

### 0. 介绍

​	本项目提取视频中包含字幕的关键帧，并使用 PaddleOCR 进行识别，对结果进行去重得到视频的字幕信息。

​	主要功能：

- 字幕关键帧提取
- OCR识别

### 1. 安装

​	首先，参考  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/installation.md) 项目介绍，安装环境与依赖。

​	可按如下步骤进行：

​	1. 安装PaddlePaddle Fluid v2.0

```bash
python -m pip install paddlepaddle==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用

- 运行 main.py

- 预览视频，在 main.py 中设置字幕框参数，top, bottom, left, right 分别为上下边框距最下方高度，左右边框距边缘宽度，单位为像素。按 Esc 退出程序，进行边框调整；按 c 继续程序。

- 程序依次执行：字幕帧检测 --> 字幕帧提取 --> 字幕帧识别，结果保存在 results 目录中。

  <img src="C:\Users\hanti\Pictures\2021-01-07.png" alt="2021-01-07" style="zoom:80%;" />