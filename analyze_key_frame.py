import cv2
import numpy as np
from tqdm import tqdm


def subtitle_frame(video_path, top=200, bottom=100, left=50, right=50, threshold=0.4, bg_mod=1):
    key_frame_index_list = []
    index = 0

    cap = cv2.VideoCapture(video_path)
    # 视频信息
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度

    print(f'[INFO] file path: {video_path}')
    print('[INFO] total frames:', total_frame)
    print(f'[INFO] resolution: {frame_height} * {frame_width}')
    print('[INFO] fps:', fps)

    # 校正字幕范围
    cv2.namedWindow('rectify subtitle region', 0)
    cv2.resizeWindow("rectify subtitle region", 1280, 720)
    while True:
        success, frame = cap.read()
        if success:
            h, w = frame.shape[0:2]
            cv2.rectangle(frame, (left, h - top), (w - right, h - bottom), (0, 0, 255), 2)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
            # clahe = cv2.createCLAHE(3, (8, 8))
            # dst = clahe.apply(thresh)
            cv2.imshow('rectify subtitle region', frame)
            # cv2.imshow('rectify subtitle region', thresh)

            if cv2.waitKey(1) == 27:  # 退出程序
                exit(0)
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                break  # 继续执行
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    print('[PROCESS] start analyze key frames...')

    if success:
        # 截取字幕部分
        h, w = frame.shape[0:2]
        frame = frame[h - top:h - bottom, left:-right-1, :]

    h, w = frame.shape[0:2]

    if bg_mod == 1:  # 深色背景
        minuend = np.full(h * w, 200)  # 被减矩阵
    else:
        minuend = np.full(h * w, 63)  # 被减矩阵

    # 进度条
    pbar = tqdm(total_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flatten_gray = gray.flatten()
    last_roi = flatten_gray - minuend
    last_roi = np.where(last_roi > 0, 1, 0)

    while success:
        if index % 8 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flatten_gray = gray.flatten()
            roi = flatten_gray - minuend
            roi = np.where(roi > 0, 1, 0)
            # print(roi.sum())

            change = roi - last_roi

            addi = np.where(change > 0, 1, 0).sum()

            if addi > roi.sum() * 0.4:  # 字幕增加
                key_frame_index_list.append(index)

            last_roi = roi
            pbar.set_postfix_str(f'{index}/{total_frame}')
            pbar.update(8)

        index += 1
        success, frame = cap.read()
        if success:
            h, w = frame.shape[0:2]
            frame = frame[h - top:h - bottom, left:-right-1, :]

    cap.release()
    pbar.close()

    print(f'[PROCESS] [{len(key_frame_index_list)}] keyframes index stored')

    return key_frame_index_list
