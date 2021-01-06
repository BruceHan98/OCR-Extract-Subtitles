import os
import cv2
from tqdm import tqdm


def extract_index_frame(video_path, key_frame_index_list, top=200, bottom=100, left=50, right=50):
    """
    根据帧数索引，提取视频的指定帧并保存
    """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        h, w = frame.shape[0:2]
        # 截取字幕部分
        frame = frame[h - top:h - bottom, left:-right-1, :]
    idx = 0
    index = 0

    print('[PROCESS] start extract key frames...')

    file_name = os.path.basename(video_path).split('.')[0]
    save_dir = 'results/' + file_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 进度条
    total_frame = len(key_frame_index_list)
    pbar2 = tqdm(total_frame)

    while success:
        if idx in key_frame_index_list and idx != 0:
            save_name = "keyframe_" + str(idx) + ".jpg"
            cv2.imwrite(f"{save_dir}/" + save_name, frame)

            key_frame_index_list.remove(idx)

            index += 1
            pbar2.set_postfix_str(f'{index}/{total_frame}')
            pbar2.update(1)

        idx = idx + 1

        success, frame = cap.read()
        if success:
            h, w = frame.shape[0:2]
            frame = frame[h - top:h - bottom, left:-right-1, :]

    cap.release()
    print('[PROCESS] extract key frames complete')

    return save_dir
