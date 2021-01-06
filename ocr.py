import time
import cv2
from pdocr.predict import TextSystem


def predict_subtitle(args, image):
    text_sys = TextSystem(args)
    dt_boxes, rec_res = text_sys(image)

    drop_score = 0.5
    text_list = []
    for text, score in rec_res:
        if score >= drop_score:
            text_list.append(text)

    return text_list


def text_process(args, image_list):
    subtitle = []
    # 图片排序
    image_list.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))

    last_result = ''
    start_time = time.time()
    for image_file in image_list:
        # print(image_file)
        img = cv2.imread(image_file)
        text = predict_subtitle(args, img)
        result = ''
        last = ''

        if len(text) > 0:
            result = text[0]
            last = result
        if len(text) > 1:
            for t in text[1:]:
                result = result + ',' + t

        if last != last_result and result != '':
            subtitle.append(result)
            print(result)
        last_result = last
    total_time = time.time() - start_time

    print('\n####### 3.提取字幕结果 #######')
    for s in subtitle:
        print(s)

    print('\n[INFO] total time of reorganization:%.3f' % total_time)

    return subtitle


def save_subtitle(subtitle_list, save_path):
    with open(save_path + '/subtitle.txt', 'w', encoding='utf-8') as f:
        for i in subtitle_list:
            f.write(str(i) + '\n')
    print('[PROCESS] subtitle saved in', save_path)
