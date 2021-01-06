from analyze_key_frame import subtitle_frame
from extract_index_frame import extract_index_frame
from ocr import text_process, save_subtitle
import pdocr.infer.utility as utility
from pdocr.pyocr.utils.utility import get_image_file_list

if __name__ == '__main__':
    args = utility.parse_args()

    args.video_path = 'videos/demo0.flv'
    args.threshold = 0.2

    args.top = 35
    args.bottom = 0
    args.left = 20
    args.right = 20

    args.bg_mod = 1  # 深色背景

    print('####### 0.分析字幕关键帧 #######')
    key_frame_index = subtitle_frame(args.video_path, args.top, args.bottom, args.left, args.right, args.threshold, args.bg_mod)
    key_frame_num = len(key_frame_index)

    print('\n####### 1.提取字幕关键帧 #######')
    subtitle_path = extract_index_frame(args.video_path, key_frame_index, args.top, args.bottom, args.left, args.right)

    image_list = get_image_file_list(subtitle_path)

    print('\n####### 2.OCR识别字幕帧 #######')
    subtitle = text_process(args, image_list)

    save_subtitle(subtitle, subtitle_path)

    print('[INFO] 图片利用率：%.3f' % ((len(subtitle) / key_frame_num) * 100), '%')
