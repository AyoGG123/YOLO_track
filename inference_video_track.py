from ultralytics import YOLO
from ultralytics.solutions import object_counter
from IPython.display import display, Image
from pytube import YouTube
from PIL import Image
import os
import cv2
import sys
import torch
import numpy as np  # 导入numpy库

HOME = os.getcwd()
print(HOME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if sys.platform == 'darwin': device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

width = 0
height = 0
region_points_1 = [(100, height - 100),
                   (width - 100, height)]
region_points_2 = [(100, height - 100),
                   (width - 100, height)]
half_width = 0
half_height = 0
half_region_points_1 = [(100, half_height - 100),
                        (width - 100, half_height)]
half_region_points_2 = [(100, half_height - 100),
                        (width - 100, half_height)]

passing_pointa_set = {}

# Load a model
# model_name = os.path.join(HOME, 'yolo', 'PT', 'yolov8m.pt')
# model_name = os.path.join(HOME, 'yolo', 'yolov9', 'yolov9c.pt')
model_name = os.path.join(HOME, 'yolo', 'yolov10', 'yolov10m.pt')

model = YOLO(model=model_name)  # load an official model
model.to(device, non_blocking=True)

# file_test = ['test/images/秉逸ppap.mp4']
file_test = ['test/images/【 Archive 】2024-04-03 『0600～』 東京 新宿 歌舞伎町 ライブ_２.mp4']
# file_test = ['test/images/【 Archive 】2024-09-16『2100～』 東京 新宿 歌舞伎町 ライブ_２.mp4']

font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
fontScale = 1  # 字體大小
bigger_fontScale = 2  # 字體大小
red_color = (40, 40, 255)
blue_color = (255, 40, 40)
thickness = 2  # 線條的厚度
lineType = cv2.LINE_AA  # 抗鋸齒

# file_test = ['test/images/2024-04-02 13-58-24.mkv']

count_num = 0
id_record_in_region1 = set()


def is_point_in_region(x, y, x1, y1, x2, y2):
    """
    檢測點 (x, y) 是否在由 (x1, y1) 和 (x2, y2) 定義的矩形內部。

    參數:
    x, y: 點的座標。
    x1, y1: 矩形左下角的座標。
    x2, y2: 矩形右上角的座標。

    返回:
    True 如果點在矩形內部，否則 False。
    """
    return x1 < x < x2 and y1 < y < y2


def draw_boxes(frame, boxes_xyxy, conf, cls, count=True):
    global width, height, half_width, half_height, count_num, region_points_1, region_points_2, half_region_points_1, half_region_points_2, id_record_in_region1
    joint_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    """
    绘制关键点
    :param frame: 视频帧
    :param keypoint_xy: 关键点的(x, y)坐标列表
    """
    xy_center = []
    if boxes_xyxy.shape == (0, 4):
        return

    # 确保关键点坐标是整数
    boxes_xyxy = (boxes_xyxy + np.array([0, half_height, 0, half_height])).astype(int)
    cls = (cls).astype(int)

    cv2.putText(frame, f'count:{count_num}', (half_width, 50), font, bigger_fontScale, blue_color,
                thickness,
                lineType)  # count更新
    cv2.rectangle(frame, region_points_1[0], region_points_1[1], blue_color, thickness)  # 比較上面的框框
    cv2.rectangle(frame, region_points_2[0], region_points_2[1], blue_color, thickness)  # 比較下面的框框

    # for i, (xy, score, label) in enumerate(zip(boxes_xyxy, conf, cls)):
    #     # if label != 0: continue  # 只標注人 #classes中已指定了
    #     # cv2.putText(frame, f'id:{id_count} person {score:.4f}', xy[0], font, fontScale, blue_color, thickness, lineType)
    #     cv2.putText(frame, f'id:{i} person', xy[0], font, fontScale, blue_color, thickness, lineType)
    #
    #     for joint_pair in joint_pairs:
    #         start_point = xy[joint_pair[0]]
    #         end_point = xy[joint_pair[1]]
    #
    #         cv2.line(frame, start_point, end_point, red_color, 2)  # 绘制连线

    for i, (xy, score, label) in enumerate(zip(boxes_xyxy, conf, cls)):
        # if label != 0: continue  # 只標注人 #classes中已指定了
        # cv2.putText(frame, f'id:{id_count} person {score:.4f}', xy[0], font, fontScale, blue_color, thickness, lineType)
        x = int((xy[0] + xy[2]) / 2)  # 方框的中間點
        y = int(xy[1])
        xy_center.append((x, y))
        if is_point_in_region(x, y, region_points_1[0][0], region_points_1[0][1], region_points_1[1][0],  # 檢測是否在region1
                              region_points_1[1][1]):
            id_record_in_region1.add(i)

        if is_point_in_region(x, y, region_points_2[0][0], region_points_2[0][1], region_points_2[1][0],  # 檢測是否在region2
                              region_points_2[1][1]):
            if i in id_record_in_region1:
                id_record_in_region1.remove(i)
                count_num += 1

        cv2.putText(frame, f'id:{i} person', (xy[0], xy[1]), font, fontScale, blue_color, thickness, lineType)  # 頭上標註id
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # 頭上標註綠點點


def inference_video_track(file_list, count=True):
    global width, height, half_width, half_height, count_num, region_points_1, region_points_2, half_region_points_1, half_region_points_2
    for file in file_list:
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            print(f"无法打开视频文件: {file}")
            continue

        # 获取视频的原始帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        half_width = int(width / 2)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        half_height = int(height / 2)

        x_offset = 100
        y_offset_1 = 250
        y_offset_2 = 7

        region_points_1 = [(x_offset, half_height + y_offset_1),
                           (width - x_offset + 100, half_height + y_offset_1 + y_offset_2)]  # 1在比較上面
        half_region_points_1 = [(x_offset, y_offset_1),
                                (width - x_offset + 100, y_offset_1 + y_offset_2)]
        region_points_2 = [(x_offset, height - y_offset_1 - y_offset_2),
                           (width - x_offset + 100, height - y_offset_1)]
        half_region_points_2 = [(x_offset, half_height - y_offset_1 - y_offset_2),  # 2在比較下面
                                (width - x_offset + 100, half_height - y_offset_1)]

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束
            frame_lower_half = frame[half_height:, :]

            with torch.no_grad():
                # 这里调用模型预测方法，获取当前帧的关键点
                results = model.track(source=frame_lower_half,
                                      vid_stride=1,  # 禎率
                                      imgsz=1600,  # 丟進模型的大小
                                      conf=0.5,  # >0.5才過
                                      device=device,
                                      show_boxes=False,
                                      show_conf=False,
                                      show_labels=False,
                                      show=False,
                                      half=True,  # 半精度
                                      visualize=False,
                                      int8=False,
                                      classes=[0],  # 如果只有一個result.boxes.id永遠都是none
                                      persist=True,  # persist=True 時，這通常意味著追蹤器會在連續的幀之間保持或「記住」追蹤的目標
                                      tracker="bytetrack.yaml")

            result = results[0]
            # frame = result.plot()

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()

            # if boxes_xyxy.shape == (0, 4):
            #     updated_boxes = boxes_xyxy
            # else:
            #     boxes_xyxy = (boxes_xyxy + np.array([0, half_height, 0, half_height]))
            #     # 初始化一個新的列表來保存更新後的座標
            #     updated_boxes = np.empty((len(boxes_xyxy), 4, 2), dtype=np.float32)
            #     for i, xy in enumerate(boxes_xyxy):
            #         # 計算右上和左下的座標
            #         right_top = np.array([xy[2], xy[1]])  # [右下x, 左上y]
            #         left_bottom = np.array([xy[0], xy[3]])  # [左上x, 右下y]
            #
            #         # 組合成完整的座標順序：左上, 右上, 右下, 左下
            #         xy = np.concatenate((xy, right_top, left_bottom))
            #
            #         # 重新排序以符合指定的順序：左上, 右上, 右下, 左下
            #         xy = xy[[0, 1, 4, 5, 2, 3, 6, 7]]
            #         xy = xy.reshape(4, 2)  # 將平坦的座標列表轉換為四個點的座標
            #         # 更新到新的數據結構中
            #         updated_boxes[i] = xy

            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            # 绘制关键点

            draw_boxes(frame, boxes_xyxy, conf, cls, True)

            resized_image = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)

            # 显示帧
            cv2.imshow('Frame with Keypoints', resized_image)

            # 检测按键事件
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):  # 偵測 'q' 鍵
                break
            elif key == 2555904:  # 右箭头键
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 100)
            elif key == 2424832:  # 左箭头键
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 100))

        cap.release()

    cv2.destroyAllWindows()


def yt_stream(url, count=True):
    # where to save
    SAVE_PATH = "test/images/"  # to_do

    try:
        # object creation using YouTube
        yt = YouTube(url)
    except:
        # to handle exception
        print("Connection Error")

        # Get all streams and filter for mp4 files
    mp4_streams = yt.streams.filter(file_extension='mp4', type='video').all()

    # get the video with the highest resolution
    d_video = mp4_streams[2]

    try:
        # downloading the video
        d_video.download(output_path=SAVE_PATH)
        print('Video downloaded successfully!')
        inference_video_track([f'{SAVE_PATH}{d_video.default_filename}'])
    except:
        print("Some Error!")


# keypoints”: [“nose”,“left_eye”,“right_eye”,“left_ear”,“right_ear”,
# “left_shoulder”,“right_shoulder”,“left_elbow”,“right_elbow”,“left_wrist”,
# “right_wrist”,“left_hip”,“right_hip”,“left_knee”,“right_knee”,“left_ankle”,“right_ankle”]

# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
#         10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
#         20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
#         30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
#         50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#         60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
#         70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

if __name__ == "__main__":
    inference_video_track(file_test)
    # yt_stream(
    #     'https://www.youtube.com/live/u6vXVo5qRHs?si=CAfPVC7X9qvsMT7P')  # 【 Archive 】2024-04-03 『0600～』 東京 新宿 歌舞伎町 ライブ_２
    # yt_stream(
    #     'https://youtu.be/O_T7PlgAh3M?si=n_lVq4EU_gO2y12Z')  # 【純享版】寧藝卓《Melody》 丨#舞台2023 The Next EP4丨#TheNext
    # yt_stream(
    #     'https://www.youtube.com/live/r11U7PIkMTs?si=YdFs19B4TvwT0WWA')  # 【 Archive 】2024-09-16『21:00～』 東京 新宿 歌舞伎町 ライブ_２
