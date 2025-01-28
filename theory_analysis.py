import numpy as np
import cv2


def read_yuv_file(file_path, width, height):
    # 计算YUV文件的大小
    frame_size = width * height * 3 // 2  # YUV 4:2:0
    
    # 打开文件
    with open(file_path, 'rb') as f:
        while True:
            # 读取一帧数据
            yuv_frame = f.read(frame_size)
            if not yuv_frame:
                break  # 如果没有数据，退出循环

            # 将YUV数据转换为numpy数组
            yuv_array = np.frombuffer(yuv_frame, dtype=np.uint8)

            # 分离Y, U, V通道
            y = yuv_array[0:width * height].reshape((height, width))
            u = yuv_array[width * height:width * height + (width * height) // 4].reshape((height // 2, width // 2))
            v = yuv_array[width * height + (width * height) // 4:].reshape((height // 2, width // 2))

            # 将U和V通道上采样到Y的大小
            u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

            # 合并YUV为BGR图像
            yuv_image = cv2.merge([y, u, v])
            bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

            # 显示图像
            # print(bgr_image)
            

# 示例调用
path="/mnt/public/wangsiyuan/k8bfn0qsj9fs1rwnc2x75z6t7/BVI-HFR/60hz/"


# 读取视频文件
video_path = path+"flowers-60fps-360-1920x1080.yuv"  # 替换为你的视频文件路径
read_yuv_file(video_path, width=640, height=480)