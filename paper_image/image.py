import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def hconcat_no_resize(img_list, pad_color=(255,255,255)):
    """
    水平拼接一组OpenCV图像, 不进行缩放。
    若图像高度不一致, 以其中最大高度为准, 对其它图用'pad_color'填充。
    返回拼接后的大图 (numpy.ndarray, BGR格式)。
    """
    valid_imgs = [img for img in img_list if img is not None]
    if not valid_imgs:
        return None
    
    max_h = max(img.shape[0] for img in valid_imgs)
    total_w = sum(img.shape[1] for img in valid_imgs)
    
    out = np.full((max_h, total_w, 3), pad_color, dtype=np.uint8)
    current_x = 0
    for img in valid_imgs:
        h, w, _ = img.shape
        out[0:h, current_x:current_x + w] = img
        current_x += w

    return out

def vconcat_no_resize(img_list, pad_color=(255,255,255)):
    """
    垂直拼接一组OpenCV图像, 不进行缩放。
    若图像宽度不一致, 以其中最大宽度为准, 对其它图用'pad_color'填充。
    返回拼接后的大图 (numpy.ndarray, BGR格式)。
    """
    valid_imgs = [img for img in img_list if img is not None]
    if not valid_imgs:
        return None
    
    max_w = max(img.shape[1] for img in valid_imgs)
    total_h = sum(img.shape[0] for img in valid_imgs)
    
    out = np.full((total_h, max_w, 3), pad_color, dtype=np.uint8)
    current_y = 0
    for img in valid_imgs:
        h, w, _ = img.shape
        out[current_y:current_y + h, 0:w] = img
        current_y += h

    return out

def make_text_image(text, width, height, font_scale=1.5, text_color=(0,0,0), bg_color=(255,255,255)):
    """
    制作一张大小为 (height, width) 的图, 纯色背景(bg_color)，
    并用cv2.putText在中间写上 text（BGR格式）。
    """
    # 全部转成大写
    text = text.upper()

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x = (width - text_w) // 2
    y = (height + text_h) // 2

    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    return img

def combine_big_list(
    big_list,
    sublist_titles=None,
    big_list_title=None,
    pad_color=(255,255,255),
    font_scale=1.5,
    text_height=50
):
    """
    处理一个“大列表” (big_list)：
    - big_list 有 4 个“小列表”，每个小列表含 2 张OpenCV图像；
    - 先将每个小列表的2张图水平拼接 => 得到4张图 (sublist_imgs)；
    - 再将这4张图水平拼接 => 得到 'row_img'；
    - 若 sublist_titles 不为空，则在这一行下面加一个“标题行”(对应4段文字)；
    - 若 big_list_title 不为空，则再在最下方加一个“整体标题行”；
    - 最后返回拼接后的大图 (numpy.ndarray)。
    """
    sublist_imgs = []
    for i, subl in enumerate(big_list):
        if not isinstance(subl, list) or len(subl) < 2:
            print(f"Warning: sublist {i} invalid or fewer than 2 images.")
            sublist_imgs.append(None)
            continue
        pair_img = hconcat_no_resize(subl, pad_color=pad_color)
        sublist_imgs.append(pair_img)

    row_img = hconcat_no_resize(sublist_imgs, pad_color=pad_color)
    if row_img is None:
        print("No valid images in big_list to combine.")
        return None

    # 加子标题行(若需要)
    if sublist_titles and len(sublist_titles) == len(big_list):
        total_w = row_img.shape[1]
        text_row_segments = []
        for title, subimg in zip(sublist_titles, sublist_imgs):
            if subimg is None:
                segment_w = total_w // 4
            else:
                segment_w = subimg.shape[1]
            segment = make_text_image(
                title, width=segment_w, height=text_height,
                font_scale=font_scale, text_color=(0,0,0), bg_color=(255,255,255)
            )
            text_row_segments.append(segment)
        sublist_titles_row = hconcat_no_resize(text_row_segments, pad_color=pad_color)
        row_img = vconcat_no_resize([row_img, sublist_titles_row], pad_color=pad_color)

    # 加大列表整体标题(若需要)
    if big_list_title:
        big_row_w = row_img.shape[1]
        big_title_img = make_text_image(
            big_list_title,
            width=big_row_w,
            height=text_height,
            font_scale=font_scale,
            text_color=(0,0,0),
            bg_color=(255,255,255)
        )
        row_img = vconcat_no_resize([row_img, big_title_img], pad_color=pad_color)

    return row_img

def final_composition(big_lists, sublist_titles_lists=None, big_list_titles=None):
    """
    - big_lists: [big_list_1, big_list_2, big_list_3, ...]
    - sublist_titles_lists: 对应每个 big_list 的4个小标题 (可为 None)
    - big_list_titles: 给每个 big_list 加一个整体标题 (可为 None)
    - 返回最终拼好的大图 (OpenCV BGR格式)，背景白色。
    """
    if sublist_titles_lists is None:
        sublist_titles_lists = [None]*len(big_lists)
    if big_list_titles is None:
        big_list_titles = [None]*len(big_lists)
    
    rows = []
    for i, bl in enumerate(big_lists):
        row_img = combine_big_list(
            big_list=bl,
            sublist_titles=sublist_titles_lists[i],
            big_list_title=big_list_titles[i],
            pad_color=(255,255,255),    # 白色背景
            font_scale=1.5,            # 字体大小
            text_height=50             # 标题行高度
        )
        rows.append(row_img)
    final_img = vconcat_no_resize(rows, pad_color=(255,255,255))
    return final_img

def show_in_matplotlib(img_bgr):
    """
    用Matplotlib显示OpenCV BGR格式图像 (在Notebook或脚本皆可)。
    """
    if img_bgr is None:
        print("No image to display.")
        return
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ------------- 示例调用 ------------- 
if __name__ == "__main__":
    root = "/home/hanling/HunyuanVideo_efficiency/paper_image"

    def read_cv2(path):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: failed to read {path}")
        return img

    # 构造3个大列表示例
    big_list_1 = [
        [
            read_cv2(f"{root}/catch_0_4x/bouncyball_0_30hz_8_720p_ori.png"),
            read_cv2(f"{root}/catch_0_4x/bouncyball_0_30hz_40_720p_ori.png")
        ],
        [
            read_cv2(f"{root}/catch_1_2x/catch_1_30hz_3_720p_ori.png"),
            read_cv2(f"{root}/catch_1_2x/catch_1_30hz_40_720p_ori.png")
        ],
        [
            read_cv2(f"{root}/catch_2_1x/catch_1_30hz_43_720p_ori.png"),
            read_cv2(f"{root}/catch_2_1x/catch_1_30hz_57_720p_ori.png")
        ],
        [
            read_cv2(f"{root}/catch_3_2x/catch_3_30hz_8_720p_ori.png"),
            read_cv2(f"{root}/catch_3_2x/catch_3_30hz_27_720p_ori.png")
        ],
    ]
    big_list_2 = [
        [
            read_cv2(f"{root}/catch_0_4x/bouncyball_0_30hz_8_720p_4x.png"),
            read_cv2(f"{root}/catch_0_4x/bouncyball_0_30hz_40_720p_4x.png")
        ],
        [
            read_cv2(f"{root}/catch_1_2x/catch_1_30hz_3_720p_2x.png"),
            read_cv2(f"{root}/catch_1_2x/catch_1_30hz_40_720p_2x.png")
        ],
        [
            read_cv2(f"{root}/catch_2_1x/catch_1_30hz_43_720p_1x.png"),
            read_cv2(f"{root}/catch_2_1x/catch_1_30hz_57_720p_1x.png")
        ],
        [
            read_cv2(f"{root}/catch_3_2x/catch_3_30hz_8_720p_2x.png"),
            read_cv2(f"{root}/catch_3_2x/catch_3_30hz_27_720p_2x.png")
        ],
    ]
    big_list_3 = [
        [
            read_cv2(f"{root}/catch_0_4x/bouncyball_0_30hz_8_720p_1x.png"),
            read_cv2(f"{root}/catch_0_4x/bouncyball_0_30hz_40_720p_1x.png")
        ],
        [
            read_cv2(f"{root}/catch_1_2x/catch_1_30hz_3_720p_1x.png"),
            read_cv2(f"{root}/catch_1_2x/catch_1_30hz_40_720p_1x.png")
        ],
        [
            read_cv2(f"{root}/catch_2_1x/catch_1_30hz_43_720p_1x.png"),
            read_cv2(f"{root}/catch_2_1x/catch_1_30hz_57_720p_1x.png")
        ],
        [
            read_cv2(f"{root}/catch_3_2x/catch_3_30hz_8_720p_1x.png"),
            read_cv2(f"{root}/catch_3_2x/catch_3_30hz_27_720p_1x.png")
        ],
    ]

    # 小列表的标题 (第二行演示)
    sublist_titles_1 = []
    sublist_titles_2 = ["4x", "2x", "1x", "2x"]
    sublist_titles_3 = []

    # 大列表整体标题
    big_title_1 = "Original Video"
    big_title_2 = "DLFR-VAE"
    big_title_3 = "Hunyuan-VAE"

    big_lists = [big_list_1, big_list_2, big_list_3]
    sublist_titles_lists = [sublist_titles_1, sublist_titles_2, sublist_titles_3]
    big_list_titles = [big_title_1, big_title_2, big_title_3]

    # 拼接
    final_img = final_composition(big_lists, sublist_titles_lists, big_list_titles)

    # 用matplotlib显示
    if final_img is not None:
        show_in_matplotlib(final_img)

        # 如需保存文件:
        out_path = os.path.join(root, "final_output_white.png")
        cv2.imwrite(out_path, final_img)
        print(f"Saved final image to {out_path}")
    else:
        print("No final image generated.")
