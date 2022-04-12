import numpy as np
import json
import os
from PIL import Image


def iou(box, clusters):
    """
   计算 IOU
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


#  计算框的 numpy 数组和 k 个簇之间的平均并集交集（IoU）。
def avg_iou(boxes, clusters):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


# 将所有框转换为原点。
def translate_boxes(boxes):
    """
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


# 使用联合上的交集（IoU）度量计算k均值聚类。
def kmeans(boxes, k, dist=np.median):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # 初始化k个聚类中心（方法是从原始数据集中随机选k个）

    while True:
        for row in range(rows):
            # 定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid)。到聚类中心的距离越小越好，但IOU值是越大越好，所以使用 1 - IOU，这样就保证距离越小，IOU值越大。
            distances[row] = 1 - iou(boxes[row], clusters)
        # 将标注框分配给“距离”最近的聚类中心（也就是这里代码就是选出（对于每一个box）距离最小的那个聚类中心）。
        nearest_clusters = np.argmin(distances, axis=1)
        # 直到聚类中心改变量为0（也就是聚类中心不变了）。
        if (last_clusters == nearest_clusters).all():
            break
        # 更新聚类中心（这里把每一个类的中位数作为新的聚类中心）
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


# 获取图片宽高
def get_image_width_high(full_image_name):
    image = Image.open(full_image_name)
    image_width, image_high = image.size[0], image.size[1]
    return image_width, image_high


# 读取 json 文件中的标注数据
def parse_label_json(label_path):
    with open(label_path, 'r') as f:
        label = json.load(f)
    result = []
    for line in label:
        bbox = line['bbox']
        x_label_min, y_label_min, x_label_max, y_label_max = bbox[0], bbox[1], bbox[2], bbox[3]
        # 计算边框的大小
        width = x_label_max - x_label_min
        height = y_label_max - y_label_min
        assert width > 0
        assert height > 0
        result.append([width, height])
    result = np.asarray(result)
    return result


# 读取 txt 标注数据文件
def parse_label_txt(label_path):
    all_label = os.listdir(label_path)
    result = []
    for i in range(len(all_label)):
        full_label_name = os.path.join(label_path, all_label[i])
        print(full_label_name)
        # 分离文件名和文件后缀
        label_name, label_extension = os.path.splitext(all_label[i])
        full_image_name = os.path.join(label_path.replace('labels', 'images'), label_name + '.jpg')
        image_width, image_high = get_image_width_high(full_image_name)
        fp = open(full_label_name, mode="r")
        lines = fp.readlines()
        for line in lines:
            array = line.split()
            x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
            x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
            y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
            y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
            # 计算边框的大小
            width = x_label_max - x_label_min
            height = y_label_max - y_label_min
            assert width > 0
            assert height > 0
            result.append([round(width, 2), round(height, 2)])
    result = np.asarray(result)

    return result


def get_kmeans(label, cluster_num=9):
    anchors = kmeans(label, cluster_num)
    ave_iou = avg_iou(label, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    # 读取 json 格式的标注数据
    # label_path = "tile_round1_train_20201231/train_annos.json"
    # label_result = parse_label_json(label_path)

    # 读取 txt 格式的标注数据
    label_path = "./data/yolo/labels/train"    # seed/images/ 内是对应图片文件
    label_result = parse_label_txt(label_path)

    anchors, ave_iou = get_kmeans(label_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print(f'anchors are: {anchor_string}')
    print(f'the average iou is: {ave_iou}')