import cv2


def letter_box_image(img, w, h, value):
    # 将图像resize到w*h，保持原图像的长宽比，填充像素的像素值为value
    img_h, img_w = img.shape[0], img.shape[1]
    dim = max(img_h, img_w)
    ratio = w / dim
    resize_h = int((img_h / dim) * w)
    resize_w = int((img_w / dim) * h)
    if ((w - resize_w) % 2):
        resize_w -= 1
    if ((h - resize_h) % 2):
        resize_h -= 1
    offset_w = int((w - resize_w) / 2)
    offset_h = int((h - resize_h) / 2)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
    resize_img = cv2.copyMakeBorder(img, offset_h, offset_h, offset_w, offset_w, cv2.BORDER_CONSTANT, value=value)
    # return resize_img, ratio, offset_w, offset_h
    return resize_img
