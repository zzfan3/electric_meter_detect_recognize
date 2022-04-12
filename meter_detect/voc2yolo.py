import xml.etree.ElementTree as ET
import pickle
import os
import shutil
from os import listdir, getcwd
from os.path import join

classes = ['head']

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(rootpath,xmlname):
    xmlpath = rootpath + '/Annotations_val'
    xmlfile = os.path.join(xmlpath,xmlname)
    with open(xmlfile, "r") as in_file:
      txtname = xmlname[:-4]+'.txt'
      txtpath = rootpath + '/labelYOLOs'
      if not os.path.exists(txtpath):
        os.makedirs(txtpath)
      txtfile = os.path.join(txtpath,txtname)
      with open(txtfile, "w+") as out_file:
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file.truncate()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":

    rootpath='D:\code\data\\brainwash'
    xmlpath=rootpath+'/brainwash_train.idl'
    txt_files_dir = "./data/brainwash/labels/train"
    to_dir = "./data/brainwash/images/train/"
    to_dir2 = "./data/brainwash/images/train2/"
    # for i in os.listdir(to_dir2):
    #     os.rename(to_dir2 + i, to_dir + i.replace(".", "_2."))
    #"brainwash_11_13_2014_images/00001000_640x480.png": (63.0, 260.0, 89.0, 287.0), (115.0, 174.0, 135.0, 193.0);
    with open(xmlpath,"r") as f:
        list=[item.strip() for item in f.readlines()]
    print(len(list))
    for item in list:
        line = item.replace(":",";")
        img_dir = line.split(";")[0]
        # print(img_dir)
        img_boxs = line.split(";")[1]
        img_dir = img_dir.replace('"', "")  # 删除分号
        # print(img_dir)
        img_path = os.path.join(rootpath,img_dir)
        img_name = img_dir.split("/")[1]
        txt_name = img_name.split(".")[0]  # 得到后缀名与文件名
        img_extension = img_name.split(".")[1]
        # if not os.path.exists(os.path.join(to_dir,img_name.replace(".",img_extension+"."))):
        #     shutil.copy(img_path,to_dir)
        #     os.rename(to_dir+img_name,to_dir+img_name.replace(".",img_extension+"."))
        # else:
        #     shutil.copy(img_path,to_dir2)
        #     os.rename(to_dir2 + img_name, to_dir2 + img_name.replace(".", img_extension + "."))
            #os.rename(to_dir2 + img_name,"img_name")


        img_boxs = img_boxs.replace(",", "")  # 删除“，”
        # print(img_boxs)
        img_boxs = img_boxs.replace("(", "")  # 删除“(”
        img_boxs = img_boxs.split(")")  # 删除“)”

        if (img_extension == 'jpg' or img_extension == 'png' ):
            txt_name=txt_name+img_extension
            if os.path.exists(txt_files_dir + "/" + txt_name + ".txt"):
                print("double"+txt_name)
                txt_name=txt_name+"_2"
            if len(img_boxs)<=1:
                with open(txt_files_dir + "/" + txt_name + ".txt", 'a') as f:
                    print("no head"+txt_name)
                    f.write('\n')
            else:
                for n in range(len(img_boxs) - 1):  # 消除空格项影响
                    box = img_boxs[n]
                    box = box.split(" ")
                    with open(txt_files_dir + "/" + txt_name + ".txt", 'a') as f:
                        f.write(' '.join(['0', str((float(box[1]) + float(box[3])) / (2 * 640)),
                                          str((float(box[2]) + float(box[4])) / (2 * 480)),
                                          str((float(box[3]) - float(box[1])) / 640),
                                          str((float(box[4]) - float(box[2])) / 480)]) + '\n')
