import os
import xml.etree.ElementTree as ET
root="./data/yolo/label/test"
file_dir=os.listdir(root)
classes=[ '12_JCQ-3', '13_JCQ_3E', '16blq_JCQ_3E', '19_JCQ3B-Y1', '1_2_SF6',
         '23ywenj_BWY', '26_raozu', '32yweij_YZF3','38_JCQ-3', '40_BWY-804A',
         '47_bj', '55ywenj_LEAD', '56blq_JCQ_3A_20', '61_JCQ-C1', 'bj_YL_OFF',
         'bj_YL_ON', 'dangwei_1', 'JCQ-C5', 'SF6_3', 'yali_2',
         'youwei_11', 'youwei_14','youwei_13', 'youwei_2', 'youwei_3',
         'youwen_1', 'youwen_2', 'youya_1', 'youya_2', 'youya_3' ]

for file_name in file_dir:
    path=os.path.join(root,file_name)
    tree = ET.parse(path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        x=(b[1]-b[0])/w
        y=(b[3]-b[2])/h
        print(b)
