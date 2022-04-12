import os
import cv2
import time
from meter_recognition_api import Bjds2
import configure
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="meter pointer recognition inference")
    parser.add_argument("--config", type=str, default="configs/bjds_youwei_22.ini", help="config file path")
    parser.add_argument("--save", type=bool, default=True, required=False, help="whether to save result")
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()

    # load ini from argparse
    argments = parse_args()
    bjds = Bjds2(argments.config)
    file_list = os.listdir(configure.config.test_dir)
    file_list.sort()

    for filename in file_list:
        if not filename.endswith('.jpg'):
            continue
        print(filename, end="\t")
        img = cv2.imread(os.path.join(configure.config.test_dir, filename))

        result_list = bjds.process(img, filename)
        for idx, value in enumerate(result_list):
            print("Point{}: {}".format(idx, value), end="\t")
        print(end="\n")

    infer_time = time.time() - start_time
    print('time={}'.format(infer_time))



if __name__ == '__main__':
    main()
