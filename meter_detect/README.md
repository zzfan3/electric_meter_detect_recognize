Power equipment detection implementation code of our paper：
If you need our dataset，Please contact us:wangsongmmd@qq.com.

In order to reduce the consumption of manpower, we collect the meter images in real substation and label the meters in these images and build a PMI (Power Meter Images) dataset containing 1800 images in real scene and a lightweight meter detection network based on Yolov5. In addition, we design an adaptive anchor which is more suitable for our dataset, and use global context module to improve the feature extraction ability of lightweight backbone without increasing the amount of calculation. FPN and PANet are used to realize the information interaction between different feature layers and achieve multi-scale prediction. Our method uses a multi-task segmented network to read the detected meter and realizes the real-time detection of electricity meter.