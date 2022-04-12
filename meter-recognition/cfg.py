# 检测器标签和config.name对应关系
bjds_dict = {
    # 避雷器
    '28_JCQ-10/800': 'JCQ_8_POINT',
    '29_JCQ-10/800': 'JCQ_6_POINT',
    # 油位表
    'youwei_1': 'youwei_1',
    'youwei_9': 'youwei_9',
    'youwei_11': 'youwei_11',
    'youwei_22': 'youwei_22',
    '52_bj': '52_bj',
    # 油温表
    '26_raozu': '26_raozu',
    'youwen_7': 'youwen_7',
    # 电流电压表
    'dianliu_dianya': 'dianliu_dianya',
    # 瓦斯
    'wasi_1': 'wasi_1',
    # 油压表
    'youya_1': 'youya_1',
    # 油流继电器
    'youliu_1': 'youliu_1',
    'youliu_2': 'youliu_2',
    'youliu_3': 'youliu_3',
    # 压力表
    'yali_3': 'yali_3'
}

# 检测器标签和分割label对应关系，对应关系与生成分割一致，背景标签是0。
seg_dict = {
    # 避雷器
    '28_JCQ-10/800': {'black_pin': 1, 'plate': 2},
    '29_JCQ-10/800': {'black_pin': 1, 'plate': 2},
    # 油位表
    'youwei_1': {'65red_pin': 1},
    'youwei_9': {'red_area': 1, 'white_area': 2},
    'youwei_11': {'yellow_pin': 1},
    'youwei_22': {'red_pin': 1},
    '52_bj': {'52black_pin_head': 1, '52black_pin_tail': 2},
    # 油温表
    '26_raozu': {'26red_pin_head': 1, '26black_pin_head': 2},
    'youwen_7': {'red_pin_head': 1, 'black_pin_head': 2},
    # 电流电压表
    'dianliu_dianya': {'black_pin': 1},
    # 瓦斯
    'wasi_1': {'seg_boll': 1},
    # 油压表
    'youya_1': {'62black_pin_head': 1, '62black_pin_tail': 2},
    # 油流继电器
    'youliu_1': {'pin': 1},
    'youliu_2': {'pin': 1},
    'youliu_3': {'pin': 1},
    # 压力表
    'yali_3': {'black_pin_head': 1, 'black_pin_tail': 2}
}
