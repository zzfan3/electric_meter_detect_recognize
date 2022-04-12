---

---

# 表计前向代码(2021-01-28)
## 结构
```
Meter-Recognition
    ├─checkpoints #所有模型放一起
    ├─configs #表计配置文件
    ├─images #测试图
    ├─meter
    │   ├─DANGWEI #档位表
    │   │   └─dangwei.py
    │   ├─JCQ #避雷器
    │   │   └─jcq.py
    │   ├─SF6 #六氟化硫表
    │   │   ├─sf6.py    
    │   │   └─utils.py
    │   ├─YOUWEI #油位计
    │   │   ├─utils.py
    │   │   └─youwei.py
    │   ├─YOUWEN #油温表
    │   │   ├─utils.py
    │   │   └─youwen.py
    │   ├─YOUYA #油压表
    │   │   └─youya.py
    │   ├─utils.py
    │   └─\_\_init\_\_.py
    ├─meter #通用方法
    │   ├─kp_seg1_new.py #关键点+新分割+单指针方法
    │   ├─kp_seg1_old.py #关键点+旧分割+单指针方法
    │   ├─kp_seg2_new.py #关键点+新分割+双指针方法
    │   ├─kp_seg2_old.py #关键点+旧分割+双指针方法
    │   └─utils.py
    ├─results #输出
    ├─template #模板
    ├─third_party #Pytorch网络
    │   ├─unet
    │   │   ├─model.py #网络结构
    │   │   ├─api.py #调用接口
    │   │   └─utils.py #前后处理
    │   ├─leinao_segmentation
    │   │   ├─model.py #网络结构
    │   │   └─api.py #调用接口
    │   └─hourglass
    │        ├─model.py
    │        ├─api.py
    │        └─utils.py
    ├─configure.py #读取配置
    ├─main.py #主函数
    ├─readme.md
    └─result_evaluate.py #验证读数（未使用，也未测试）
```
执行

```shell
python main.py configs/bjds_1_2SF6.ini
```



## 包含识别方法的表

#### 油温表

- 使用method里的通用方法：关键点+单指针分割+椭圆拟合

- 使用method里的通用方法：关键点+双指针分割+椭圆拟合

#### 避雷器

- 单独设计的算法

#### 六氟化硫

- 关键点+模板匹配算法（不再使用）

- 使用method里的通用方法：关键点+单指针分割+椭圆拟合

#### 油位计

- 独立设计算法

#### 档位表

- 使用method里的通用方法：关键点+单指针分割+椭圆拟合

#### 油压表

- 使用method里的通用方法：关键点+单指针分割+椭圆拟合算法