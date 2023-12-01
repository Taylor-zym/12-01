标题：12.01汇报（Convolution & Residual network）

作者：曾一鸣

时间：2023年12月1日

#  12.01汇报（Convolution & Residual network）

[toc]

## 1. 数据集（data set）

### MINIST-1D

论文：Scaling down Deep Learning [(arxiv.org)](https://arxiv.org/abs/2011.14439)

博客：[Scaling down Deep Learning (greydanus.github.io)](https://greydanus.github.io/2020/12/01/scaling-down/)

> a minimalist, low-memory, and low-compute alternative to classic deep learning benchmarks.

作者将其比作生物界的果蝇。

#### 如何生成？

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302209248.png" alt="image-20231130220949208" style="zoom:80%;" />

在论文中，作者首先介绍数据集如何构建：[greydanus/mnist1d: ](https://github.com/greydanus/mnist1d)

> In order to build a synthetic dataset, we are going to pass the templates through a series of random transformations. This includes adding random amounts of padding, translation, correlated noise, iid noise, and scaling.

1. pad
2. shear
3. translate
4. corr_noise_like
5. iid_noise_like
6. interpolate

<video \centering src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302348029.mp4"></video>

Examples in training set: 4000              Examples in test set: 1000 

Length of each example: 40     				Number of classes: 10

#### 如何使用？

```python
import pickle,requests
url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
data = requests.get(url, allow_redirects=True).content
open('data.pkl', 'wb').write(data)
f = open('data.pkl','rb')
minist_1d = pickle.load(f)
```

此时返回的`minist_1d`是一个字典包含：

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302150701.png" alt="image-20231130215028824" style="zoom: 80%;" />

**在MINIST-1d上做的实验**

> Visualizing the performance of common models on the MNIST-1D dataset. This dataset separates them cleanly according to whether they use nonlinear features (logistic regression vs. MLP) or whether they have spatial inductive biases (MLP vs. CNN). Humans do best of all. Best viewed with zoom. 

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302158782.png" alt="image-20231130215846719" style="zoom:50%;" />

1. **彩票假设(lottery ticket hypothesis)：从w0开始训练一个大网络直到收敛，可以从该网络中寻找到一个子网络，保持极少的参数量，却能够以更快的速度收敛到大网络的性能。**

2. **Observing deep double descent.**

	![image-20231130221828214](https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302218283.png)

	

3. **Gradient-based metalearning**

4. **Measuring the spatial priors of deep networks.**

5. **Benchmarking pooling methods**

### 其他的常见数据集

- Fashion-MNIST
- CIFAR 10 & CIFAR 100数据集
- ImageNet数据集
- COCO数据集

## 2. 卷积网络

### **Invariance **不变性

$$
f[t[x]]=f[x]
$$

### **equivariant **等变性

$$
f[t[x]]=t[f[x]]
$$

神经网络中的一些参数：

- Padding
- Stride
- kernel size
- dilation

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None)
```

### kernel

#### 卷积如何计算？

**channel=1:**

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302225734.gif" alt="12张动图帮你看懂卷积神经网络到底是什么 | 电子创新网 Imgtec 社区" style="zoom:50%;" />

**channel$\geq2$:**

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302227533.gif" alt="tensorFlow（一）相关重要函数理解_dengtinghuan5005的博客-CSDN博客" style="zoom:50%;" />

If the kernel is size $K × K$, and there are Ci input channels, Tensors each output channel is a weighted sum of  $C_i × K × K$ quantities plus one bias. It follows that to compute Co output channels, we need  $C_i × C_o × K × K$ weights and $C_o$ biases.

在上图中，输入是$3×5×5$的张量，卷积核大小$3×3$，若输出通道为$2$，则必须卷积核权重矩阵为:$2×3×5×5$。下面再给出一个例子：

![image-20231130225747143](https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302258149.png)

***已知Input维度、卷积核大小、Stride、Padding、dilation估Output size：[^1]***

![image-20231130231016524](https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302310576.png)

[^1]:https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

#### 使用1×1卷积核.

因为使用了最小窗口，$1×1$卷积失去了卷积层的特有能力——在高度和宽度维度上，识别相邻元素间相互作用的能力。 其实$1×1$卷积的唯一计算发生在通道上。

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302258028.png" alt="image-20231130225858997" style="zoom: 80%;" />

- 多输入，多输出通道可以用来扩展卷积层的模型。
- 当以每像素为基础应用时，$1×1$卷积层相当于全连接层。
- $1×1$卷积层通常用于调整网络层的通道数量和控制模型复杂性。

*后续在Residual Network的编程实现需要用到。*

### 上采样(UnSampling) 和 下采样(DownSampling)

1. UnSampling
	1. 用stride为2的卷积层实现：卷积过程导致的图像变小是为了提取特征。下采样的过程是一个信息损失的过程，而池化层是不可学习的，用stride为2的可学习卷积层来代替pooling可以得到更好的效果，当然同时也增加了一定的计算量。
	2. 用stride为2的池化层实现：池化下采样是为了降低特征的维度。如Max-pooling和Average-pooling，目前通常使用Max-pooling，因为他计算简单而且能够更好的保留纹理特征。
2. DownSampling
	1. 插值，一般使用的是**双线性插值**，因为效果最好，虽然计算上比其他插值方式复杂，但是相对于卷积计算可以说不值一提，其他插值方式还有最近邻插值、三线性插值等；
	2. 转置卷积又或是说反卷积(Transpose Conv)，通过对输入feature map间隔填充0，再进行标准的卷积计算，可以使得输出feature map的尺寸比输入更大；相比上池化，**使用反卷积进行图像的“上采样”是可以被学习的**（会用到卷积操作，其参数是可学习的）。

### Applications

1. Image classification

2. Image segmentation

3. Object detection

  **YOLO**：*You Only Look Once*

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302324931.png" alt="YOLO_AllyLi0224的博客-CSDN博客" style="zoom: 25%;" />

​					在IOS上有一个软件：`idetection`

```python
import torch
# Model
# or yolov5n - yolov5x6, custom
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  
# url or file, Path, PIL, OpenCV, numpy, list
img = 'https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302333055.png'  
# Inference
results = model(img)
results.save()
```

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202311302335305.png" alt="image-20231130233534004" style="zoom:33%;" />

​			4.  **图像去噪、去模糊、超分辨率和去马赛克**

### 神经网络案例

#### AlexNet

```python
import torchvision.models as models
import numpy as np
from torch.utils.tensorboard import SummaryWriter
dir(models)

model = models.AlexNet()
img=torch.rand(1,3,256,256)
writer = SummaryWriter('log日志')
writer.add_graph(model, x)
writer.close()
```

Terminal运行如下指令:

```
tensorboard --logdir=log日志
```

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202312011023752.png" alt="image-20231201102347652" style="zoom:50%;" />

![image-20231201103230657](https://gitee.com/Taylor_zym/img/raw/master/imgs/202312011032714.png)

## 3. 残差网络

>  Is learning better networks as easy as stacking more layers?

从8层的AlexNet到18层的AGG，图像分类的性能确实得到很大的提升，但随着深度的继续加深，性能接着下降了。

### Why?

> Every network we have seen so far processes the data sequentially


$$
\begin{aligned}
\mathbf{h}_1 & =\mathbf{f}_1\left[\mathbf{x}, \phi_1\right] \\
\mathbf{h}_2 & =\mathbf{f}_2\left[\mathbf{h}_1, \phi_2\right] \\
\mathbf{h}_3 & =\mathbf{f}_3\left[\mathbf{h}_2, \phi_3\right] \\
\mathbf{y} & =\mathbf{f}_4\left[\mathbf{h}_3, \phi_4\right]
\end{aligned}
$$

然后我们进一步可以写成这样：

$$
\mathbf{y}=\mathbf{f}_4\left[\mathbf{f}_3\left[\mathbf{f}_2\left[\mathbf{f}_1\left[\mathbf{x}, \phi_1\right], \phi_2\right], \phi_3\right], \phi_4\right] .
$$


然而当我们逐渐加深神经网络层数，分类的表现能力又随之而下降，这是我们什么呢？一般来讲添加层数，神经网络容量更大，学习能力会增强。

> Indeed, the decrease is present for both the training set and the test set, which implies that the problem is training deeper networks rather than the inability of deeper networks to generalize.

实际上这种下降，在训练集和测试机上分类能力都下降了，所以这就意味着并非**过拟合**，而是**不收敛**。

目前可能的推测如下：

- initialization（书中）：shattered gradients

- explodingor vanishing gradients

	假设定义损失函数为$L(\hat y,y)$，链式法则：
	$$
	\frac{\partial \mathbf{y}}{\partial \mathbf{f}_1}=\frac{\partial \mathbf{f}_4}{\partial \mathbf{f}_3} \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1} .
	$$

	$$
	\frac{\partial \mathbf{L}}{\partial \mathbf{f}_1}=\frac{\partial \mathbf{L}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial \mathbf{f}_1}
	$$

### Residual Network

论文：[ Deep Residual Learning for Image Recognition (arxiv.org)](https://arxiv.org/abs/1512.03385)

<img src="https://gitee.com/Taylor_zym/img/raw/master/imgs/202312011112553.png" alt="image-20231201111242507" style="zoom:50%;" />

残差神经网络(ResNet)是由微软研究院的何恺明、张祥雨、任少卿、孙剑等人提出的。ResNet 在2015 年的ILSVRC（ImageNet Large Scale Visual Recognition Challenge）中取得了冠军。

#### Residual Block

深度神经网络是在浅神经网络再添加神经网络。所以残差神经网络的一个很简单的想法就是，添加一个恒等映射，什么也不学，效果肯定不会变差，也不会影响梯度。

于是作者提出残差学习来解决深度学习层数过高性能下降的问题。

假设该块输入$X$目标学习出$H(x)$，网络设计如下，我们可以希望其能学习到$F(x)=H(x)-x$。

之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。这有点类似与电路中的“短路”，所以是一种短路连接（*shortcut connection*）。

![image-20231201111453006](https://gitee.com/Taylor_zym/img/raw/master/imgs/202312011114040.png)

下面看看书上的一个例子：

![image-20231201115351119](https://gitee.com/Taylor_zym/img/raw/master/imgs/202312011153173.png)
$$
\begin{aligned}
\mathbf{h}_1 & =\mathbf{x}+\mathbf{f}_1\left[\mathbf{x}, \phi_1\right] \\
\mathbf{h}_2 & =\mathbf{h}_1+\mathbf{f}_2\left[\mathbf{h}_1, \phi_2\right] \\
\mathbf{h}_3 & =\mathbf{h}_2+\mathbf{f}_3\left[\mathbf{h}_2, \phi_3\right] \\
\mathbf{y} & =\mathbf{h}_3+\mathbf{f}_4\left[\mathbf{h}_3, \phi_4\right],
\end{aligned}
$$
展开写成：
$$
\begin{aligned}
\mathbf{y}=\mathbf{x} & +\mathbf{f}_1[\mathbf{x}] \\
& +\mathbf{f}_2\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]\right] \\
& +\mathbf{f}_3\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]+\mathbf{f}_2\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]\right]\right] \\
& +\mathbf{f}_4\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]+\mathbf{f}_2\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]\right]+\mathbf{f}_3\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]+\mathbf{f}_2\left[\mathbf{x}+\mathbf{f}_1[\mathbf{x}]\right]\right]\right]
\end{aligned}
$$
其中梯度：
$$
\frac{\partial \mathbf{y}}{\partial \mathbf{f}_1}=\mathbf{I}+\frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1}+\left(\frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_1}+\frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1}\right)+\left(\frac{\partial \mathbf{f}_4}{\partial \mathbf{f}_1}+\frac{\partial \mathbf{f}_4}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1}+\frac{\partial \mathbf{f}_4}{\partial \mathbf{f}_3} \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_1}+\frac{\partial \mathbf{f}_4}{\partial \mathbf{f}_3} \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1}\right)
$$
***这使得神经网络的“深度”首次突破了100层、最大的神经网络甚至超过了1000层。***





![image-20231201120004335](https://gitee.com/Taylor_zym/img/raw/master/imgs/202312011200413.png)