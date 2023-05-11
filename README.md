<h1><div align = "center"><font size="6"><b>虚拟化技术与云计算课程设计</b></font></div></h1>
<div align = "center"><font size="5"><b>使用时间序列和序列数据进行深度学习--Demo
    </b></font></div>


[TOC]



## 一、背景介绍

​	社交媒体流行度预测是时间序列预测中的典型应用场景。研究者需要根据短期的观察，预测出某个社交媒体帖子未来的热度，从而尽快识别热点事件。这一研究将会为社交媒体平台的流量预警、舆论监控和热点追踪提供有效帮助。

​	新浪微博是近年来比较流行的社交和新闻软件。许多流行话题都是通过大量微博用户的转发引起广泛阅读，进而成为社会关注的热点事件。从这一点来看，某微博的转发情况，一定程度上可以反映这一微博的热度。类似地，国外的Twitter与微博同为社交新闻软件，两者在信息传播方面存在相似规律。

​	在本demo中，我们将从新浪微博、Twitter两个数据集出发，设计深度神经网络对它们的流行度变化趋势进行学习，并预测它们未来可能的流行度值。

## 二、数据集介绍

### 2.1 Weibo2016

​	这一数据集收集自新浪微博。它包含了从2016年6月1日到2016年7月1日的119313条微博，并记录了它们在接下来的24小时内的转发量。我们只保留上午8:00到下午6:00之间发布的微博，并过滤了总转发量小于10的帖子，以减少低质量数据产生的噪声。对于每条微博，我们以12分钟作为采样窗口，生成了120个时间步长的时间序列。

### 2.2 Twitter2011

​	这一数据集收集自Twitter，包含了从2011年10月7日到2011年11月7日的166076条推特。我们对它使用和Weibo2016相同的预处理。

处理后的两个数据集如下：

| 数据集名称 |   Weibo2016   |  Twitter2011  |
| :--------: | :-----------: | :-----------: |
| 数据集大小 | (33033,120,1) | (56505,120,1) |
| 平均转发量 |     30.81     |     52.73     |

## 三、实验设置

​	为了较全面地展示函数库的使用方法，我们分别设计了这些网络用于进行深度学习：

- MLP
- BiLSTM
- GRU
- TCN
- LSTNet

​	值得说明的是，我们的函数库也实现了卷积层、池化层、Dish-TS层、RevIN层、Dropout层等工具，但这些神经网络层往往被用于构建更复杂的网络，而不是单独作为一种训练模型，故在本Demo中没有被使用。

​	在实验中，我们将利用每条微博/推特前三小时内的转发量数据进行训练，并预测该微博/推特在未来24小时内的转发量变化。也就是说，对于时间步长为120的数据，我们以前15个时间步为输入，以后105个时间步为标签进行训练，要求模型输出后105个时间步的精确值。

​	实验使用的其他参数如下表所示：

|   参数名   |  值   |
| :--------: | :---: |
| Batch Size |  32   |
|   Epoch    |  30   |
|   学习率   | 0.001 |
|   优化器   | Adam  |
|  损失函数  |  MAE  |
| 训练集占比 |  0.7  |
| 验证集占比 | 0.15  |
| 测试集占比 | 0.15  |



## 四、实验步骤

### 4.1 数据处理

根据大多数用户习惯，我们要求输入数据必须为Numpy数组，且格式为(Size,Time Step,Variable)，分别表示数据个数、每条数据的时间步长、每条数据的变量个数；要求标签必须为Numpy数组，且格式为(Size,Time Step)。

在这一步中，我们将对输入的数据进行简单的归一化，并将它们转换为Torch.Tensor格式:

```python
data,label=sequenceInput(data,label,Normalization='rescale_zero_one')
print(data.shape,label.shape)
```

这样，data和label被转化为了tensor格式，可以用于接下来的训练中：

```python
torch.Size([33033, 15, 1]) torch.Size([33033, 105])
```

### 4.2 模型设置

在这一步中，我们将设置训练的参数和网络结构。首先，按照三中的描述，我们设置相关参数如下：

```python
option=trainingOptions(train_scale=0.7,val_scale=0.15,test_scale=0.15,
					batch_size=32,epochs=30, learning_rate=0.001,
					loss_function='MSE',optimizer='adam',
					device='cpu',shuffle='once',verbose=True,VerboseFrequency=1,
					OutputNetwork='last-iteration',ValidationPatience=5)
option.print_var()
```

可以看到参数已设置如下：

```python
--train_scale 0.7 --test_scale 0.15 --val_scale 0.15 --batch_size 32 --learning_rate 0.001 --epochs 30 --loss_function MSE --optimizer adam --device cpu --shuffle once --verbose True --VerboseFrequency 1 --OutputNetwork last-iteration --ValidationPatience 5
```

接下来，我们设置网络结构：

#### 4.2.1 MLP

​	在多层感知机中，由于任务和模型的特殊性，我们将输入数据降低一维，即输入数据格式为(Size,Time Step)。我们用三个全连接层实现一个多层感知机：

```python
	data=data.squeeze(-1)
    linear1=fullyConnectedLayer(data.shape[-1],128)
    func1=activeFunctionLayer('relu')
    linear2=fullyConnectedLayer(128,128)
    func2=activeFunctionLayer('relu')
    linear3=fullyConnectedLayer(128,label.shape[-1])
    layers=[linear1,func1,linear2,func2,linear3]
```

​	这个多层感知机由输入层、隐含层、输出层构成，其中隐藏层有128个节点，采用Relu函数作为层间的激活函数。

​	在搭建完成后，对这个网络结构进行分析：

```python
analyzeNetwork(layers,input=[option.batch_size,data.shape[-1]],device=option.device)
```

​	可以看到网络结构如下：

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Layer1      torch.Size([32, 128])            2048
              Layer2      torch.Size([32, 128])               0
              Layer3      torch.Size([32, 128])           16512
              Layer4      torch.Size([32, 128])               0
              Layer5      torch.Size([32, 105])           13545
================================================================
Total params: 32,105
Trainable params: 32,105
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.15
Params size (MB): 0.12
Estimated Total Size (MB): 0.27
----------------------------------------------------------------
```

#### 4.2.2 BiLSTM

​	使用双向LSTM进行学习，搭建模型如下：

```python
layer1=bilstmLayer(input_size=data.shape[-1],hidden_size=128,output_size=105,
				 num_layers=5,OutputMode='last')
layers=[layer1]
```

​	分析网络结构如下：

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Layer1      torch.Size([32, 105])         1742185
================================================================
Total params: 1,742,185
Trainable params: 1,742,185
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 6.65
Estimated Total Size (MB): 6.67
----------------------------------------------------------------
```

#### 4.2.3 GRU

​	使用GRU进行学习，搭建模型如下：

```python
layer1=gruLayer(input_size=data.shape[-1],hidden_size=128,num_layers=5,OutputMode='last')
layer2=fullyConnectedLayer(128,label.shape[-1])
layers=[layer1,layer2]
```

​	分析网络结构如下：

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Layer1      torch.Size([32, 128])          444672
              Layer2      torch.Size([32, 105])           13545
================================================================
Total params: 458,217
Trainable params: 458,217
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 1.75
Estimated Total Size (MB): 1.81
----------------------------------------------------------------
```

#### 4.2.4 TCN

​	使用TCN进行学习，搭建模型如下：

```python
layer1=tcnLayer(input_size=data.shape[1],output_size=label.shape[-1],
			   num_channels=[option.batch_size,128,256])
layers=[layer1]
```

分析网络结构如下：

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Layer1      torch.Size([32, 105])          448265
================================================================
Total params: 448,265
Trainable params: 448,265
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 1.71
Estimated Total Size (MB): 1.74
----------------------------------------------------------------
```

#### 4.2.5 LSTNet

使用LSTNet进行学习，搭建模型如下：

```python
layer1=lstnetLayer(input_size=data.shape[-1], channel=data.shape[1], hidden_size=128,  					 output_size=label.shape[-1])
layers=[layer1]
```

分析网络结构如下：

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Layer1      torch.Size([32, 105])          326761
================================================================
Total params: 326,761
Trainable params: 326,761
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 1.25
Estimated Total Size (MB): 1.27
----------------------------------------------------------------
```

### 4.3 训练网络

在设置完训练参数和网络结构后，就可以开始训练：

```python
model=trainNetwork(data=data,label=label,layers=layers,options=option)
```

这样模型就会按照既定的参数进行学习，并最终返回一个模型。在此过程中，`trainNetwork`函数会按照需求打印训练、验证和测试过程中的损失。

如果需要验证模型效果，可以查看这一步中最后打印的Test Loss

如果需要进行预测。可以用最终返回的模型进行计算，如：

```python
prediction=model(data)
```

## 五、实验结果

在天演平台上计算得到的六种模型的测试集上的MAE损失记录如下：

|  模型  | Weibo2016 | Twitter2011 |
| :----: | :-------: | :---------: |
|  MLP   |  1.5022   |   0.4019    |
| BiLSTM |  1.6977   |   0.4035    |
|  GRU   |  1.8630   |   0.4186    |
|  TCN   |  1.4766   |   0.4027    |
| LSTNet |  1.6755   |   0.4064    |

您可以在平台上修改项目文件`test.py`更换模型或数据集，并验证我们的结果。
