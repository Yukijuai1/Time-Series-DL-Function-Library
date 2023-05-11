import torch
import pickle
import numpy as np

from timeseries.utils.sequenceInput import sequenceInput
from timeseries.trainingOptions import trainingOptions
from timeseries.trainNetwork import trainNetwork
from timeseries.utils.fullyConnectedLayer import fullyConnectedLayer
from timeseries.analyzeNetwork import analyzeNetwork
from timeseries.utils.activeFunctionLayer import activeFunctionLayer
from timeseries.RNN.lstnetLayer import lstnetLayer
from timeseries.RNN.bilstmLayer import bilstmLayer
from timeseries.RNN.gruLayer import gruLayer
from timeseries.Transformer.transformerLayer import transformerLayer
from timeseries.Transformer.layerNormalization import layerNormalization
from timeseries.CNN.tcnLayer import tcnLayer

Dataset_name=['Weibo','Twitter']
Dataset='Twitter'

Method_name=['MLP','BiLSTM','GRU','Transformer','TCN','LSTNet']
Method='BiLSTM'

#从数据文件中读取数据
if Dataset=='Weibo':
    with open('Dataset/Weibo/weibo2016_multi_train.pkl','rb') as f1:
        train_data=pickle.load(f1)
    with open('Dataset/Weibo/weibo2016_multi_test.pkl','rb') as f2:
        test_data=pickle.load(f2)
    with open('Dataset/Weibo/weibo2016_multi_val.pkl','rb') as f3:
        val_data=pickle.load(f3)
elif Dataset=='Twitter':
    with open('Dataset/Twitter/twitter_multi_train.pkl','rb') as f1:
        train_data=pickle.load(f1)
    with open('Dataset/Twitter/twitter_multi_test.pkl','rb') as f2:
        test_data=pickle.load(f2)
    with open('Dataset/Twitter/twitter_multi_val.pkl','rb') as f3:
        val_data=pickle.load(f3)

#制作输入数据和标签
Data=np.concatenate((train_data,test_data,val_data),axis=0)
data=Data[:,:15].reshape(-1,15,1)
label=Data[:,15:]
#print(data.shape,label.shape)

#归一化
data,label=sequenceInput(data,label)
#print(data.shape,label.shape)


#确定训练参数
option=trainingOptions(train_scale=0.7,val_scale=0.15,test_scale=0.15,batch_size=32,epochs=30, learning_rate=0.0001,loss_function='MAE',optimizer='adam',device='gpu',shuffle='once',verbose=True,VerboseFrequency=1,OutputNetwork='last-iteration',ValidationPatience=5)
#option.print_var()

#设计网络
if Method=='MLP':
    data=data.squeeze(-1)
    linear1=fullyConnectedLayer(data.shape[-1],128)
    func1=activeFunctionLayer('relu')
    linear2=fullyConnectedLayer(128,128)
    func2=activeFunctionLayer('relu')
    linear3=fullyConnectedLayer(128,label.shape[-1])
    layers=[linear1,func1,linear2,func2,linear3]
    analyzeNetwork(layers,input_size=[option.batch_size,data.shape[-1]])
elif Method=='BiLSTM':
    layer1=bilstmLayer(input_size=data.shape[-1],hidden_size=128,output_size=105,num_layers=5,OutputMode='last')
    layers=[layer1]
    analyzeNetwork(layers,input_size=[option.batch_size,data.shape[-2],data.shape[-1]])
elif Method=='GRU':
    layer1=gruLayer(input_size=data.shape[-1],hidden_size=128,num_layers=5,OutputMode='last')
    layer2=fullyConnectedLayer(128,label.shape[-1])
    layers=[layer1,layer2]
    analyzeNetwork(layers,input_size=[option.batch_size,data.shape[-2],data.shape[-1]])
elif Method=='Transformer':
    layer1=layerNormalization(data.shape[-1])
    layer2=transformerLayer(input_dim=data.shape[-1],output_dim=data.shape[-1],num_layers=5)
    layer3=bilstmLayer(input_size=data.shape[-1],hidden_size=128,output_size=105,num_layers=5,OutputMode='last')
    layers=[layer1,layer2,layer3]
    analyzeNetwork(layers,input_size=[option.batch_size,data.shape[-2],data.shape[-1]])
elif Method=='TCN':
    layer1=tcnLayer(input_size=data.shape[1],output_size=label.shape[-1],num_channels=[option.batch_size,128,256])
    layers=[layer1]
    analyzeNetwork(layers,input_size=[option.batch_size,data.shape[-2],data.shape[-1]])
elif Method=='LSTNet':
    layer1=lstnetLayer(input_size=data.shape[-1], channel=data.shape[1], hidden_size=128,  output_size=label.shape[-1])
    layers=[layer1]
    analyzeNetwork(layers,input_size=[option.batch_size,data.shape[-2],data.shape[-1]])

#训练网络
model=trainNetwork(data=data,label=label,layers=layers,options=option)

#进行预测
prediction=model(data)
print(prediction.shape)







