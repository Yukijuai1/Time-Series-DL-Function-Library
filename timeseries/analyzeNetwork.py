import torch
import numpy as np

def layer_summary(model,test_tensor,size):
    if len(size)==2:
        input_size=(size[1],)
        batch_size=size[0]
    elif len(size)==3:
        input_size=(size[1],size[2])
        batch_size=size[0]
    
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for name, layer in model.named_children():
        test_tensor = layer(test_tensor)
        var_num=sum(p.numel() for p in layer.parameters())
        line= "{:>20}  {:>25} {:>15}".format(name, str(test_tensor.shape),var_num)
        print(line)
        trainable_num = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total_params += var_num
        total_output += np.prod(test_tensor.shape)
        trainable_params += trainable_num

    
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return


def analyzeNetwork(layers,input_size):
    assert isinstance(layers,list), 'layers must be a list'
    if len(input_size)==3:
        model = torch.nn.Sequential()
        for i in range(len(layers)):
            model.add_module('Layer' + str(i+1), layers[i])
        test_tensor=torch.randn(input_size[0],input_size[1],input_size[2])
        layer_summary(model,test_tensor,input_size)
        return
    elif len(input_size)==2:
        model = torch.nn.Sequential()
        for i in range(len(layers)):
            model.add_module('Layer' + str(i+1), layers[i])
        test_tensor=torch.randn(input_size[0],input_size[1])
        layer_summary(model,test_tensor,input_size)
        return
    else:
        print('Analyze Network: input must be a list with 2 or 3 elements')
        return
