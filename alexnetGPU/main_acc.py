from pytorchfi.errormodels import random_weight_location,random_value,bit_flip_weights,bit_flip_weight_IEEE
from pytorchfi.core import fault_injection as pfi_core
from sklearn.preprocessing import label_binarize
from  Data_load import load_Cifar10
from Alexnet import train_eval_alexnet,compute_accuracy
import copy
import  torch.nn as nn
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import csv
# torch.random.seed()
# np.random.seed()
activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook
def fault_injection(model,Conv_n,max_value,total_param=-1,probability=10**-5,seed=0):
    torch.manual_seed(seed)
    number_fault=0
    bernouli = np.random.binomial(size= int(total_param),p=probability,n=1)
    for i in range(int(total_param)):
        if bernouli[i]==1:
            print("fault tazrigh shod")
            number_fault +=1
            fault_model_object= pfi_core(model,32,32,64,use_cuda=True,c=3)
            if Conv_n>4:
                (conv,kH, kW) = random_weight_location(fault_model_object, conv=Conv_n)
                c_in=-1
                k=-1
                corrupt_idx = [kH, kW]
            else:
                (conv, k, c_in, kH, kW) = random_weight_location(fault_model_object,conv=Conv_n)
                corrupt_idx = [k, c_in, kH, kW]
            bit_position_random =  torch.randint(0,10,(1,))
            curr_layer = 0
            for name, param in model.named_parameters():
                if "weight" in name and "features" in name:
                    if curr_layer == conv:
                        corrupt_idx = (
                            tuple(corrupt_idx)
                            if isinstance(corrupt_idx, list)
                            else corrupt_idx
                        )
                        orig_value = param.data[corrupt_idx]

                    curr_layer += 1
                elif "weight" in name and "classifier" in name:
                    if curr_layer == conv:
                        corrupt_idx = (
                            tuple(corrupt_idx)
                            if isinstance(corrupt_idx, list)
                            else corrupt_idx
                        )
                        orig_value = param.data[corrupt_idx]
                    curr_layer += 1
            # faulty_val = bit_flip_weights(orig_value,max_value,bit_position_random)
            if orig_value>100000 or torch.isinf(torch.abs(orig_value)) or orig_value<-100000:
                continue
            faulty_val = bit_flip_weight_IEEE(orig_value, bit_position_random)
            fault_model = fault_model_object.declare_weight_fi(conv_num=conv, k=k, c=c_in, h=kH, w=kW, value=faulty_val)
            model = copy.deepcopy(fault_model)
        else:
            continue

    return model
def return_activation_output(name_module,test_loader,model):
    b = 0
    result=[]
    for name, module in model.named_modules():
        if name == name_module:
            module.register_forward_hook(get_activation(name))
    for data, label in test_loader:
        data = data.to('cuda')
        label = label.to('cuda')
        output = model(data)
        
        # print(output)
        # print(output)
        X = activation[name_module]

        result.append(X.cpu().detach().numpy())
        if b == 20:
            break
        b+=1
    return  result

def accuracy_vs_faultrate(fault_rate,model,test_losder,name_layer,conv_number,seed):
    for name, param in model.named_parameters():
        if "weight" in name and name_layer in name:
            total_params = param.numel()
            max_value = param.max()
    if max_value < 1:
        max_value = 1
    fault_model = fault_injection(model, conv_number, max_value, total_params, fault_rate,seed)
    acc = compute_accuracy(fault_model, test_loader, 'cuda')
    return  acc
def output_act_fault_model(fault_rate,model,name_layer,conv_number,name_activation,seed):
    for name, param in model.named_parameters():
        if "weight" in name and name_layer in name:
            total_params = param.numel()
            max_value = param.max()
    if max_value < 1:
        max_value = 1
    fault_model = fault_injection(model, conv_number, max_value, total_params, fault_rate,seed)
    result = return_activation_output(name_activation, test_loader, fault_model)
    out = np.array(result).flatten()
    out = out[out < 1E308]
    if np.max(out) > 10**30:
        out = out[out > 10 ** 38]
    n, bins, patches = plt.hist(x=out, range=(out.min(), out.max()), color='#0504aa',
                                alpha=0.7, rwidth=0.85, histtype='stepfilled')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title("conv{}:{}:{}".format(conv_number, fault_rate, out.max()))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig('conv{}_{}_{}.png'.format(conv_number, fault_rate, out.max()))
    plt.show()
    return np.max(out)
def output_act_model(model,name_activation,test_loader):
    result = return_activation_output(name_activation, test_loader, model)
    out = np.array(result).flatten()
    out = out[out < 1E308]
    return np.max(out)
def output_act_model_without_flatten(model,name_activation,test_loader):
    result = return_activation_output(name_activation, test_loader, model)
    out = np.array(result)
    out = out.reshape(result.shape[0] * result.shape[1], -1)
    return out.max(axis=0)
def auc_compute(fault_rates,conv_number,model,test_loader,layer_name,seed):
    acc_list=[]
    orig_model = copy.deepcopy(model)
    for f in fault_rates:
       acc= accuracy_vs_faultrate(f,model,test_loader,layer_name,conv_number,seed)
       acc_list.append((acc))
       model = copy.deepcopy(orig_model)
    fault_rates = np.array([0.2,0.4,0.6,0.8,1])
    acc_list = np.array(acc_list)
    # print(acc_list)
    auc = np.trapz(acc_list,fault_rates)
    # print(auc)
    return  auc
class bounded_relu(nn.Module):
    def __init__(self, bound = 1):
        super().__init__()
        self.bound = bound
    def forward(self, input):
        A= torch.min(torch.tensor(self.bound).to('cuda'),input.to('cuda'))
        return torch.max((torch.tensor(0).float()).to('cuda'),A)
def model_with_treshold(model,number_activation,T,features_class):
    for name, module in model.named_children():
        if isinstance(module,nn.Sequential):
            for name, module in module.named_children():
                if name==number_activation: # should be number
                    print(model._modules[features_class][int(number_activation)])
                    model._modules[features_class][int(number_activation)] = bounded_relu(T)

    return model

def Interval_Search(T,AUC):
    index = torch.argmax(AUC)
    if index==3:
        S = torch.tensor([T[2],T[3]])
    elif index==0:
        S = torch.tensor([T[0], T[1]])
    else:
        S = torch.tensor([T[index-1],T[index+1]])
    T_prime = T[index]
    return S,T_prime
def AUC_Calculation(S,model,fault_rates,number_activation,name_layer,conv_number,test_loader,seed):
    T1 = torch.min(S)
    T2 = T1 + (torch.max(S)-torch.min(S))/3
    T3 = T2 + (torch.max(S)-torch.min(S))/3
    T4 = torch.max(S)
    T=[T1,T2,T3,T4]
    AUC=[]
    for i in range(4):
        orig_model = copy.deepcopy(model)
        model_new = model_with_treshold(model, number_activation, T[i],name_layer.split('.')[0])
        auc = auc_compute(fault_rates, conv_number, model_new, test_loader, name_layer,seed)
        AUC.append(auc)
        model = copy.deepcopy(orig_model)
    return torch.tensor(AUC),torch.tensor(T)
def main_search_alg(model,N,M,name_activation,act_number,test_loader,fault_rates,name_layer,conv_number,delta_b,seed):
    i=1
    counter=1
    ACT_max = output_act_model(model,name_activation,test_loader)
    print("activation_max={}".format(ACT_max))
    while counter<=N:
        if i==1:
            S = torch.tensor([0,ACT_max])
            print(S)
            AUC,T = AUC_Calculation(S,model,fault_rates,act_number,name_layer,conv_number,test_loader,seed)
        else:
            S,T_result =Interval_Search(T,AUC)
            print(S)
            print(T_result)
            AUC, T = AUC_Calculation(S, model, fault_rates, act_number, name_layer, conv_number, test_loader,seed)
        counter+=1
        i+=1
        Delta=[]
        for j in range(3):
            Delta.append(torch.abs(AUC[j+1]-AUC[j]))
        Delta = torch.tensor(Delta)
        if torch.max(Delta)<=delta_b and counter>=M:
            return  T_result
    return  T_result
def model_with_bound_layer_act_max(model,name_activation,number_activation,test_loader):
    for act_name,act_number in zip(name_activation,number_activation):
            act_max = output_act_model(model, act_name, test_loader)
            model = model_with_treshold(model, act_number, act_max, act_name.split('.')[0])
    return model
def  model_with_bound_layer_act_flip(model,name_activation,number_activation,test_loader,name_layer,conv_number,
                                     fault_rates,N=3,M=2,delta_b=0.0001,seed=0):
    for act_name, act_number,name_l in zip(name_activation, number_activation,name_layer):
        act_bound = main_search_alg(model, N, M, act_name, act_number, test_loader, fault_rates, name_l, conv_number,
                                delta_b,seed)
        print("activation band ={}".format(act_bound))
        model = model_with_treshold(model, act_number, act_bound, act_name.split('.')[0])
    return model
def  model_with_bound_neuron_act_max(model,name_activation,number_activation,test_loader):
    for act_name, act_number in zip(name_activation, number_activation):
        act_max = output_act_model(model, act_name, test_loader)
        model = model_with_treshold(model, act_number, act_max, act_name.split('.')[0])
    return model
def model_with_custom_bound(model,name_activation,number_activation,test_loader,device,T):
    for act_name,act_number in zip(name_activation,number_activation):
        model = model_with_treshold(model, act_number, T, act_name.split('.')[0])
    return model 
def model_with_custom_bound_single(model,name_activation,number_activation,test_loader,device,T):
    model = model_with_treshold(model, number_activation, T, name_activation.split('.')[0])
    return model     
def model_with_fault_custom_bound(model,fault_rate,name_activation,number_activation,test_loader,device,T,conv_number):
    for name, param in model.named_parameters():
        if "weight" in name and name_layer in name:
            total_params = param.numel()
            max_value = param.max()
    if max_value < 1:
        max_value = 1
    fault_model = fault_injection(model, conv_number, max_value, total_params, fault_rate,seed) 
    for act_name,act_number in zip(name_activation,number_activation):
        model = model_with_treshold(fault_model, act_number, T, act_name.split('.')[0])
    return model         
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_Cifar10()
    model ,accuracy= train_eval_alexnet(10,train_loader,test_loader,training=False)
    ###############ACCURACY VS FAULTERATE######################
    # orig_model = copy.deepcopy(model)
    # fault_rates=[10**-7,10**-6,10**-5,10**-4,10**-3]
    # fault_rates_i = list(range(len(fault_rates)))
    # conv_number = 5
    # name_layer ='classifier.1'
    # acc=[]
    # for f in fault_rates:
    #     ac=accuracy_vs_faultrate(f,model,test_loader,name_layer,conv_number)
    #     acc.append(ac)
    #     model = copy.deepcopy(orig_model)
    # plt.plot(fault_rates_i,acc,'bo',fault_rates_i,acc,'r')
    # plt.xticks(fault_rates_i, fault_rates)
    # plt.savefig('acc_vs_fault_{}.png'.format(conv_number))
    # plt.show()

    ##########################################################################################
    ################ OUTPUT OF FAULT ACT ############################################
    # fault_rate=10**-4
    # name_layer = 'features.0'
    # conv_number=0
    # name_activation = 'features.1'
    # act_max=output_act_fault_model(fault_rate, model, name_layer, conv_number, name_activation)
    # print(act_max)
    ############################################################################
    #################### AUC VS TRESHOLD##################################
    # conv_number = 0
    # name_layer = 'features.0'
    # name_activation = 'features.1'
    # number_activation='1'
    # orig_model = copy.deepcopy(model)
    # fault_rates = [10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4]
    # x_index = list(range(1,60,5))
    # AUC_list=[]
    # for T in range(1,60,5):
    #     new_model=model_with_treshold(model, number_activation, T)
    #     auc = auc_compute(fault_rates, conv_number, new_model, test_loader, name_layer)
    #     AUC_list.append(auc)
    #     model = copy.deepcopy(orig_model)
    # AUC_list = np.array(AUC_list)
    # plt.plot(x_index,AUC_list,'bo',x_index,AUC_list,'r')
    # plt.xticks(x_index, fault_rates)
    # plt.savefig('auc_vs_tresh_{}.png'.format(conv_number))
    # plt.show()
    ######################################################################
    ################CREATE BOUNDED MODEL#################################
    # activation_name=['features.1','features.4','features.7','features.9',
    #             'features.11','classifier.2','classifier.5']
    # activation_number = ['1','4','7','9','11','2','5']
    # layer_name = ['features.0', 'features.3', 'features.6', 'features.8',
    #               'features.10', 'classifier.1', 'classifier.4']
    # fault_rates = [10 ** -7, 5 * 10 ** -6, 10 ** -5, 5 * 10 ** -4, 10 ** -3]
    # conv_number = 3
    # orig_model = copy.deepcopy(model)
    # new_model_max = model_with_bound_layer_act_max(model, activation_name,activation_number,test_loader)
    # new_model_flipact = model_with_bound_layer_act_flip(model,activation_name,activation_number,test_loader
    #                                                     ,layer_name,conv_number)
    # orig_model_bound_max = copy.deepcopy(new_model_max)
    # orig_model_bound_flipact = copy.deepcopy(new_model_flipact)
    # fault_rates_i = list(range(len(fault_rates)))
    # name_layer ='features.6'
    # acc_unbound=[]
    # acc_bound_max=[]
    # acc_bound_flip=[]
    # for f in fault_rates:
    #     ac=accuracy_vs_faultrate(f,model,test_loader,name_layer,conv_number)
    #     acm = accuracy_vs_faultrate(f, new_model_max, test_loader, name_layer, conv_number)
    #     acf = accuracy_vs_faultrate(f, new_model_flipact, test_loader, name_layer, conv_number)
    #     acc_unbound.append(ac)
    #     acc_bound_max.append(acm)
    #     acc_bound_flip.append(acf)
    #     model = copy.deepcopy(orig_model)
    #     new_model_max = copy.deepcopy(orig_model_bound_max)
    #     new_model_flipact = copy.deepcopy(orig_model_bound_flipact)
    # plt.plot(fault_rates_i,acc_unbound,'bo',fault_rates_i,acc_unbound,'r',fault_rates_i,acc_bound_max,
    #          'ro',fault_rates_i,acc_bound_max,'b',fault_rates_i,acc_bound_flip,'r-',fault_rates_i,acc_bound_flip,'b-')
    # plt.xticks(fault_rates_i, fault_rates)
    # plt.savefig('bound_vs_unbound_max_vs_unbound_flip_{}.png'.format(conv_number))
    # plt.show()
    #########################output of neuron ###########################################
    # activation_name=['features.1','features.4','features.7','features.9',
    #             'features.11','classifier.2','classifier.5']
    # activation_number = ['1','4','7','9','11','2','5']
    # model = model_with_bound_neuron_act_max(model,activation_name,activation_number,test_loader)
    # orig_model = copy.deepcopy(model)
    # fault_rates=[10**-7,10**-6,10**-5,10**-4,10**-3]
    # fault_rates_i = list(range(len(fault_rates)))
    # conv_number = 5
    # name_layer ='classifier.1'
    # acc=[]
    # for f in fault_rates:
    #     ac=accuracy_vs_faultrate(f,model,test_loader,name_layer,conv_number)
    #     acc.append(ac)
    #     model = copy.deepcopy(orig_model)
    # plt.plot(fault_rates_i,acc,'bo',fault_rates_i,acc,'r')
    # plt.xticks(fault_rates_i, fault_rates)
    # plt.savefig('acc_vs_fault_{}.png'.format(conv_number))
    # plt.show()
    ####################compare nerun and layer#########################
    # activation_name=['features.1','features.4','features.7','features.9',
    #             'features.11','classifier.2','classifier.5']
    # activation_number = ['1','4','7','9','11','2','5']
    # fault_rates = [10 ** -7, 5 * 10 ** -6, 10 ** -5, 5 * 10 ** -4, 10 ** -3]
    # conv_number = 3
    # orig_model = copy.deepcopy(model)
    # new_model_max = model_with_bound_layer_act_max(model, activation_name,activation_number,test_loader)
    # new_model_neuron_act = model_with_bound_neuron_act_max(model, activation_name,activation_number,test_loader)
    # orig_model_bound_max = copy.deepcopy(new_model_max)
    # orig_model_bound_neuron_act = copy.deepcopy(new_model_neuron_act)
    # fault_rates_i = list(range(len(fault_rates)))
    # name_layer ='features.6'
    # acc_unbound=[]
    # acc_bound_max=[]
    # acc_bound_flip=[]
    # ac=0
    # acm =0
    # acf =0
    # for f in fault_rates:
    #     for i in range(1000):
    #         ac+=accuracy_vs_faultrate(f,model,test_loader,name_layer,conv_number,i)
    #         acm+= accuracy_vs_faultrate(f, new_model_max, test_loader, name_layer, conv_number,i)
    #         acf+= accuracy_vs_faultrate(f, new_model_neuron_act, test_loader, name_layer, conv_number,i)
    #         model = copy.deepcopy(orig_model)
    #         new_model_max = copy.deepcopy(orig_model_bound_max)
    #         new_model_flipact = copy.deepcopy(orig_model_bound_neuron_act)
    #     acc_unbound.append(ac/1000)
    #     acc_bound_max.append(acm/1000)
    #     acc_bound_flip.append(acf/1000)
    #     ac=0
    #     acm=0
    #     acf=0
    # plt.plot(fault_rates_i,acc_unbound,'bo',fault_rates_i,acc_unbound,'r',fault_rates_i,acc_bound_max,
    #          'ro',fault_rates_i,acc_bound_max,'b',fault_rates_i,acc_bound_flip,'r-',fault_rates_i,acc_bound_flip,'g')
    # plt.xticks(fault_rates_i, fault_rates)
    # plt.savefig('bound_vs_unbound_max_vs_unbound_neuron_{}.png'.format(conv_number))
    # plt.show()
    ###############################accuracy vs treshhold#####################################
    # activation_name='features.1'#['features.1','features.4','features.7','features.9',
    #             #'features.11','classifier.2','classifier.5']
    # activation_number ='1'#['1','4','7','9','11','2','5']
    # layer_name ='features.0' #['features.0', 'features.3', 'features.6', 'features.8',
    # #               'features.10', 'classifier.1', 'classifier.4']
    # fault_rates = 10**-4
    # seed=1
    # conv_number = 0
    # T =range(0,30,2)
    # for name, param in model.named_parameters():
    #     if "weight" in name and layer_name in name:
    #         total_params = param.numel()
    #         max_value = param.max()
    # if max_value < 1:
    #     max_value = 1
    # print(total_params) 
    # orig_model = copy.deepcopy(model) 
    # fault_model = fault_injection(model, conv_number, max_value, total_params, fault_rates,seed) 
    # accuracy_fault= compute_accuracy(fault_model,test_loader,DEVICE)
    # model = copy.deepcopy(orig_model)
    
    # acc_bound_list = []
    # acc_original=[]
    # acc_fault=[] 
    # acc_max=0
    # for tr in T:
    #     acc_bound=0
    #     print(tr)
    #     model_bound = model_with_custom_bound_single(model,activation_name,activation_number,test_loader,DEVICE,tr)
    #     bound_model = copy.deepcopy(model_bound) 
    #     for i in range(10): 
    #         fault_model = fault_injection(model_bound, conv_number, max_value, total_params, fault_rates,seed) 
    #         acc_bound += compute_accuracy(fault_model,test_loader,DEVICE)
    #         model = copy.deepcopy(bound_model)
    #     acc_original.append(accuracy)    
    #     acc_fault.append(accuracy_fault)
    #     acc_bound_list.append(acc_bound/10)
    #     if acc_bound/10 >acc_max:
    #         print(acc_bound/10,tr)
    #         acc_max=acc_bound/10
    #     model = copy.deepcopy(orig_model)
    # plt.plot(T,acc_original,'b',T,acc_fault,'g',T,acc_bound_list,'r')    
    # plt.xticks(np.arange(0,30,2))
    # plt.yticks(np.arange(0,1,0.1))
    # plt.savefig("10_4_0.png")
########################################bound 2 #######################################################3    
    activation_name='features.1'
    activation_number ='1'
    model_bound = model_with_custom_bound_single(model,activation_name,activation_number,test_loader,DEVICE,2)
    acc_bound = compute_accuracy(model_bound,test_loader,DEVICE)
    print(acc_bound)








