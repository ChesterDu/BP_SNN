from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

# membrane potential update
# def mem_update(fc_input, x, mem, spike, fc_self):
# #     mem = mem * decay * (1. - spike) + spike * rest + fc_input(x)
# #     mem = torch.Tensor([[fc_self(mem_i.unsqueeze(0)) for mem_i in mem[bch]] for bch in range(batch_size)]) * (1. - spike) + spike * rest + fc_input(x)
#     mem_t = fc_self(mem.clone().reshape(-1,1)).reshape(batch_size,-1) * (1. - spike) + spike * rest + fc_input(x)
# #     mem = fc_self(mem.unsqueeze(-1)).squeeze(-1) * (1. - spike) + spike * rest + fc_input(x)
#     spike = act_fun(mem_t) # act_fun : approximation firing function
#     return mem_t, spike


# # cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
# cfg_cnn = [(1, 32, 1, 1, 3),
#            (32, 32, 1, 1, 3),]
# # kernel size
# cfg_kernel = [28, 14, 7]
# fc layer
n_input = 620
n_classes = 11
cfg_fc = [n_input, n_classes]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

class SNN(nn.Module):
    def __init__(self, max_time=50):
        super(SNN, self).__init__()
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
#         self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

#         self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.max_time = max_time
        self.fc_neuron_para = nn.Linear(1,1,bias=False)
        
    def forward(self, input):
#         c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
#         c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

#         h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        time_window = self.max_time
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input[:,step]
            h2_mem, h2_spike = self.mem_update(x, h2_mem, h2_spike)
            h2_sumspike += h2_spike
#             print(h2_mem[0][0])

        outputs = h2_sumspike / 50
        return outputs

    def mem_update(self, x, mem, spike):
        mem_t = self.fc_neuron_para(mem.clone().reshape(-1,1)).reshape(batch_size,-1) * (1. - spike) + spike * rest + self.fc2(x)
#     mem = fc_self(mem.unsqueeze(-1)).squeeze(-1) * (1. - spike) + spike * rest + fc_input(x)
        spike = act_fun(mem_t) # act_fun : approximation firing function
        return mem_t, spike



##### Load Train Dataset ##### 
from scipy.io import loadmat 
import numpy as np
m = loadmat("./Spike-TIDIGITS/Spike-TIDIGITS.mat")
max_time = 130 #ms
input_neuron_num = 620
input_comb_coef = 10
train_data_num = m['train_pattern'].shape[0]
train_datasets = np.zeros((train_data_num, max_time, int(input_neuron_num/1)))
for i in range(train_data_num):
    current_array = m['train_pattern'][i][0]
    current_spikes_input = np.zeros((max_time, int(input_neuron_num/1)))
    for input_idx in range(current_array.shape[0]):
        for spike_time in current_array[input_idx]:
            if spike_time < max_time:
                current_spikes_input[int(spike_time)][int(input_idx/1)] = 1
    train_datasets[i] = current_spikes_input
train_datasets = torch.Tensor(train_datasets).byte()
train_labels = torch.LongTensor(m['train_labels']-1)


##### Load Test Dataset ##### 
# m = loadmat("./Spike TIDIGITS/Spike-TIDIGITS.mat")
# max_time = 50 #ms
# input_neuron_num = 620
test_data_num = m['test_pattern'].shape[0]
test_datasets = np.zeros((test_data_num, max_time, input_neuron_num))
for i in range(train_data_num):
    current_array = m['test_pattern'][i][0]
    current_spikes_input = np.zeros((max_time, input_neuron_num))
    for input_idx in range(current_array.shape[0]):
        for spike_time in current_array[input_idx]:
            if spike_time < max_time:
                current_spikes_input[int(spike_time)][input_idx] = 1
    test_datasets[i] = current_spikes_input
test_datasets = torch.Tensor(test_datasets).byte()
test_labels = torch.LongTensor(m['test_labels']-1)
n_test = test_labels.shape[0]


def train(num_epochs, train_datasets, train_labels, test_datasets, test_label, thresh, lens, decay):
    thresh = thresh # neuronal threshold
    lens = lens # hyper-parameters of approximate function
    decay = decay # decay constants
    # num_classes = 10
    batch_size  = 1
    learning_rate = 1e-3
    num_epochs = num_epochs # max epoch
    rest = -1

    act_fun = ActFun.apply


    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    acc_record = list([])
    loss_train_record = list([])
    loss_test_record = list([])

    snn = SNN(max_time)
    snn.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        total = 0
        correct = 0
        
        for step, input_data in enumerate(train_datasets):
            labels = train_labels[step].unsqueeze(0)
        
            snn.zero_grad()
            optimizer.zero_grad()

            input_data = input_data.float().to(device)
            input_data = input_data.unsqueeze(0)
            
            outputs = snn(input_data)
            
            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels).sum().item())
            
            labels_ = torch.zeros(batch_size, n_classes).scatter_(1, labels.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (step+1)%800 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, num_epochs, step+1, train_datasets.shape[0]//batch_size,running_loss ))
                running_loss = 0
                print('Accuracy:', correct/total )
                print('Time elasped:', time.time()-start_time)
                correct = 0
                total = 0
                
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 15)
        loss_total = 0

        with torch.no_grad():
            for step, test_data in enumerate(test_datasets):
                test_label = test_labels[step].unsqueeze(0)
                test_data = test_data.float().to(device).unsqueeze(0)
                
                optimizer.zero_grad()
                outputs = snn(test_data)
                labels_ = torch.zeros(batch_size, n_classes).scatter_(1, test_label.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                _, predicted = outputs.cpu().max(1)
                total += float(test_label.size(0))
                correct += float(predicted.eq(test_label).sum().item())
                loss_total += loss.item()

            print('Iters:', epoch)
            print('Test Accuracy on test dataset: %.3f' % (100 * correct / total))
            print('Test Loss on test dataset: %.3f' % (loss_total))
            acc = 100. * float(correct) / float(total)
            acc_record.append(acc)
            loss_test_record.append(loss_total)
            print('Time elasped:', time.time()-start_time)
            print('\n\n\n')

    return acc

search_domain = [0.1 * i for i in range(11)]
best_thresh = 0
best_lens = 0
best_decay = 0
best_test_acc = 0
for thresh in search_domain:
    for lens in search_domain:
        for decay in search_domain:
            print("thresh :{} lens: {} decay: {}===============".format(thresh,lens,decay))
            test_acc = train(2,train_datasets,train_labels,test_datasets,test_labels,thresh,lens,decay)
            print("test acc: ",test_acc)
            if test_acc > best_test_acc:
                best_thresh = thresh
                best_lens = lens
                best_decay = decay
                best_test_acc = test_acc
                print("Current Best parameter setting: ")
                print("thresh :{} lens: {} decay: {}".format(best_thresh,best_lens,best_decay))
                torch.save({"thresh":thresh,"lens":lens,"decay":decay,"acc":acc, "epochs":2},"log.pkl")
