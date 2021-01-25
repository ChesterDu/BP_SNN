import argparse
import data
import model
from model import init_weights, lr_scheduler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import os
import time
import tqdm
import random

parser = argparse.ArgumentParser()

parser.add_argument("--method",type=str)
parser.add_argument("--dataset",type=str,default="TIDIGITS")
parser.add_argument("--train_batch_size",type=int,default=1)
parser.add_argument("--eval_batch_size",type=int,default=1)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--num_epochs",type=int,default=10000)
parser.add_argument("--gpu_id",type=int,default=-1)
parser.add_argument("--criterion",type=str,default="MSE")
parser.add_argument("--n_classes",type=int,default=11)
parser.add_argument("--n_input",type=int,default=620)

args = parser.parse_args()

if args.gpu_id == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.gpu_id))

train_set, test_set = data.load_data(args)
# train_set = TensorDataset(train_datasets,train_labels)
train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle=True,drop_last=False)
# test_set = TensorDataset(test_datasets,test_labels)
test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle=False,drop_last=False)

snn,ActFun = model.make_model(args.method,device)
act_fun = ActFun.apply
snn = snn.to(device)
n_classes = args.n_classes


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

snn = snn.apply(init_weights)

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr)
for epoch in range(args.num_epochs):
    running_loss = 0
    start_time = time.time()
    total = 0
    correct = 0
    bar = tqdm.tqdm(total = len(train_loader))
    bar.update(0)
    
    for step, (input_data,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = input_data.shape[0]

        input_data = input_data.float().to(device)
        labels = labels.to(device)
        
        outputs = snn(input_data,act_fun)
        # outputs = snn(input_data)
        
        _, predicted = outputs.max(1)
        # print(predicted.shape)
        # print(labels.shape)
        total += float(labels.shape[0])
        # print(predicted.eq(labels))
        correct += float(predicted.eq(labels.squeeze(-1)).sum().item())
        # print(total,correct)
        
        labels_ = torch.zeros(batch_size, n_classes).to(device).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs, labels_)
        # loss.requires_grad = True
#         loss = criterion(outputs.cpu(), labels.reshape(-1))
        running_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        bar.update(1)
        if (step+1)%50 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, args.num_epochs, step+1, len(train_loader),running_loss ))
            loss_train_record.append(running_loss)
            running_loss = 0
            print('Accuracy:', correct/total )
            print('Time elasped:', time.time()-start_time)
            correct = 0
            total = 0
            
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, args.lr, 10)
    loss_total = 0

    with torch.no_grad():
        for step, (test_data,test_label) in enumerate(test_loader):
            test_label = test_label.to(device)
            test_data = test_data.float().to(device)
            
            optimizer.zero_grad()
            outputs = snn(test_data,act_fun)

            _, predicted = outputs.max(1)
            total += float(test_label.shape[0])
            correct += float(predicted.eq(test_label.squeeze(-1)).sum().item())

        print('Iters:', epoch)
        print('Test Accuracy on test dataset: %.3f' % (100 * correct / total))
        # print('Test Loss on test dataset: %.3f' % (loss_total))
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        # loss_test_record.append(loss_total)
        print('Time elasped:', time.time()-start_time)
        print('\n\n\n')
        torch.save({"acc":acc_record,"train_loss":loss_train_record},"log/{}.pkl".format(args.method))
        # torch.save(model.state_dict(),"model.pkl")
