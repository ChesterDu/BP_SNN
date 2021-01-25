import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
thresh = 0.75 # neuronal threshold
lens = thresh # hyper-parameters of approximate function
decay = 0.2 # decay constants
rest = -0.1
# define approximate firing function
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * torch.sigmoid(10*x)
        return x


class ActFun_modified(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) <= lens
        grad_pos = grad_input > 0
        grad_neg = grad_input < 0
        is_spike = input >= thresh
        is_not_spike = input < thresh #enforce

        return grad_input * (1.0 + is_spike.float()) * temp.float()


class ActFun_orig(torch.autograd.Function):
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

def mem_update(fc_input, x, mem, spike, act_fun):
    mem_t = mem * decay * (1. - spike)  + fc_input(x)
    spike = act_fun(mem_t) # act_fun : approximation firing function
    return mem_t, spike

n_input = 620
n_classes = 11
n_hidden1 = 200
n_hidden2 = 620
init_mem_flag = False
cfg_fc = [n_input, n_hidden1, n_hidden2, n_classes,init_mem_flag]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

class Print(torch.nn.Module):
    def forward(self, x):
        print("=========")
        print(x.shape)
        return x

class SnnLinear(nn.Module):
    def __init__(self,input_dim, output_dim,act_fun,device=torch.device("cpu")):
        super(SnnLinear, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.h_mem = None
        self.h_spike = None
        self.h_sumspike = None
        self.act_fun = act_fun
    def forward(self,input):
        batch_size = input.shape[0]
        if cfg_fc[-1]:
            self.init_mem(input.shape)

        self.h_mem, self.h_spike = self.mem_update(self.fc, input, self.h_mem, self.h_spike, self.act_fun)

        self.h_sumspike += self.h_spike

        return self.h_spike
    def init_mem(self,shape):
        self.h_mem = self.h_spike = self.h_sumspike = torch.zeros(shape[0], self.output_dim, device=self.device)
    
    def mem_update(self,fc_input, x, mem, spike, act_fun):
        mem_t = mem * decay * (1. - spike)  + fc_input(x)
        spike = self.act_fun(mem_t) # act_fun : approximation firing function
        return mem_t, spike
        

class SnnConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,act_fun,device=torch.device("cpu")):
        super(SnnConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        # self.max_time = max_time
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.h_mem = None
        self.h_spike = None
        self.h_sumspike = None
        self.device = device
        self.act_fun = act_fun
    def forward(self,input):
        if cfg_fc[-1]:
            self.init_mem(input.shape)
            # print(1)
            # print(self.h_mem,self.h_spike,self.h_sumspike)

        self.h_mem, self.h_spike = self.mem_update(self.conv, input, self.h_mem, self.h_spike, self.act_fun)

        self.h_sumspike += self.h_spike

        return self.h_spike

    def init_mem(self,shape):
        batch_size = shape[0]
        h_out = shape[2] + self.padding * 2 - self.kernel_size + 1
        w_out = shape[3] + self.padding * 2 - self.kernel_size + 1
        self.h_mem = self.h_spike = self.h_sumspike = torch.zeros(batch_size, self.out_channels, h_out, w_out, device=self.device)
    
    def mem_update(self, fc_input, x, mem, spike, act_fun):
        mem_t = mem * decay * (1. - spike)  + fc_input(x)
        spike = self.act_fun(mem_t) # act_fun : approximation firing function
        return mem_t, spike
        

class SNN_Lente(nn.Module):
    def __init__(self,device,act_fun,max_time=50):
        super(SNN_Lente, self).__init__()
        self.net = torch.nn.Sequential(
        SnnConv2d(in_channels=1,out_channels=3,kernel_size=5,stride=1,padding=2,act_fun=act_fun,device=device),
        Swish(),
        nn.AvgPool2d(kernel_size=2,stride =2),

        # SnnConv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0,act_fun=act_fun,device=device),
        # nn.ReLU(),
        # nn.AvgPool2d(kernel_size=2,stride=2),
        
        nn.Flatten(),
        # Print(),
        # SnnLinear(576,120,act_fun=act_fun,device=device),
        # Print(),
        # nn.ReLU(),
        SnnLinear(867,84,act_fun=act_fun,device=device),
        # nn.ReLU(),
        Swish(),
        SnnLinear(84,10,act_fun=act_fun,device=device)
        )
        self.device = device
        self.max_time = max_time

    def forward(self,input):
        batch_size = input.shape[0]
        time_window = self.max_time
        out_sumspike = torch.zeros(batch_size,n_classes,device=self.device)
        cfg_fc[-1] = True
        for step in range(time_window): # simulation time steps
            x = input[:,step]
            out_sumspike += self.net(x.reshape(batch_size,1,34,34))
            cfg_fc[-1] = False

        outputs = out_sumspike / 50
        return outputs



class SNN2(nn.Module):
    def __init__(self,device,max_time=50):
        super(SNN2, self).__init__()
        self.fc1 = nn.Sequential(
#             nn.Linear(cfg_fc[0], cfg_fc[1]),
            nn.Linear(cfg_fc[0], cfg_fc[2], bias=False),
            nn.ReLU()
#             nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(cfg_fc[2], cfg_fc[3], bias=False),
            nn.ReLU()
#             nn.Sigmoid()
        )
        self.max_time = max_time
        self.fc_neuron_para = nn.Sequential(
                nn.Linear(1,1,bias=False),
                nn.ReLU()
        )
        self.device = device
        
    def forward(self, input,act_fun):

        time_window = self.max_time
        batch_size = input.shape[0]
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[2], device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[3], device=self.device)

        for step in range(time_window): # simulation time steps
            x = input[:,step]
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike, act_fun)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, act_fun)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / 50
        return outputs

class SNN1(nn.Module):
    def __init__(self, device, max_time=50):
        super(SNN1, self).__init__()

        self.fc2 = nn.Sequential(
            nn.Linear(cfg_fc[0], cfg_fc[-1], bias = False),
#             custom_fc(cfg_fc[0], cfg_fc[-1]),
            nn.ReLU()
#             nn.Sigmoid()
        )
        self.max_time = max_time
        self.fc_neuron_para = nn.Sequential(
                nn.Linear(1,1,bias=False),
                nn.ReLU()
        )
        self.device = device
        
    def forward(self, input,act_fun):

        time_window = self.max_time
#         h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        batch_size = input.shape[0]
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[-1], device=self.device)

        for step in range(time_window): # simulation time steps
            x = input[:,step]
            h2_mem, h2_spike = mem_update(self.fc2, x, h2_mem, h2_spike, act_fun)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / 50
        return outputs

    
class SNN3(nn.Module):
    def __init__(self, device, max_time=50):
        super(SNN3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(cfg_fc[0], 200, bias=False),
            Swish()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 100, bias=False),
            Swish()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 11, bias=False),
            Swish()
        )
        self.max_time = max_time
        self.fc_neuron_para = nn.Sequential(
                nn.Linear(1,1,bias=False),
                nn.ReLU()
        )
        self.device = device
        
    def forward(self, input,act_fun):
        batch_size = input.shape[0]
        time_window = self.max_time
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, 200, device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, 100, device=self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, 11, device=self.device)

        for step in range(time_window): # simulation time steps
            x = input[:,step]
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike, act_fun)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, act_fun)
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike, act_fun)
            h3_sumspike += h3_spike

        outputs = h3_sumspike / 50
        return outputs
    
class SNN4(nn.Module):
    def __init__(self, device, max_time=50):
        super(SNN4, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(620, 200, bias = False),
            Swish()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(200,100, bias = False),
            Swish()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(100,11, bias = False),
            Swish()
        )

        self.max_time = max_time
        
        self.h1 = None
        self.h2_spike_l = None
        self.h2_mem_l = None

        self.h2 = None
        self.h3_spike_l = None
        self.h3_mem_l = None

        self.h3 = None
        self.h4_spike_l = None
        self.h4_mem_l = None

        self.device = device
        
        
        self.CW = 10 # Kernel effect window size
        
    def forward(self, input,act_fun):

        time_window = self.max_time
        
        batch_size = input.shape[0]
        self.h1 = torch.zeros(batch_size, time_window, cfg_fc[1], device=self.device).float()
        self.h2_spike_l = torch.zeros(batch_size, time_window, cfg_fc[1], device=self.device).float()
        self.h2_mem_l = torch.zeros(batch_size, time_window, cfg_fc[1], device=self.device).float()
        
        h2_spike = torch.zeros(batch_size, cfg_fc[1], device=self.device)
        # h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=self.device)

        self.h2 = torch.zeros(batch_size, time_window, 100, device=self.device).float()
        self.h3_spike_l = torch.zeros(batch_size, time_window, 100, device=self.device).float()
        self.h3_mem_l = torch.zeros(batch_size, time_window, 100, device=self.device).float()

        h3_spike = torch.zeros(batch_size,100,device=self.device)
        
        self.h3 = torch.zeros(batch_size, time_window, cfg_fc[3], device=self.device).float()
        self.h4_spike_l = torch.zeros(batch_size, time_window, cfg_fc[3], device=self.device).float()
        self.h4_mem_l = torch.zeros(batch_size, time_window, cfg_fc[3], device=self.device).float()
        
        h4_spike = torch.zeros(batch_size, cfg_fc[3], device=self.device)
        h4_sumspike = torch.zeros(batch_size, cfg_fc[3], device=self.device)
        
        

        for t in range(time_window): # simulation time steps
            x = input[:,t]
            self.h1[:,t] = self.fc1(x)
#             print(f't={t}:{self.h1[:,t]}')
            
#             h2_mem = torch.zeros(batch_size, cfg_fc[-1], device=device)
        
            for t_k in range(self.CW):
                if t - t_k >= 0:
                    self.h2_mem_l[:,t] += self.K(t_k) * self.h1[:, t-t_k] - self.K_spike(t_k) * self.h2_spike_l[:, t-t_k]
                    
            h2_spike = act_fun(self.h2_mem_l[:,t].clone())
            
            self.h2_spike_l[:,t] = h2_spike * (self.h2_mem_l[:,t].clone())

            self.h2[:,t] = self.fc2(h2_spike)
            # print(self.h2.shape)
#             print(f't={t}:{self.h1[:,t]}')
            
#             h2_mem = torch.zeros(batch_size, cfg_fc[-1], device=device)
        
            for t_k in range(self.CW):
                if t - t_k >= 0:
                    self.h3_mem_l[:,t] += self.K(t_k) * self.h2[:, t-t_k] - self.K_spike(t_k) * self.h3_spike_l[:, t-t_k]
                    
            h3_spike = act_fun(self.h3_mem_l[:,t].clone())
            # print(h3_spike.shape)
            
            self.h3_spike_l[:,t] = h3_spike * (self.h3_mem_l[:,t].clone())

            self.h3[:,t] = self.fc3(h3_spike)
        
#             h2_mem, h2_spike = mem_update(self.fc2, x, h2_mem, h2_spike, self.fc_neuron_para)
            for t_k in range(self.CW):
                if t - t_k >= 0:
                    self.h4_mem_l[:,t] += self.K(t_k) * self.h3[:, t-t_k] - self.K_spike(t_k) * self.h4_spike_l[:, t-t_k]

            h4_spike = act_fun(self.h4_mem_l[:,t].clone())  
            h4_sumspike += h4_spike
            # print(h3_sumspike)
    
        outputs = h4_sumspike / 50
        return outputs
    
    # define effect window [0,CW)
    def K(self, t0):
        t_s = 2 #time constant
        
#         return math.exp(-t0/t_s)
        
        if(t0 < 0):
            return 0
        
        t = t0 - int(self.CW/2)
        
        if(t < 0):
            return -0.5 * math.exp(t/(t_s/2))

        return math.exp(-t/t_s)
    
    def K_spike(self, t0):
        t_s = 2 #time constant
        return math.exp(-t0/t_s)

def init_weights(m):
    if type(m) == nn.Linear:
        print("init")
        torch.nn.init.xavier_normal_(m.weight)
#         torch.nn.init.uniform_(m.weight, -0.2, 0.2)

class custom_fc(nn.Module):
    def __init__(self, n_input, n_output):
        super(custom_fc,self).__init__()
#         self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(4)])
#         self.params.append(nn.Parameter(torch.randn(4,1)))
        
        self.params = nn.Parameter(torch.zeros(1, n_input, n_output))
#         self.params.unsqueeze(0)
        
#         self.params += 0.01
        torch.nn.init.normal_(self.params, mean=-1.0, std=1)
        print(self.params.shape)
        print(self.params)

    def forward(self,x):
        
        # [batch, n_input]
        x = x.unsqueeze(1)
        
        # [batch, 1, n_input]
        
        
        
        x = torch.bmm(x, F.sigmoid(self.params))
            
        # [batch, 1, n_output]
        x = x.squeeze(1)
#         print(x)
        return x

class SNN(nn.Module):
    def __init__(self, device, max_time=50):
        super(SNN, self).__init__()
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
#         self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

#         self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.max_time = max_time
        self.fc_neuron_para = nn.Linear(1,1,bias=False)
        self.device = device

    def forward(self, input,act_fun):
#         c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
#         c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

#         h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        time_window = self.max_time
        batch_size = input.shape[0]
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=self.device)

        for step in range(time_window): # simulation time steps
            x = input[:,step]
            h2_mem, h2_spike = mem_update(x, h2_mem, h2_spike,act_fun)
            # self.mem_update(x, h2_mem, h2_spike)
            h2_sumspike += h2_spike
#             print(h2_mem[0][0])

        outputs = h2_sumspike / 50
        return outputs



def make_model(method, device):
    if method == "basic_with_original_actfun":
        model = SNN1(device,max_time=130)
        ActFun = ActFun_orig
    if method == "basic_with_modified_actfun":
        model = SNN1(device,max_time=130)
        ActFun = ActFun_modified
    if method == "2_layer_with_modified_actfun":
        model = SNN2(device, max_time=256)
        ActFun = ActFun_modified
    if method == "3_layer_with_modified_actfun":
        model = SNN3(device, max_time=130)
        ActFun = ActFun_modified
    if method == "conv_with_modified_actfun":
        model = SNN4(device,max_time=130)
        ActFun = ActFun_modified
    if method == "lenet_conv_with_modified_actfun":
        ActFun = ActFun_modified
        act_fun = ActFun.apply
        model = SNN_Lente(device,act_fun,max_time=256)

    return model, ActFun
