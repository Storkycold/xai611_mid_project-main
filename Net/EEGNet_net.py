import torch.nn as nn
from layers import LinearWithConstraint, Conv2dWithConstraint, ConvSamePad2d
from torch.autograd import Function
import torch.nn as nn
from layers import LinearWithConstraint, Conv2dWithConstraint, ConvSamePad2d

class EEGNet(nn.Module):
    def __init__(self, args, shape):
        super(EEGNet, self).__init__()
        self.num_ch = shape[2]
        self.F1 = 16
        self.F2 = 32
        self.D = 2
        self.sr = 250
        self.P1 = 4
        self.P2 = 8
        self.t1 = 16
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, kernel_size=(1, self.sr//2), bias=False, padding='same'),
            nn.BatchNorm2d(self.F1)
        )
        #output shape:(batch_size, self.F1, 1, time_steps/2)

        # Spatial conv (Depth-wise conv, EEGNet 2nd block)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.num_ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P1))
        )
        # Separable conv (EEGNet 3rd block)
        self.separable_conv = nn.Sequential(
            # depth-wise
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, self.t1), groups=self.F1 * self.D, bias=False),
            # point-wise
            nn.Conv2d(self.F2, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P2))
        )
        # Dense
        self.linear = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(in_features=self.F2 * 33, out_features=4, max_norm=0.25)
        )
        #input shape: from separable_conv
        #output shape: (batch_size, 4)

    def forward(self, x):
        out = self.temporal_conv(x)
        out = self.spatial_conv(out)
        out = self.separable_conv(out)
        return out 

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd 
        return x.view_as(x) 

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd 
        return output, None

#DANN 
class DANN(nn.Module):
    def __init__(self, args, shape):
        super(DANN, self).__init__()
        #temporal_conv, spatial_conv, separable_conv
        #EEGNet을 없애고 feature_extractor로 한번에?
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.F1, kernel_size=(1, self.sr//2),
                      bias=False, padding='same'),
            nn.BatchNorm2d(self.F1),
            Conv2dWithConstraint(
                self.F1, self.F1 * self.D, kernel_size=(self.num_ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P1)),
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1,
                      self.t1), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F2, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P2))
        )
        #linear
        self.label_classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(in_features=self.F2 * 33, out_features=4, max_norm=0.25)
        )
        #domain_classifier
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.F2 * 33, out_features=2)
            #predict whether source or target
        )
        #

    def forward(self, x, alpha):
        # Feature extraction
        out = self.feature_extractor(x)
        # out.size(0): batch_size 
        out = out.view(out.shape(0), -1)
        reverse_out = GradReverse.apply(out, alpha)
        # label_classifier
        class_output = self.label_classifier(out)
        # domain_classifier
        domain_output = self.domain_classifier(reverse_out)

        return class_output, domain_output
