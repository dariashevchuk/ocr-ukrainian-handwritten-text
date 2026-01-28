import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class OCR_CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(OCR_CRNN, self).__init__()     
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        original_weights = resnet.conv1.weight.data
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data = original_weights.sum(dim=1, keepdim=True)
        resnet.conv1 = new_conv1
        
        resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)
        
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None)) 
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size + 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x) 
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x.log_softmax(2)

class OCR_SimpleCNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(OCR_SimpleCNN, self).__init__()
        
        # Custom "Own Architecture" Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), 
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), 
            
            # Final collapse to height=1
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size + 1)

    def forward(self, x):
        x = self.features(x)       
        x = x.squeeze(2)           
        x = x.permute(0, 2, 1)     
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x.log_softmax(2)