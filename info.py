import torch
from torchinfo import summary

from model import AlexNet

def main():
    model = AlexNet(3, 1000)
    summary(model, input_size=[1, 3, 224, 224])
    X = torch.rand((1, 3, 224, 224)).to("cuda")
    print(model(X).shape)

if __name__ == "__main__":
    main()