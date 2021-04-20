import torch

if __name__ == "__main__":
    print("temp")
    shape = (3,32,32)
    print(*shape)
    bs = 1
    dummy_x = torch.empty(bs,*shape)
    
    print(dummy_x.flatten(1).size(1))