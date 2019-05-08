import torch


class DataTypes:
    def __init__(self, device='cpu'):
        device = device.lower()
        assert device in ['cpu', 'gpu']
        self.__dict__ = {}

        if device == 'cpu':
            self.float = torch.FloatTensor
            self.double = torch.DoubleTensor
            self.half = torch.HalfTensor
            self.char = torch.CharTensor
            self.short = torch.ShortTensor
            self.int = torch.IntTensor
            self.long = torch.LongTensor
            self.byte = torch.ByteTensor
        else:
            self.float = torch.cuda.FloatTensor
            self.double = torch.cuda.DoubleTensor
            self.half = torch.cuda.HalfTensor
            self.char = torch.cuda.CharTensor
            self.short = torch.cuda.ShortTensor
            self.int = torch.cuda.IntTensor
            self.long = torch.cuda.LongTensor
            self.byte = torch.cuda.ByteTensor

