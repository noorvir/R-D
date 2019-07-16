import torch


class DataTypes:

    def __init__(self, device='cpu'):
        self.__dict__ = {}
        self._device = device.lower()
        assert self._device in ['cpu', 'cuda']

        if self._device == 'cpu':
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

    def __repr__(self):
        if hasattr(self, "_device"):
            return "<torch  datatypes: type='%s'>" % self._device
        else:
            return "<torch datatypes struct>"


class _Batch(list):

    def __init__(self, dtypes):
        super(_Batch, self).__init__()
        self._dtypes = dtypes

    # def __repr__(self):
    #     if len(self) == 0:
    #         return str('[]')
        # else:
        #     return str(self)
        #     return str("<correspondences: shape=%s" % (self[0].shape,))

    def append(self, *els):
        row = torch.zeros(1, 2, len(els)).type(self._dtypes.byte)

        for i, el in enumerate(els):
            row[:, :, i] = el

        if len(self) == 0:
            super().append(row)
        else:
            assert len(els) == self[0].shape[2], ("Number of correspondence "
                                                  "types (dim=2) in batch must "
                                                  "be the same as that set in "
                                                  "the first call to append(*els).")
            super().append(row)


class BatchCorrespondencesStruct(list):
    """
    Structure of form:
    batch = [batch_el_0, batch_el_1 ... batch_el_N]

    Where, each batch_el is a Tensor of shape:
        [num_correspondences, 2, num_correspondence_types]

    For example:
    batch_el_0 = Tensor([[[mat_match_id_1x, mat_match_id_1y],
                          [non_mat_match_id_1x, non_mat_match_id_1y]
                          [obj_match_id_1x, obj_match_id_1y]
                        ]])

    Here, batch_el_0.shape = (1, 2, 3)
    """
    def __init__(self, batch_size, dtypes):
        super().__init__()
        # Ugly hacky assert to circumvent the "double import trap"
        assert hasattr(dtypes, "_device")
        self._dtypes = dtypes
        self.extend([_Batch(dtypes) for _ in range(batch_size)])

    def to_tensors(self):
        _l = []
        for i in range(len(self)):
            _l.append(torch.cat(self[i]))

        return _l
