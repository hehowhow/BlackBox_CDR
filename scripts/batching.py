"""Batch GPU indexing without changing PyTorch's shuffle sampler or optimizer steps."""
import torch
from torch.utils.data import TensorDataset, DataLoader


class BatchedTensorDataset(TensorDataset):
    def __getitems__(self, indices):
        indices = torch.as_tensor(indices, device=self.tensors[0].device)
        return tuple(tensor[indices] for tensor in self.tensors)


def identity(batch):
    return batch


def loader(tensors, batch_size):
    return DataLoader(BatchedTensorDataset(*tensors), batch_size=batch_size, shuffle=True, collate_fn=identity)


if __name__ == '__main__':
    tensors = (torch.arange(211, device='cuda').reshape(-1, 1), torch.arange(211, device='cuda'))
    torch.manual_seed(2020)
    reference = list(DataLoader(TensorDataset(*tensors), batch_size=32, shuffle=True))
    torch.manual_seed(2020)
    actual = list(loader(tensors, 32))
    for ref, out in zip(reference, actual):
        for a, b in zip(ref, out):
            assert torch.equal(a, b)
    assert len(reference) == len(actual)
    print('PASS: vectorized batches exactly match original shuffled batches, including final batch')
