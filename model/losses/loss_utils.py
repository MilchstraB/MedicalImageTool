import torch


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
    (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # new axis order
    axis_order = (1, 0) + tuple(range(2, len(tensor.shape)))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = torch.permute(tensor, dims=axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return torch.flatten(transposed, start_dim=1, end_dim=-1)


def class_weights(tensor):
    # normalize the input first
    tensor = torch.nn.functional.softmax(tensor, dim=1)
    flattened = flatten(tensor)
    nominator = (1. - flattened).sum(-1)
    denominator = flattened.sum(-1)
    class_weights = nominator / denominator
    class_weights.stop_gradient = True

    return
