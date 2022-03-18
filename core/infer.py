import torch
import torch.nn.functional as F


def get_reverse_list(ori_shape, transforms):
    """
    get reverse list of transform.
    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.
    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    d, h, w = ori_shape[0], ori_shape[1], ori_shape[2]
    for op in transforms:
        if op.__class__.__name__ in ['Resize3D']:
            reverse_list.append(('resize', (d, h, w)))
            d, h, w = op.size[0], op.size[1], op.size[2]

    return reverse_list


def reverse_transform(pred, ori_shape, transforms, mode='trilinear'):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    intTypeList = [torch.int8, torch.int16, torch.int32, torch.int64]
    dtype = pred.dtype
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            d, h, w = item[1][0], item[1][1], item[1][2]
            if pred.device == 'cpu' and dtype in intTypeList:
                pred = pred.type(torch.float32)
                pred = F.interpolate(pred, (d, h, w), mode=mode)
                pred = pred.type(dtype)
            else:
                pred = F.interpolate(pred, (d, h, w), mode=mode)
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return


def inference(model, im, ori_shape=None, transforms=None):
    """
    Inference for image.
    Args:
        model (nn.Module): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, d, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, d, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
        im = im.permute(0, 2, 3, 4, 1)

    logits = model(im)

    if ori_shape is not None and ori_shape != logits.shape[2:]:
        logits = reverse_transform(logits,
                                   ori_shape,
                                   transforms,
                                   mode='bilinear')

    pred = torch.argmax(logits, dim=1, keepdim=True, dtype='int32')

    return pred, logits
