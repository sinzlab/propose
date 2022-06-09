
def mpjpe(pred, gt):
    return (((pred - gt) ** 2).sum(-1) ** .5).mean()
