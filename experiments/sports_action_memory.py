"""Reuse stable input storage and release inactive MPS cache between action clips."""
SHAPE=(1,100,3,448,796)


def allocate_inputs(torch):
    return (torch.empty(SHAPE,dtype=torch.uint8,device='cpu'),
            torch.empty(SHAPE,dtype=torch.uint8,device='mps'))


def copy_rgb(torch,pixels,host,device):
    if pixels.shape!=(100,448,796,3) or str(pixels.dtype)!='uint8':
        raise ValueError('Action RGB input contract differs')
    host.copy_(torch.from_numpy(pixels).permute(0,3,1,2).unsqueeze(0))
    device.copy_(host)
    return device


def release_inactive(torch):
    torch.mps.synchronize()
    torch.mps.empty_cache()
