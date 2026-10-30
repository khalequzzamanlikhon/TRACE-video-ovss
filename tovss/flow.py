"""RAFT-large optical flow (torchvision, cached weights) + warping and forward-backward occlusion check."""
import torch
import torch.nn.functional as F


class Raft:
    def __init__(self, device='cuda'):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.C_T_SKHT_V2
        self.model = raft_large(weights=w).to(device).eval()
        self.tf = w.transforms()

    @torch.no_grad()
    def __call__(self, a, b):
        """a, b: (1,3,H,W) float in [0,1]. Returns flow a->b (1,2,H,W). Pads to multiples of 8."""
        H, W = a.shape[2:]
        ph, pw = (-H) % 8, (-W) % 8
        if ph or pw:
            a = F.pad(a, (0, pw, 0, ph), mode='replicate')
            b = F.pad(b, (0, pw, 0, ph), mode='replicate')
        a2, b2 = self.tf(a, b)
        f = self.model(a2, b2, num_flow_updates=12)[-1]
        return f[:, :, :H, :W]


def backward_warp(x, flow, mode='nearest'):
    """Sample x (B,C,H,W) (defined in frame t-1) at p + flow(p), flow = flow t->t-1 (1,2,H,W).
    Returns warped x in frame t, in-bounds mask (H,W) and the sampling grid."""
    _, _, H, W = flow.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing='ij')
    x2, y2 = xs + flow[0, 0], ys + flow[0, 1]
    grid = torch.stack([2 * x2 / max(W - 1, 1) - 1, 2 * y2 / max(H - 1, 1) - 1], -1)[None]
    out = F.grid_sample(x.float(), grid, mode=mode, align_corners=True)
    inb = (x2 >= 0) & (x2 <= W - 1) & (y2 >= 0) & (y2 <= H - 1)
    return out, inb, grid


def fb_valid(f_tp, f_pt, grid, a1=0.01, a2=0.5):
    """Sundaram et al. 2010 forward-backward consistency (H,W) bool."""
    fb = F.grid_sample(f_pt, grid, mode='bilinear', align_corners=True)
    d = (f_tp + fb).pow(2).sum(1)[0]
    m = f_tp.pow(2).sum(1)[0] + fb.pow(2).sum(1)[0]
    return d < a1 * m + a2
