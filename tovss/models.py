"""Frozen CLIP + DINO encoders shared by the training-free per-frame models.

The per-frame heads (pipeline/perframe.py) read the value tokens of CLIP's last block and project them as
z = proj(ln_post(out_proj(A @ v))), with no residual and no FFN.
"""
import torch
import torch.nn.functional as F
import open_clip

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


def _norm(x, mean, std):
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


class Encoders(torch.nn.Module):
    def __init__(self, clip_name='ViT-B-16', clip_pretrained='openai', dino_name='dinov2_vitb14_reg', device='cuda'):
        super().__init__()
        self.device = device
        self.clip, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained=clip_pretrained)
        self.clip = self.clip.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.V = self.clip.visual
        self.patch = self.V.patch_size[0] if isinstance(self.V.patch_size, (tuple, list)) else self.V.patch_size
        self.dino_name = dino_name
        if dino_name.startswith('dinov2'):
            self.dino = torch.hub.load('facebookresearch/dinov2', dino_name)
            self.dino_patch = 14
        else:  # DINO v1, e.g. dino_vitb8 / dino_vitb16
            self.dino = torch.hub.load('facebookresearch/dino:main', dino_name)
            self.dino_patch = 8 if dino_name.endswith('8') else 16
        self.dino = self.dino.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    # ---------------- text ----------------
    @torch.no_grad()
    def encode_queries(self, queries, templates):
        """queries: list (per class) of list of names. Returns (Q, D) normalised and owner index (Q,)."""
        embs, owner = [], []
        for c, names in enumerate(queries):
            for n in names:
                tok = self.tokenizer([t.format(n) for t in templates]).to(self.device)
                e = F.normalize(self.clip.encode_text(tok).float(), dim=-1).mean(0)
                embs.append(F.normalize(e, dim=-1))
                owner.append(c)
        return torch.stack(embs), torch.tensor(owner, device=self.device)

    # ---------------- image ----------------
    @torch.no_grad()
    def clip_tokens(self, img):
        """img (1,3,H,W) in [0,1], H,W multiples of patch. Returns q,k,v (1,N+1,D), last block, grid."""
        V = self.V
        x = _norm(img, CLIP_MEAN, CLIP_STD)
        gh, gw = x.shape[2] // self.patch, x.shape[3] // self.patch
        x = V.conv1(x).flatten(2).transpose(1, 2)
        x = torch.cat([V.class_embedding.view(1, 1, -1).to(x.dtype).expand(x.shape[0], -1, -1), x], 1)
        pe = V.positional_embedding
        n0 = int(round((pe.shape[0] - 1) ** 0.5))
        pp = pe[1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2)
        pp = F.interpolate(pp, (gh, gw), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(gh * gw, -1)
        x = V.ln_pre(x + torch.cat([pe[:1], pp]).to(x.dtype))
        for blk in V.transformer.resblocks[:-1]:
            x = blk(x)
        blk = V.transformer.resblocks[-1]
        q, k, v = F.linear(blk.ln_1(x), blk.attn.in_proj_weight, blk.attn.in_proj_bias).chunk(3, -1)
        return q, k, v, blk, (gh, gw)


def query_logits_to_class(qlogits, owner, num_classes):
    """qlogits (N,Q) -> (N,C): max over the synonyms of each class."""
    out = qlogits.new_full((qlogits.shape[0], num_classes), float('-inf'))
    return out.scatter_reduce(1, owner.view(1, -1).expand_as(qlogits), qlogits, reduce='amax')
