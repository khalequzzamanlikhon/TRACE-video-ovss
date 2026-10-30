"""Frozen CLIP + DINO encoders and training-free per-frame heads.

Correct value path (fixes v1 bug C1): z = proj(ln_post(out_proj(A @ v))), no residual, no FFN.
DINO features are computed on an image resized so its patch grid equals CLIP's grid (fixes M6).
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

    @torch.no_grad()
    def project(self, v_mix, blk):
        """v_mix (1,N+1,D) value tokens incl. CLS slot -> normalised patch embeddings (1,N,E)."""
        z = self.V.ln_post(blk.attn.out_proj(v_mix))[:, 1:]
        if self.V.proj is not None:
            z = z @ self.V.proj
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def dino_keys(self, img, grid):
        gh, gw = grid
        x = F.interpolate(img, (self.dino_patch * gh, self.dino_patch * gw), mode='bilinear', align_corners=False)
        x = _norm(x, IN_MEAN, IN_STD)
        if self.dino_name.startswith('dinov2'):
            t = self.dino.forward_features(x)['x_norm_patchtokens']
        else:
            t = self.dino.get_intermediate_layers(x, n=1)[0][:, 1:]
        return F.normalize(t.float(), dim=-1)

    @torch.no_grad()
    def encode(self, img, need_dino=True):
        q, k, v, blk, grid = self.clip_tokens(img)
        out = {'q': q, 'k': k, 'v': v, 'blk': blk, 'grid': grid}
        if need_dino:
            out['E'] = self.dino_keys(img, grid)[0]          # (N, Dd)
        return out


def query_logits_to_class(qlogits, owner, num_classes):
    """qlogits (N,Q) -> (N,C): max over the synonyms of each class."""
    out = qlogits.new_full((qlogits.shape[0], num_classes), float('-inf'))
    return out.scatter_reduce(1, owner.view(1, -1).expand_as(qlogits), qlogits, reduce='amax')


def intra_proxy_attention(E, tau=0.07, adaptive=False):
    """E (N,D) normalised -> (N,N) row-stochastic proxy attention."""
    S = E @ E.T
    if adaptive:
        S = S.masked_fill(S < S.mean(-1, keepdim=True), float('-inf'))
    return torch.softmax(S / tau, -1)


class PerFrameHead:
    """Training-free per-frame OVSS heads that share the Encoders.

    method: 'maskclip' (A = I) | 'proxy' (DINO proxy attention) | 'sclip' (CSA: q-q + k-k attention)
    """

    def __init__(self, enc, text, owner, num_classes, method='proxy', tau=0.07, adaptive=False, logit_scale=100.0):
        self.enc, self.text, self.owner, self.C = enc, text, owner, num_classes
        self.method, self.tau, self.adaptive, self.scale = method, tau, adaptive, logit_scale

    def needs_dino(self):
        return self.method == 'proxy'

    def attention(self, feats):
        if self.method == 'proxy':
            return intra_proxy_attention(feats['E'], self.tau, self.adaptive)
        if self.method == 'sclip':
            nh = self.enc.V.transformer.resblocks[-1].attn.num_heads
            q, k = feats['q'][0, 1:], feats['k'][0, 1:]
            N, D = q.shape
            qh, kh = q.view(N, nh, D // nh).transpose(0, 1), k.view(N, nh, D // nh).transpose(0, 1)
            sc = (D // nh) ** -0.5
            return 0.5 * (torch.softmax(qh @ qh.transpose(1, 2) * sc, -1) +
                          torch.softmax(kh @ kh.transpose(1, 2) * sc, -1))   # (nh,N,N)
        return None  # maskclip

    def mix_values(self, feats, A):
        v = feats['v'][0]
        if A is None:
            return v[None]
        vp = v[1:]
        if A.dim() == 3:  # per-head attention
            nh = A.shape[0]
            N, D = vp.shape
            vh = vp.view(N, nh, D // nh).transpose(0, 1)
            mixed = (A @ vh).transpose(0, 1).reshape(N, D)
        else:
            mixed = A @ vp
        return torch.cat([v[:1], mixed], 0)[None]

    def logits_from_values(self, v_mix, feats):
        z = self.enc.project(v_mix, feats['blk'])[0]                    # (N,E)
        ql = self.scale * z @ self.text.T                                # (N,Q)
        return query_logits_to_class(ql, self.owner, self.C)             # (N,C)

    def __call__(self, feats):
        return self.logits_from_values(self.mix_values(feats, self.attention(feats)), feats)
