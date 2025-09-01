import os, math, argparse, random
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# schedules 
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 1e-5, 0.999).float()

# model 
class UNetSmall(nn.Module):
    """Predicts v; input is [x_t, cond] + scalar timestep map."""
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 96):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=False),
            )
        self.inc   = nn.Sequential(nn.Conv2d(in_ch + 1, base_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.down1 = nn.Sequential(blk(base_ch,   base_ch*2), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(blk(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(blk(base_ch*4, base_ch*8), nn.MaxPool2d(2))
        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2); self.conv3 = blk(base_ch*8, base_ch*4)
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2); self.conv2 = blk(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2);   self.conv1 = blk(base_ch*2, base_ch)
        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond, t_idx):
        B, _, H, W = xt_and_cond.shape
        t_map = t_idx.view(B,1,1,1).float().repeat(1,1,H,W)
        x = torch.cat([xt_and_cond, t_map], dim=1)
        e1 = self.inc(x); e2 = self.down1(e1); e3 = self.down2(e2); e4 = self.down3(e3)
        d3 = self.up3(e4); d3 = torch.cat([d3, e3], dim=1); d3 = self.conv3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.conv2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.conv1(d1)
        return self.outc(d1)

# v <-> {x0, eps} 
def v_to_x0_eps(x_t: torch.Tensor, v: torch.Tensor, alpha_bar_t: torch.Tensor):
    sab   = torch.sqrt(alpha_bar_t).view(-1,1,1,1)
    s1mab = torch.sqrt(1.0 - alpha_bar_t).view(-1,1,1,1)
    x0  = sab * x_t - s1mab * v
    eps = s1mab * x_t + sab * v
    return x0, eps

# metrics 
def masked_mae(pred, tgt, mask: Optional[torch.Tensor] = None) -> float:
    if mask is None: w = torch.ones_like(pred[:, :1])
    else: w = (mask.unsqueeze(1) if mask.ndim==3 else mask).float().to(pred.device); w = (w>0).float()
    num = (w * (pred - tgt).abs()).sum(); den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

def masked_mse(pred, tgt, mask: Optional[torch.Tensor] = None) -> float:
    if mask is None: w = torch.ones_like(pred[:, :1])
    else: w = (mask.unsqueeze(1) if mask.ndim==3 else mask).float().to(pred.device); w = (w>0).float()
    num = (w * (pred - tgt)**2).sum(); den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

# viz helpers 
def percentile_stretch_uint8(x: np.ndarray, p2=2, p98=98) -> np.ndarray:
    lo, hi = np.percentile(x, [p2, p98])
    if hi - lo < 1e-6: hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return (y * 255).astype(np.uint8)

def s1_preview_from_cond(x_cond_chn: torch.Tensor) -> np.ndarray:
    arr = x_cond_chn.detach().cpu().numpy()
    def st(u): return percentile_stretch_uint8(u)
    if arr.shape[0] >= 2:
        A, B = arr[0], arr[1]
        return np.dstack([st(A), st(B), st(A)])
    ch = st(arr[0]); return np.dstack([ch,ch,ch])

def to_true_cir_from_t4(t4: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    a = t4.detach().cpu().numpy()
    B2, B3, B4, B8 = a[0], a[1], a[2], a[3]
    true_rgb = np.dstack([percentile_stretch_uint8(B4), percentile_stretch_uint8(B3), percentile_stretch_uint8(B2)])
    cir_rgb  = np.dstack([percentile_stretch_uint8(B8), percentile_stretch_uint8(B4), percentile_stretch_uint8(B3)])
    return true_rgb, cir_rgb

def tile2x3(s1_rgb, gt_true, pr_true, gt_cir, pr_cir, err_img) -> Image.Image:
    # resize each tile to same width for a neat 2x3 layout
    target_w = 1024
    def rz(im):
        h,w = im.shape[:2]
        if w == target_w: return Image.fromarray(im)
        nh = int(h * (target_w / w))
        return Image.fromarray(im).resize((target_w, nh), Image.BILINEAR)
    tiles = [rz(s1_rgb), rz(gt_true), rz(pr_true), rz(gt_cir), rz(pr_cir), rz(err_img)]
    h1 = max(t.size[1] for t in tiles[:3]); h2 = max(t.size[1] for t in tiles[3:])
    row1 = Image.new('RGB', (sum(t.size[0] for t in tiles[:3]), h1), (255,255,255))
    row2 = Image.new('RGB', (sum(t.size[0] for t in tiles[3:]), h2), (255,255,255))
    x = 0
    for t in tiles[:3]:
        if t.size[1] != h1: t = t.resize((t.size[0], h1), Image.BILINEAR)
        row1.paste(t, (x,0)); x += t.size[0]
    x = 0
    for t in tiles[3:]:
        if t.size[1] != h2: t = t.resize((t.size[0], h2), Image.BILINEAR)
        row2.paste(t, (x,0)); x += t.size[0]
    canvas = Image.new('RGB', (row1.size[0], row1.size[1]+row2.size[1]), (255,255,255))
    canvas.paste(row1, (0,0)); canvas.paste(row2, (0,row1.size[1]))
    return canvas

# I/O 
def load_npz_as_tensors(path: str, device: torch.device):
    d = np.load(path)
    x_cond = torch.from_numpy(np.nan_to_num(d["inputs"].astype(np.float32))).unsqueeze(0).to(device)
    x_gt   = torch.from_numpy(np.nan_to_num(d["target"].astype(np.float32))).unsqueeze(0).to(device)
    mask   = None
    if "mask" in d:
        mask = torch.from_numpy(np.nan_to_num(d["mask"].astype(np.float32))).unsqueeze(0).to(device)
    return x_cond, x_gt, mask, x_cond.size(1), x_gt.size(1)

def select_files(patch_dir: str, file_list: Optional[str], use_first_n: int) -> List[str]:
    if file_list and os.path.isfile(file_list):
        with open(file_list, "r") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        files = [nm if nm.endswith(".npz") else nm for nm in names]
    else:
        files = sorted([f for f in os.listdir(patch_dir) if f.endswith(".npz")])
    if use_first_n > 0:
        files = files[:min(use_first_n, len(files))]
    return files

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--t_small", type=int, default=20)
    ap.add_argument("--use_first_n", type=int, default=20, help="take first N (sorted) files")
    ap.add_argument("--file_list", type=str, default="", help="txt with npz filenames (one per line) to force exact order")
    ap.add_argument("--seed_base", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed_base)

    files = select_files(args.patch_dir, args.file_list, args.use_first_n)
    assert files, "No .npz files found/selected"
    print(f"[INFO] Will process {len(files)} files in fixed order.")

    # probe channels
    probe = np.load(os.path.join(args.patch_dir, files[0]))
    Cc = probe["inputs"].shape[0]; Ct = probe["target"].shape[0]

    # model
    model = UNetSmall(in_ch=Cc + Ct, out_ch=Ct, base_ch=args.base_ch).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True); model.eval()

    # schedule
    betas = cosine_beta_schedule(args.T).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    for i, fname in enumerate(files):
        p = os.path.join(args.patch_dir, fname)
        x_cond, x_gt, mask, _, _ = load_npz_as_tensors(p, device)
        H, W = x_gt.size(2), x_gt.size(3)
        print(f"[{i+1:03d}/{len(files)}] {fname}  HxW={H}x{W}")

        # t=0 identity
        t0 = torch.zeros((1,), dtype=torch.long, device=device)
        x_t0 = x_gt.clone()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            v0 = model(torch.cat([x_t0, x_cond], dim=1), t0)
        a0 = alpha_bar[t0]
        x0_hat_t0, _ = v_to_x0_eps(x_t0, v0, a0)
        x0_hat_t0 = torch.clamp(x0_hat_t0, 0.0, 1.0)

        # one-step at small t
        t_small = max(1, min(args.t_small, args.T - 1))
        t_idx = torch.full((1,), t_small, dtype=torch.long, device=device)
        a_t = alpha_bar[t_idx]
        # synthesize x_t with fixed seed
        noise = torch.randn_like(x_gt)
        x_t = torch.sqrt(a_t).view(-1,1,1,1) * x_gt + torch.sqrt(1 - a_t).view(-1,1,1,1) * noise
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            v = model(torch.cat([x_t, x_cond], dim=1), t_idx)
        x0_hat, _ = v_to_x0_eps(x_t, v, a_t)
        x0_hat = torch.clamp(x0_hat, 0.0, 1.0)

        # metrics
        mae = masked_mae(x0_hat, x_gt, mask); mse = masked_mse(x0_hat, x_gt, mask)
        print(f"   [t=0] MAE~0 | [one-step@t={t_small}] MAE={mae:.6f} MSE={mse:.6f}")

        # build six tiles (no text)
        s1_rgb = s1_preview_from_cond(x_cond.squeeze(0))
        gt_true, gt_cir   = to_true_cir_from_t4(x_gt.squeeze(0)[:4])
        pr_true, pr_cir   = to_true_cir_from_t4(x0_hat.squeeze(0)[:4])
        err_map = np.abs(x0_hat.squeeze(0).detach().cpu().numpy() - x_gt.squeeze(0).detach().cpu().numpy()).mean(axis=0)
        err_u8  = percentile_stretch_uint8(err_map); err_rgb = np.dstack([err_u8, err_u8, err_u8])

        panel = tile2x3(s1_rgb, gt_true, pr_true, gt_cir, pr_cir, err_rgb)
        panel.save(os.path.join(args.out_dir, f"{i:03d}_panel.png"))

    print(f"[DONE] Panels saved to: {args.out_dir}")
if __name__ == "__main__":
    main()
