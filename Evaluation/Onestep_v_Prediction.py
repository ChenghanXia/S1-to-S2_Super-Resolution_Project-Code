import os, math, argparse, numpy as np
from typing import Optional
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
    """
    Small UNet that consumes [x_t, cond] + integer timestep map, predicts **v** (C_tgt channels).
    """
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 96):
        super().__init__()
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=False),
            )
        # Encoder
        self.inc   = nn.Sequential(nn.Conv2d(in_ch + 1, base_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.down1 = nn.Sequential(conv_block(base_ch,   base_ch*2), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(conv_block(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(conv_block(base_ch*4, base_ch*8), nn.MaxPool2d(2))
        # Decoder
        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = conv_block(base_ch*8, base_ch*4)
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = conv_block(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = conv_block(base_ch*2, base_ch)
        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond, t_idx):
        B, _, H, W = xt_and_cond.shape
        t_map = t_idx.view(B, 1, 1, 1).float().repeat(1, 1, H, W)  # integer timestep channel
        x = torch.cat([xt_and_cond, t_map], dim=1)
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        u3 = self.up3(e4); d3 = torch.cat([u3, e3], dim=1); d3 = self.conv3(d3)
        u2 = self.up2(d3); d2 = torch.cat([u2, e2], dim=1); d2 = self.conv2(d2)
        u1 = self.up1(d2); d1 = torch.cat([u1, e1], dim=1); d1 = self.conv1(d1)
        return self.outc(d1)

# v <-> {x0, eps} conversions 
def v_to_x0_eps(x_t: torch.Tensor, v: torch.Tensor, alpha_bar_t: torch.Tensor):
    """
    Given (per eva_v17):
      x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
      v   = sqrt(ab) * eps - sqrt(1-ab) * x0
    Then:
      x0  = sqrt(ab) * x_t - sqrt(1-ab) * v
      eps = sqrt(1-ab) * x_t + sqrt(ab) * v
    """
    sab   = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    s1mab = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
    x0  = sab * x_t - s1mab * v
    eps = s1mab * x_t + sab * v
    return x0, eps

# metrics 
def masked_mae(pred, tgt, mask: Optional[torch.Tensor] = None) -> float:
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim == 3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt).abs()).sum(dim=(1,2,3)).mean()
    den = (w.sum(dim=(1,2,3)).clamp_min(1e-8) * pred.size(1)).mean()
    return (num / den).item()

def masked_mse(pred, tgt, mask: Optional[torch.Tensor] = None) -> float:
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim == 3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt) ** 2).sum(dim=(1,2,3)).mean()
    den = (w.sum(dim=(1,2,3)).clamp_min(1e-8) * pred.size(1)).mean()
    return (num / den).item()

# simple fixed-range viz 
def stretch_to_uint8_fixed(x_chw: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = x_chw.detach().cpu().numpy()
    C, H, W = x.shape
    y = np.empty((C, H, W), dtype=np.uint8)
    for c in range(C):
        yc = (x[c] - lo[c]) / (hi[c] - lo[c] + 1e-8)
        yc = np.clip(yc, 0, 1)
        y[c] = (yc * 255.0).astype(np.uint8)
    return y

def per_image_lo_hi_from_gt(xgt_chw: torch.Tensor, q_low=2.0, q_high=98.0):
    arr = xgt_chw.detach().cpu().numpy()
    C = arr.shape[0]
    lo = np.zeros(C, dtype=np.float32)
    hi = np.ones(C, dtype=np.float32)
    for c in range(C):
        v = arr[c].reshape(-1)
        lo[c] = np.percentile(v, q_low)
        hi[c] = np.percentile(v, q_high)
        if hi[c] - lo[c] < 1e-6:
            hi[c] = lo[c] + 1.0
    return lo, hi

def to_rgb_panels_fixed(t4_u8: np.ndarray):
    # t4_u8: (4,H,W) uint8 [B2,B3,B4,B8]
    B2, B3, B4, B8 = t4_u8[0], t4_u8[1], t4_u8[2], t4_u8[3]
    true_rgb = np.dstack([B4, B3, B2])
    cir_rgb  = np.dstack([B8, B4, B3])
    return true_rgb, cir_rgb

# main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--t_small", type=int, default=20, help="small t for one-step denoise")
    ap.add_argument("--use_first_n", type=int, default=1, help="how many npz files to check (>=1)")
    ap.add_argument("--viz_q_low", type=float, default=2.0)
    ap.add_argument("--viz_q_high", type=float, default=98.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_files = sorted([f for f in os.listdir(args.patch_dir) if f.endswith(".npz")])
    assert npz_files, "No .npz in patch_dir"

    # load one to probe channel counts
    probe = np.load(os.path.join(args.patch_dir, npz_files[0]))
    Cc = probe["inputs"].shape[0]
    Ct = probe["target"].shape[0]

    # model
    model = UNetSmall(in_ch=Cc + Ct, out_ch=Ct, base_ch=args.base_ch).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    # make it robust to various checkpoint dict wrappers
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # schedule
    betas = cosine_beta_schedule(args.T).to(dev)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    nfiles = max(1, min(args.use_first_n, len(npz_files)))

    for i in range(nfiles):
        npz_path = os.path.join(args.patch_dir, npz_files[i])
        d = np.load(npz_path)
        x_cond = torch.from_numpy(np.nan_to_num(d["inputs"].astype(np.float32))).unsqueeze(0).to(dev)  # (1,Cc,H,W)
        x_gt   = torch.from_numpy(np.nan_to_num(d["target"].astype(np.float32))).unsqueeze(0).to(dev)  # (1,Ct,H,W)
        mask   = None
        if "mask" in d:
            mask = torch.from_numpy(np.nan_to_num(d["mask"].astype(np.float32))).unsqueeze(0).to(dev)

        H, W = x_gt.size(2), x_gt.size(3)
        print(f"\n[INFO] Sample {i+1}/{nfiles}: {os.path.basename(npz_path)}  cond={Cc} tgt={Ct}  HxW={H}x{W}")

        # per-image viz ranges from GT
        lo, hi = per_image_lo_hi_from_gt(x_gt[0], q_low=args.viz_q_low, q_high=args.viz_q_high)

        # (A) t=0 identity (v-pred formulation) 
        with torch.no_grad():
            t0 = torch.zeros((1,), dtype=torch.long, device=dev)
            x_t0 = x_gt.clone()
            # predict v (not really used at t=0, but for completeness)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                v0 = model(torch.cat([x_t0, x_cond], dim=1), t0)
            # reconstruct x0 via v_to_x0_eps
            a0 = alpha_bar[t0]
            x0_hat_t0, _ = v_to_x0_eps(x_t0, v0, a0)
            x0_hat_t0 = torch.clamp(x0_hat_t0, 0.0, 1.0)

            mae0 = masked_mae(x0_hat_t0, x_gt, mask)
            mse0 = masked_mse(x0_hat_t0, x_gt, mask)
            print(f"[t=0 identity] MAE={mae0:.6f}  MSE={mse0:.6f}  (should be ~0.0)")

            # save quick previews
            u8 = stretch_to_uint8_fixed(x0_hat_t0[0].cpu(), lo, hi)
            true_rgb, cir_rgb = to_rgb_panels_fixed(u8)
            Image.fromarray(true_rgb).save(os.path.join(args.out_dir, f"{i:03d}_t0_true.png"))
            Image.fromarray(cir_rgb ).save(os.path.join(args.out_dir, f"{i:03d}_t0_cir.png"))

        # (B) one-step at small t (v-pred) 
        with torch.no_grad():
            t_small = max(1, min(args.t_small, args.T - 1))
            t_idx = torch.full((1,), t_small, dtype=torch.long, device=dev)
            a_t = alpha_bar[t_idx]

            # forward diffuse GT to x_t
            noise = torch.randn_like(x_gt)
            x_t = torch.sqrt(a_t).view(-1,1,1,1) * x_gt + torch.sqrt(1 - a_t).view(-1,1,1,1) * noise

            # predict v and reconstruct x0
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                v = model(torch.cat([x_t, x_cond], dim=1), t_idx)
            x0_hat, eps_hat = v_to_x0_eps(x_t, v, a_t)
            x0_hat = torch.clamp(x0_hat, 0.0, 1.0)

            mae = masked_mae(x0_hat, x_gt, mask)
            mse = masked_mse(x0_hat, x_gt, mask)
            print(f"[one-step@t={t_small}] MAE={mae:.6f}  MSE={mse:.6f}")

            # save previews: pred & gt
            u8p = stretch_to_uint8_fixed(x0_hat[0].cpu(), lo, hi)
            u8g = stretch_to_uint8_fixed(x_gt[0].cpu(), lo, hi)
            pr_true, pr_cir = to_rgb_panels_fixed(u8p)
            gt_true, gt_cir = to_rgb_panels_fixed(u8g)
            Image.fromarray(pr_true).save(os.path.join(args.out_dir, f"{i:03d}_pred_true.png"))
            Image.fromarray(pr_cir ).save(os.path.join(args.out_dir, f"{i:03d}_pred_cir.png"))
            Image.fromarray(gt_true).save(os.path.join(args.out_dir, f"{i:03d}_gt_true.png"))
            Image.fromarray(gt_cir ).save(os.path.join(args.out_dir, f"{i:03d}_gt_cir.png"))

    print(f"\n[INFO] Done. Debug images saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
