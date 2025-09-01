import os, math, argparse, numpy as np
from PIL import Image
import torch
import torch.nn as nn

# schedule
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clip(betas, 1e-5, 0.999)
    return betas.float()

# tiny U-Net  
class UNetSmall(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 96):
        super().__init__()
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=False),
            )
        self.inc   = nn.Sequential(nn.Conv2d(in_ch + 1, base_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.down1 = nn.Sequential(conv_block(base_ch,   base_ch*2), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(conv_block(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(conv_block(base_ch*4, base_ch*8), nn.MaxPool2d(2))

        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = conv_block(base_ch*8, base_ch*4)

        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = conv_block(base_ch*4, base_ch*2)

        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = conv_block(base_ch*2, base_ch)

        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond, t_idx):
        B, _, H, W = xt_and_cond.shape
        t_map = t_idx.view(B, 1, 1, 1).float().repeat(1, 1, H, W)
        x = torch.cat([xt_and_cond, t_map], dim=1)
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        d3 = self.up3(e4); d3 = torch.cat([d3, e3], dim=1); d3 = self.conv3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.conv2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.conv1(d1)
        return self.outc(d1)

# viz helpers 
def _stretch_to_uint8(x: torch.Tensor):
    x = x.detach().cpu().numpy()
    p2, p98 = np.percentile(x, [2, 98])
    if p98 - p2 < 1e-6: p98 = p2 + 1.0
    x = np.clip((x - p2) / (p98 - p2), 0, 1)
    return (x * 255).astype(np.uint8)

def save_rgb_triplet(t4: torch.Tensor, out_path_true: str, out_path_cir: str):
    # t4: (4,H,W) -> [B2,B3,B4,B8]
    B2, B3, B4, B8 = t4[0], t4[1], t4[2], t4[3]
    true_rgb = np.dstack([_stretch_to_uint8(B4), _stretch_to_uint8(B3), _stretch_to_uint8(B2)])
    cir_rgb  = np.dstack([_stretch_to_uint8(B8), _stretch_to_uint8(B4), _stretch_to_uint8(B3)])
    Image.fromarray(true_rgb).save(out_path_true)
    Image.fromarray(cir_rgb).save(out_path_cir)

def masked_mae(pred, tgt, mask=None):
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim == 3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt).abs()).sum()
    den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

def masked_mse(pred, tgt, mask=None):
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim == 3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt) ** 2).sum()
    den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

# main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--t_small", type=int, default=20, help="small t for one-step denoise")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pick a real npz (skip folders)
    npz_files = sorted([f for f in os.listdir(args.patch_dir) if f.endswith(".npz")])
    assert npz_files, "No .npz in patch_dir"
    npz_path = os.path.join(args.patch_dir, npz_files[0])
    d = np.load(npz_path)

    x_cond = torch.from_numpy(np.nan_to_num(d["inputs"].astype(np.float32))).unsqueeze(0).to(dev)  # (1,Cc,H,W)
    x_gt   = torch.from_numpy(np.nan_to_num(d["target"].astype(np.float32))).unsqueeze(0).to(dev)  # (1,Ct,H,W)
    mask   = None
    if "mask" in d:
        mask = torch.from_numpy(np.nan_to_num(d["mask"].astype(np.float32))).unsqueeze(0).to(dev)  # (1,1,H,W) after unsqueeze

    Cc, Ct = x_cond.size(1), x_gt.size(1)
    print(f"[INFO] Using sample: {os.path.basename(npz_path)}  cond={Cc} tgt={Ct}  HxW={x_gt.size(2)}x{x_gt.size(3)}")

    # model
    model = UNetSmall(in_ch=Cc + Ct, out_ch=Ct, base_ch=args.base_ch).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(state, strict=True)
    model.eval()

    # schedule tensors
    betas = cosine_beta_schedule(args.T).to(dev)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # (A) t = 0 identity check 
    with torch.no_grad():
        t0 = torch.zeros((1,), dtype=torch.long, device=dev)
        # at t=0: x_t == x0, reconstruction formula gives exactly x0 regardless of eps
        x_t0 = x_gt.clone()
        # eps prediction (not used effectively at t=0, but compute for completeness)
        eps0 = model(torch.cat([x_t0, x_cond], dim=1), t0)
        x0_hat_t0 = x_t0  # since sqrt(1-alpha_bar[0])==0, sqrt(alpha_bar[0])==1

        mae0 = masked_mae(x0_hat_t0, x_gt, mask)
        mse0 = masked_mse(x0_hat_t0, x_gt, mask)
        print(f"[t=0 identity] MAE={mae0:.6f}  MSE={mse0:.6f}  (should be ~0.0)")

        # save quick previews
        save_rgb_triplet(x0_hat_t0[0].cpu(), os.path.join(args.out_dir, "t0_true.png"),
                         os.path.join(args.out_dir, "t0_cir.png"))

    # (B) one-step denoise at small t 
    with torch.no_grad():
        t_small = max(1, min(args.t_small, args.T - 1))
        t_idx = torch.full((1,), t_small, dtype=torch.long, device=dev)
        a_t = alpha_bar[t_idx].view(-1, 1, 1, 1)
        # synthesize x_t from GT
        noise = torch.randn_like(x_gt)
        x_t = torch.sqrt(a_t) * x_gt + torch.sqrt(1 - a_t) * noise

        eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
        x0_hat = (x_t - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
        x0_hat = torch.clamp(x0_hat, 0.0, 1.0)

        mae = masked_mae(x0_hat, x_gt, mask)
        mse = masked_mse(x0_hat, x_gt, mask)
        print(f"[one-step@t={t_small}] MAE={mae:.6f}  MSE={mse:.6f}")

        # save previews for visual inspection
        save_rgb_triplet(x0_hat[0].cpu(), os.path.join(args.out_dir, "pred_true.png"),
                         os.path.join(args.out_dir, "pred_cir.png"))
        save_rgb_triplet(x_gt[0].cpu(), os.path.join(args.out_dir, "gt_true.png"),
                         os.path.join(args.out_dir, "gt_cir.png"))

    print(f"[INFO] Debug images saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
