import os, math, argparse, csv, random
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn

# ---------------------- schedule ----------------------
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clip(betas, 1e-5, 0.999)
    return betas.float()

# ---------------------- model ----------------------
class UNetSmall(nn.Module):
    """UNet used in training; time index injected as a scalar map channel."""
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 96):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=False),
            )
        self.inc   = nn.Sequential(nn.Conv2d(in_ch + 1, base_ch, 3, padding=1), nn.ReLU(inplace=False))
        self.down1 = nn.Sequential(block(base_ch,   base_ch*2), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(block(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.down3 = nn.Sequential(block(base_ch*4, base_ch*8), nn.MaxPool2d(2))

        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = block(base_ch*8, base_ch*4)
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = block(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = block(base_ch*2, base_ch)
        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        B, _, H, W = xt_and_cond.shape
        t_map = t_idx.view(B,1,1,1).float().repeat(1,1,H,W)
        x = torch.cat([xt_and_cond, t_map], dim=1)
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        d3 = self.up3(e4); d3 = torch.cat([d3,e3], dim=1); d3 = self.conv3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2,e2], dim=1); d2 = self.conv2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1,e1], dim=1); d1 = self.conv1(d1)
        return self.outc(d1)

# ---------------------- viz helpers ----------------------
def _percentile_stretch_uint8(x: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(x, [2, 98])
    if p98 - p2 < 1e-6: p98 = p2 + 1.0
    x = np.clip((x - p2) / (p98 - p2), 0, 1)
    return (x * 255).astype(np.uint8)

def s2_true_cir_from_t4(t4: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # t4: (4,H,W) -> [B2,B3,B4,B8]
    arr = t4.detach().cpu().numpy()
    B2,B3,B4,B8 = arr[0],arr[1],arr[2],arr[3]
    true_rgb = np.dstack([
        _percentile_stretch_uint8(B4),
        _percentile_stretch_uint8(B3),
        _percentile_stretch_uint8(B2),
    ])
    cir_rgb = np.dstack([
        _percentile_stretch_uint8(B8),
        _percentile_stretch_uint8(B4),
        _percentile_stretch_uint8(B3),
    ])
    return true_rgb, cir_rgb

def s1_preview_from_cond(x_cond: torch.Tensor) -> np.ndarray:
    # heuristic preview from S1 VV/VH
    arr = x_cond.detach().cpu().numpy()
    def st(u): return _percentile_stretch_uint8(u)
    if arr.shape[0] == 2:
        VV, VH = arr[0], arr[1]
        rgb = np.dstack([st(VV), st(VH), st(VV)])
    elif arr.shape[0] == 1:
        ch = st(arr[0]); rgb = np.dstack([ch,ch,ch])
    else:
        rgb = np.dstack([st(arr[i]) for i in range(3)])
    return rgb

def save_panel(pred: torch.Tensor, gt: Optional[torch.Tensor], x_cond: torch.Tensor,
               mask: Optional[torch.Tensor], out_path: str,
               title: str = "",
               zoom: int = 0, zoom_k: int = 0):
    """Create a side-by-side panel: S1 preview | GT True/CIR | Pred True/CIR | Abs-Error heatmaps.
    Optionally include zoom_k crops of size zoom x zoom (select top-k error regions if GT exists).
    """
    # Build previews
    s1_rgb = s1_preview_from_cond(x_cond.squeeze(0))
    if pred.size(1) >= 4:
        pred_true, pred_cir = s2_true_cir_from_t4(pred.squeeze(0)[:4])
    else:
        # fallback: replicate channels
        ch = _percentile_stretch_uint8(pred.squeeze(0)[0].detach().cpu().numpy())
        pred_true = np.dstack([ch,ch,ch]); pred_cir = pred_true.copy()
    if gt is not None:
        gt_true, gt_cir = s2_true_cir_from_t4(gt.squeeze(0)[:4])
        abs_err = np.abs(pred.squeeze(0).detach().cpu().numpy() - gt.squeeze(0).detach().cpu().numpy())
        # simple combined error (mean over bands)
        err_map = abs_err.mean(axis=0)
        err_img = _percentile_stretch_uint8(err_map)
        err_img = np.dstack([err_img, err_img, err_img])
    else:
        gt_true = gt_cir = None
        err_img = None

    # Layout tiles
    tiles = [ ("S1 preview", s1_rgb) ]
    if gt_true is not None:
        tiles += [("GT TrueColor", gt_true), ("GT CIR", gt_cir)]
    tiles += [("Pred TrueColor", pred_true), ("Pred CIR", pred_cir)]
    if err_img is not None:
        tiles += [("Abs-Error (mean over bands)", err_img)]

    # Tile arrangement: 2 rows
    Wt = max(t[1].shape[1] for t in tiles)
    Ht = max(t[1].shape[0] for t in tiles)
    scale_target = 1024  # width of each tile for high-res panel
    def resize(im):
        h,w = im.shape[:2]
        if w == scale_target: return im
        new_h = int(h * (scale_target / w))
        return np.array(Image.fromarray(im).resize((scale_target,new_h), Image.BILINEAR))
    tiles = [(name, resize(img)) for name,img in tiles]

    # compute grid size
    cols = 3 if gt_true is not None else 3
    # with GT: S1 | GT True | Pred True  (row1)
    #         GT CIR | Pred CIR | Error  (row2)
    if gt_true is not None:
        row1 = [tiles[0][1], tiles[1][1], tiles[3][1]]
        row2 = [tiles[2][1], tiles[4][1], tiles[5][1]]
        labels = [tiles[0][0], tiles[1][0], tiles[3][0], tiles[2][0], tiles[4][0], tiles[5][0]]
    else:
        row1 = [tiles[0][1], tiles[1][1], tiles[2][1]]  # S1 | Pred True | Pred CIR
        row2 = None
        labels = [tiles[0][0], tiles[1][0], tiles[2][0]]

    def hstack(images):
        h = max(im.shape[0] for im in images)
        canv = Image.new('RGB', (sum(im.shape[1] for im in images), h), (255,255,255))
        x = 0
        for im in images:
            pim = Image.fromarray(im)
            if pim.size[1] != h:
                pim = pim.resize((pim.size[0], h), Image.BILINEAR)
            canv.paste(pim, (x,0)); x += pim.size[0]
        return np.array(canv)

    rows = [hstack(row1)]
    if row2 is not None:
        rows.append(hstack(row2))
    panel = rows[0] if len(rows)==1 else np.array(Image.fromarray(rows[0]).resize((rows[0].shape[1], rows[0].shape[0])));
    if row2 is not None:
        panel = np.vstack([rows[0], rows[1]])

    # Add titles
    canvas = Image.fromarray(panel)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    # write top title
    if title:
        draw.text((10, 5), title, fill=(0,0,0), font=font)
    # write sub-labels roughly above tiles
    # skip precise placement for brevity

    canvas.save(out_path)

    # Optional zoom crops from top-k error regions
    if zoom > 0 and zoom_k > 0 and gt is not None:
        err = np.abs(pred.squeeze(0).detach().cpu().numpy() - gt.squeeze(0).detach().cpu().numpy()).mean(axis=0)
        H,W = err.shape
        # find top-k windows by mean error
        scores = []
        for _ in range(1000):  # sample windows (random sampling to keep it fast)
            i = random.randint(0, max(0,H-zoom)); j = random.randint(0, max(0,W-zoom))
            scores.append((err[i:i+zoom, j:j+zoom].mean(), i, j))
        scores.sort(reverse=True)
        for k in range(min(zoom_k, len(scores))):
            _, i, j = scores[k]
            def crop_and_save(arr, name):
                im = Image.fromarray(arr)
                crop = im.crop((j,i,j+zoom,i+zoom)).resize((zoom*2, zoom*2), Image.NEAREST)
                crop.save(out_path.replace('.png', f'_{name}_zoom{k}.png'))
            crop_and_save(s1_preview_from_cond(x_cond.squeeze(0)), 's1')
            if gt_true is not None:
                crop_and_save(s2_true_cir_from_t4(gt.squeeze(0)[:4])[0], 'gt_true')
            crop_and_save(s2_true_cir_from_t4(pred.squeeze(0)[:4])[0], 'pred_true')

# ---------------------- metrics ----------------------
def masked_mae(pred, tgt, mask=None) -> float:
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim==3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt).abs()).sum()
    den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

def masked_mse(pred, tgt, mask=None) -> float:
    if mask is None:
        w = torch.ones_like(pred[:, :1])
    else:
        w = (mask.unsqueeze(1) if mask.ndim==3 else mask).float().to(pred.device)
        w = (w > 0).float()
    num = (w * (pred - tgt)**2).sum()
    den = w.sum() * pred.size(1)
    return (num / (den + 1e-8)).item()

def psnr(pred, tgt, mask=None) -> float:
    m = masked_mse(pred, tgt, mask)
    if m <= 1e-12: return 99.0
    return 10.0 * math.log10(1.0 / m)

def ssim_simple(pred, tgt, C1=0.01**2, C2=0.03**2) -> float:
    # very simple global SSIM-like index (not windowed)
    mu_x = pred.mean().item(); mu_y = tgt.mean().item()
    vx = pred.var().item();    vy = tgt.var().item()
    cxy = ((pred - pred.mean()) * (tgt - tgt.mean())).mean().item()
    return ((2*mu_x*mu_y + C1) * (2*cxy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (vx + vy + C2) + 1e-8)

def sam(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    """Spectral Angle Mapper (radians) averaged over masked pixels."""
    # pred,tgt: (1,C,H,W)
    p = pred.squeeze(0); g = tgt.squeeze(0)
    if mask is not None:
        m = (mask.squeeze(0) > 0).bool()
    else:
        m = torch.ones_like(p[0], dtype=torch.bool)
    p = p[:, m]; g = g[:, m]
    dot = (p * g).sum(dim=0)
    p_norm = torch.clamp(p.norm(dim=0), min=1e-8)
    g_norm = torch.clamp(g.norm(dim=0), min=1e-8)
    cos = torch.clamp(dot / (p_norm * g_norm), -1.0, 1.0)
    angle = torch.arccos(cos)
    return angle.mean().item()

def ergas(pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor], scale_ratio: float = 4.0) -> float:
    """ERGAS metric (lower is better). scale_ratio is HR/LR (e.g., 10m/40m = 0.25 => use 4.0)."""
    C = pred.size(1)
    rmse_sq = 0.0
    for c in range(C):
        p = pred[:,c:c+1]; g = tgt[:,c:c+1]
        mse_c = masked_mse(p, g, mask)
        rmse_c = math.sqrt(max(mse_c, 0.0))
        mean_c = g.mean().item() + 1e-8
        rmse_sq += (rmse_c / mean_c) ** 2
    return 100.0 * (1.0/ C * rmse_sq) ** 0.5 * (1.0 * scale_ratio)

# ---------------------- I/O ----------------------
def load_npz_as_tensors(path: str, device: torch.device):
    d = np.load(path)
    x_cond = torch.from_numpy(np.nan_to_num(d["inputs"].astype(np.float32))).unsqueeze(0).to(device)
    x_gt   = torch.from_numpy(np.nan_to_num(d["target"].astype(np.float32))).unsqueeze(0).to(device)
    mask   = None
    if "mask" in d:
        mask = torch.from_numpy(np.nan_to_num(d["mask"].astype(np.float32))).unsqueeze(0).to(device)
    cloud = None
    for key in ["cloud_mask","s2_cloud","qa60"]:
        if key in d:
            cloud = np.nan_to_num(d[key].astype(np.float32))
            break
    if cloud is None and "s2_cloud_prob" in d:
        prob = np.nan_to_num(d["s2_cloud_prob"].astype(np.float32))
        cloud = (prob >= 0.5).astype(np.float32)
    return x_cond, x_gt, mask, x_cond.size(1), x_gt.size(1), cloud

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# ---------------------- core evals ----------------------
@torch.no_grad()
def ddpm_ddim_generate(model, x_cond, alpha_bar, t_start=200, steps=20):
    dev = x_cond.device
    Ct = model.outc.out_channels
    # start from a synthetic x_t seeded from unknown x0 (we use standard DDIM backward with placeholder gt-free)
    x_t = torch.randn((1, Ct, x_cond.size(2), x_cond.size(3)), device=dev)
    ts = torch.linspace(t_start, 0, steps+1, dtype=torch.long, device=dev)
    for i in range(steps):
        t_cur  = ts[i].view(1)
        t_next = ts[i+1].view(1)
        a_cur  = alpha_bar[t_cur].view(-1,1,1,1)
        a_next = alpha_bar[t_next].view(-1,1,1,1)
        eps = model(torch.cat([x_t, x_cond], dim=1), t_cur)
        x0_hat = (x_t - torch.sqrt(1 - a_cur)*eps) / torch.sqrt(a_cur + 1e-8)
        x_t = torch.sqrt(a_next)*x0_hat + torch.sqrt(1 - a_next)*eps
    x0_hat_final = torch.clamp(x0_hat, 0.0, 1.0)
    return x0_hat_final

@torch.no_grad()
def ddim_multistep_eval(model, x_gt, x_cond, alpha_bar, mask, t_start=200, steps=20):
    t_start = max(1, min(int(t_start), len(alpha_bar)-1))
    dev = x_gt.device
    t_idx = torch.full((1,), t_start, dtype=torch.long, device=dev)
    a_t = alpha_bar[t_idx].view(-1,1,1,1)
    noise = torch.randn_like(x_gt)
    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*noise
    ts = torch.linspace(t_start, 0, steps+1, dtype=torch.long, device=dev)
    for i in range(steps):
        t_cur  = ts[i].view(1)
        t_next = ts[i+1].view(1)
        a_cur  = alpha_bar[t_cur].view(-1,1,1,1)
        a_next = alpha_bar[t_next].view(-1,1,1,1)
        eps = model(torch.cat([x_t, x_cond], dim=1), t_cur)
        x0_hat = (x_t - torch.sqrt(1 - a_cur)*eps) / torch.sqrt(a_cur + 1e-8)
        x_t = torch.sqrt(a_next)*x0_hat + torch.sqrt(1 - a_next)*eps
    x0_hat_final = torch.clamp(x0_hat, 0.0, 1.0)
    mae = masked_mae(x0_hat_final, x_gt, mask)
    mse = masked_mse(x0_hat_final, x_gt, mask)
    return mae, mse, x0_hat_final

@torch.no_grad()
def eps_diagnostics(model, x_gt, x_cond, alpha_bar, t_small: int):
    dev = x_gt.device
    t_small = max(1, min(int(t_small), len(alpha_bar)-1))
    t_idx = torch.full((1,), t_small, dtype=torch.long, device=dev)
    a_t = alpha_bar[t_idx].view(-1,1,1,1)
    z   = torch.randn_like(x_gt)
    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*z
    pred_eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
    mse = torch.mean((pred_eps - z)**2).item()
    cos = torch.sum(pred_eps*z).item() / (
        torch.sqrt(torch.sum(pred_eps**2)).item() * torch.sqrt(torch.sum(z**2)).item() + 1e-8
    )
    return mse, cos

@torch.no_grad()
def one_step_recon(model, x_gt, x_cond, alpha_bar, mask, t_small: int, rng_seed: Optional[int] = None):
    if rng_seed is not None: torch.manual_seed(rng_seed)
    dev = x_gt.device
    t_small = max(1, min(int(t_small), len(alpha_bar)-1))
    t_idx = torch.full((1,), t_small, dtype=torch.long, device=dev)
    a_t = alpha_bar[t_idx].view(-1,1,1,1)
    noise = torch.randn_like(x_gt)
    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*noise
    eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
    x0_hat = (x_t - torch.sqrt(1 - a_t)*eps) / torch.sqrt(a_t + 1e-8)
    x0_hat = torch.clamp(x0_hat, 0.0, 1.0)
    mae = masked_mae(x0_hat, x_gt, mask)
    mse = masked_mse(x0_hat, x_gt, mask)
    return mae, mse, x0_hat

# ---------------------- utils ----------------------
def cloud_fraction(cloud_arr: Optional[np.ndarray]) -> Optional[float]:
    if cloud_arr is None: return None
    total = cloud_arr.size
    if total == 0: return None
    return float((cloud_arr > 0.5).sum()) / total

# ---------------------- main (batch) ----------------------
def main():
    ap = argparse.ArgumentParser("Batch eval & viz suite")
    ap.add_argument("--mode", required=True, choices=["tsweep","ddim","eps","seed_stats","per_band","ablate","cloudy_viz","night_demo"]) 
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--max_files", type=int, default=0, help="0 means ALL .npz in patch_dir")
    ap.add_argument("--save_viz_n", type=int, default=6, help="save previews for first N samples")
    # DDIM
    ap.add_argument("--t_start", type=int, default=200)
    ap.add_argument("--ddim_steps", type=int, default=20)
    # t-sweep / eps / seed_stats / per_band / ablate
    ap.add_argument("--t_small", type=int, default=20)
    ap.add_argument("--t_values", type=int, nargs="*", default=[5,10,20,40,80,160])
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--seed_base", type=int, default=1234)
    # cloudy / viz
    ap.add_argument("--select_top_cloud", type=int, default=12)
    ap.add_argument("--zoom", type=int, default=256)
    ap.add_argument("--zoom_k", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File list
    files = sorted([f for f in os.listdir(args.patch_dir) if f.endswith(".npz")])
    assert files, "No .npz found"
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"[INFO] Evaluating {len(files)} files")

    # Warm up first file to build the model
    x_cond0, x_gt0, mask0, Cc0, Ct0, _ = load_npz_as_tensors(os.path.join(args.patch_dir, files[0]), device)
    model = UNetSmall(in_ch=Cc0 + Ct0, out_ch=Ct0, base_ch=args.base_ch).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    betas = cosine_beta_schedule(args.T).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    viz_dir = os.path.join(args.out_dir, "previews"); ensure_dir(viz_dir)

    # ---------------- MODE: TSWEEP ----------------
    if args.mode == "tsweep":
        csv_path = os.path.join(args.out_dir, "tsweep.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t","MAE","MSE"])
            for i, fname in enumerate(files):
                x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                assert Cc==Cc0 and Ct==Ct0, "Channel mismatch across files"
                # fixed noise per-file for all t
                torch.manual_seed(args.seed_base)
                z = torch.randn_like(x_gt)
                for t in args.t_values:
                    t_small = max(1, min(int(t), args.T-1))
                    t_idx = torch.full((1,), t_small, dtype=torch.long, device=device)
                    a_t = alpha_bar[t_idx].view(-1,1,1,1)
                    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*z
                    eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
                    x0_hat = (x_t - torch.sqrt(1 - a_t)*eps) / torch.sqrt(a_t + 1e-8)
                    x0_hat = torch.clamp(x0_hat, 0.0, 1.0)
                    mae = masked_mae(x0_hat, x_gt, mask)
                    mse = masked_mse(x0_hat, x_gt, mask)
                    w.writerow([fname, t_small, f"{mae:.6f}", f"{mse:.6f}"]) 
                if i < args.save_viz_n:
                    # save a panel for the middle t
                    mid_t = args.t_values[len(args.t_values)//2]
                    t_idx = torch.full((1,), mid_t, dtype=torch.long, device=device)
                    a_t = alpha_bar[t_idx].view(-1,1,1,1)
                    torch.manual_seed(args.seed_base)
                    z = torch.randn_like(x_gt)
                    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*z
                    eps = model(torch.cat([x_t, x_cond], dim=1), t_idx)
                    x0_hat = (x_t - torch.sqrt(1 - a_t)*eps) / torch.sqrt(a_t + 1e-8)
                    x0_hat = torch.clamp(x0_hat, 0.0, 1.0)
                    save_panel(x0_hat, x_gt, x_cond, mask,
                               os.path.join(viz_dir, f"{i:03d}_tsweep_t{mid_t}.png"),
                               title=f"t-sweep middle t={mid_t}")
        print("[DONE] TSWEEP")

    # ---------------- MODE: DDIM ----------------
    elif args.mode == "ddim":
        csv_path = os.path.join(args.out_dir, "ddim_metrics.csv")
        maes, mses, psnrs, sams, ergases = [], [], [], [], []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_start","ddim_steps","MAE","MSE","PSNR","SAM(rad)","ERGAS"]) 
            for i, fname in enumerate(files):
                x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                mae, mse, x0 = ddim_multistep_eval(model, x_gt, x_cond, alpha_bar, mask,
                                                   t_start=args.t_start, steps=args.ddim_steps)
                psn = psnr(x0, x_gt, mask); sa = sam(x0, x_gt, mask); eg = ergas(x0, x_gt, mask)
                maes.append(mae); mses.append(mse); psnrs.append(psn); sams.append(sa); ergases.append(eg)
                w.writerow([fname, args.t_start, args.ddim_steps, f"{mae:.6f}", f"{mse:.6f}", f"{psn:.3f}", f"{sa:.4f}", f"{eg:.2f}"])
                if i < args.save_viz_n:
                    save_panel(x0, x_gt, x_cond, mask,
                               os.path.join(viz_dir, f"{i:03d}_ddim_panel.png"),
                               title=f"DDIM t_start={args.t_start}, steps={args.ddim_steps}",
                               zoom=args.zoom, zoom_k=args.zoom_k)
        # summary
        def mstd(a):
            t = torch.tensor(a); return t.mean().item(), t.std(unbiased=False).item()
        with open(os.path.join(args.out_dir, "ddim_summary.txt"), "w") as f:
            f.write(f"files: {len(files)}  t_start: {args.t_start}  steps: {args.ddim_steps}\n")
            f.write(f"MAE mean/std:  {mstd(maes)[0]:.6f} / {mstd(maes)[1]:.6f}\n")
            f.write(f"MSE mean/std:  {mstd(mses)[0]:.6f} / {mstd(mses)[1]:.6f}\n")
            f.write(f"PSNR mean/std: {mstd(psnrs)[0]:.3f} / {mstd(psnrs)[1]:.3f}\n")
            f.write(f"SAM  mean/std: {mstd(sams)[0]:.4f} / {mstd(sams)[1]:.4f}\n")
            f.write(f"ERGAS mean/std:{mstd(ergases)[0]:.2f} / {mstd(ergases)[1]:.2f}\n")
        print("[DONE] DDIM")

    # ---------------- MODE: EPS ----------------
    elif args.mode == "eps":
        csv_path = os.path.join(args.out_dir, "eps_diag.csv")
        mses, coses = [], []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_small","eps_MSE","cosine"]) 
            for fname in files:
                x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                mse, cos = eps_diagnostics(model, x_gt, x_cond, alpha_bar, t_small=args.t_small)
                mses.append(mse); coses.append(cos)
                w.writerow([fname, args.t_small, f"{mse:.6f}", f"{cos:.6f}"]) 
        def mstd(a): t = torch.tensor(a); return t.mean().item(), t.std(unbiased=False).item()
        with open(os.path.join(args.out_dir, "eps_summary.txt"), "w") as f:
            f.write(f"files: {len(files)}  t_small: {args.t_small}\n")
            f.write(f"eps_MSE mean/std: {mstd(mses)[0]:.6f} / {mstd(mses)[1]:.6f}\n")
            f.write(f"cosine  mean/std: {mstd(coses)[0]:.6f} / {mstd(coses)[1]:.6f}\n")
        print("[DONE] EPS")

    # ---------------- MODE: SEED_STATS ----------------
    elif args.mode == "seed_stats":
        csv_path = os.path.join(args.out_dir, "seed_stats.csv")
        mae_means, mae_stds, mse_means, mse_stds = [], [], [], []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_small","n_seeds","MAE_mean","MAE_std","MSE_mean","MSE_std"]) 
            for fname in files:
                x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                maes, mses = [], []
                for s in range(args.n_seeds):
                    mae, mse, _ = one_step_recon(model, x_gt, x_cond, alpha_bar, mask,
                                                 t_small=args.t_small, rng_seed=args.seed_base + s)
                    maes.append(mae); mses.append(mse)
                maes_t = torch.tensor(maes); mses_t = torch.tensor(mses)
                mae_mu, mae_sd = maes_t.mean().item(), maes_t.std(unbiased=False).item()
                mse_mu, mse_sd = mses_t.mean().item(), mses_t.std(unbiased=False).item()
                mae_means.append(mae_mu); mae_stds.append(mae_sd)
                mse_means.append(mse_mu); mse_stds.append(mse_sd)
                w.writerow([fname, args.t_small, args.n_seeds,
                            f"{mae_mu:.6f}", f"{mae_sd:.6f}", f"{mse_mu:.6f}", f"{mse_sd:.6f}"]) 
        with open(os.path.join(args.out_dir, "seed_stats_summary.txt"), "w") as f:
            f.write(f"files: {len(files)}  t_small: {args.t_small}  n_seeds: {args.n_seeds}\n")
            f.write(f"MAE mean_of_means/std_of_means: {np.mean(mae_means):.6f} / {np.std(mae_means):.6f}\n")
            f.write(f"MSE mean_of_means/std_of_means: {np.mean(mse_means):.6f} / {np.std(mse_means):.6f}\n")
            f.write(f"Avg per-file MAE_std: {np.mean(mae_stds):.6f}\n")
            f.write(f"Avg per-file MSE_std: {np.mean(mse_stds):.6f}\n")
        print("[DONE] SEED_STATS")

    # ---------------- MODE: PER_BAND ----------------
    elif args.mode == "per_band":
        band_accum: Dict[int, List[float]] = {}
        csv_path = os.path.join(args.out_dir, "per_band_all.csv")
        wrote_header = False
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            for i, fname in enumerate(files):
                x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                mae, mse, x0 = one_step_recon(model, x_gt, x_cond, alpha_bar, mask, t_small=args.t_small)
                C = x_gt.size(1)
                if not wrote_header:
                    hdr = ["file"]
                    for b in range(C): hdr += [f"band{b}_MAE", f"band{b}_MSE", f"band{b}_PSNR", f"band{b}_SSIMs", f"band{b}_SAM"]
                    w.writerow(hdr); wrote_header = True
                row = [fname]
                for b in range(C):
                    p = x0[:,b:b+1]; g = x_gt[:,b:b+1]
                    mae_b = masked_mae(p,g,mask); mse_b = masked_mse(p,g,mask)
                    psnr_b = psnr(p,g,mask); ssim_b = ssim_simple(p,g); 
                    # For SAM per band vs scalar it's ill-posed; report NaN or skip; here we put NaN
                    row += [f"{mae_b:.6f}", f"{mse_b:.6f}", f"{psnr_b:.3f}", f"{ssim_b:.4f}", "NaN"]
                w.writerow(row)
        # aggregate across dataset per band can be computed from CSV later
        print("[DONE] PER_BAND")

    # ---------------- MODE: ABLATE ----------------
    elif args.mode == "ablate":
        csv_path = os.path.join(args.out_dir, "ablate_all.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file","t_small","baseline_MAE","baseline_MSE","channel","MAE","MSE","dMAE","dMSE"]) 
            ch_stats: Dict[int, List[Tuple[float,float,float,float]]] = {}
            for fname in files:
                x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
                # Build baseline with fixed noise for comparability within this file
                t_small = max(1, min(args.t_small, args.T-1))
                t_idx = torch.full((1,), t_small, dtype=torch.long, device=device)
                a_t = alpha_bar[t_idx].view(-1,1,1,1)
                torch.manual_seed(args.seed_base)
                base_noise = torch.randn_like(x_gt)
                x_t_base = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*base_noise
                eps_b = model(torch.cat([x_t_base, x_cond], dim=1), t_idx)
                x0_b = (x_t_base - torch.sqrt(1 - a_t)*eps_b) / torch.sqrt(a_t + 1e-8)
                x0_b = torch.clamp(x0_b, 0.0, 1.0)
                base_mae = masked_mae(x0_b, x_gt, mask); base_mse = masked_mse(x0_b, x_gt, mask)

                for ch in range(Cc):
                    x_t = torch.sqrt(a_t)*x_gt + torch.sqrt(1 - a_t)*base_noise
                    x_cond_ab = x_cond.clone(); x_cond_ab[:, ch:ch+1].zero_()
                    eps = model(torch.cat([x_t, x_cond_ab], dim=1), t_idx)
                    x0_hat = (x_t - torch.sqrt(1 - a_t)*eps) / torch.sqrt(a_t + 1e-8)
                    x0_hat = torch.clamp(x0_hat, 0.0, 1.0)
                    mae = masked_mae(x0_hat, x_gt, mask); mse = masked_mse(x0_hat, x_gt, mask)
                    dMAE, dMSE = mae - base_mae, mse - base_mse
                    w.writerow([fname, t_small, f"{base_mae:.6f}", f"{base_mse:.6f}", ch,
                                f"{mae:.6f}", f"{mse:.6f}", f"{dMAE:.6f}", f"{dMSE:.6f}"]) 
                    ch_stats.setdefault(ch, []).append((mae, mse, dMAE, dMSE))
        print("[DONE] ABLATE")

    # ---------------- MODE: CLOUDY_VIZ ----------------
    elif args.mode == "cloudy_viz":
        # Select top-N cloudiest files (if cloud info available); otherwise random
        cloud_list = []
        for fname in files:
            _, _, _, _, _, cloud = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
            frac = cloud_fraction(cloud)
            cloud_list.append((fname, -frac if frac is not None else 0.0))  # negative to sort descending
        cloud_list.sort(key=lambda x: x[1])
        selected = [fn for fn,_ in cloud_list[:args.select_top_cloud]] if cloud_list else files[:args.select_top_cloud]
        print(f"[INFO] Selected {len(selected)} cloudy samples for viz")
        for i, fname in enumerate(selected):
            x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
            # Use a modest multi-step DDIM to reconstruct
            _, _, x0 = ddim_multistep_eval(model, x_gt, x_cond, alpha_bar, mask, t_start=200, steps=20)
            save_panel(x0, x_gt, x_cond, mask,
                       os.path.join(viz_dir, f"{i:03d}_cloudy_panel.png"),
                       title=f"Cloudy case: {fname}", zoom=args.zoom, zoom_k=args.zoom_k)
        print("[DONE] CLOUDY_VIZ")

    # ---------------- MODE: NIGHT_DEMO ----------------
    elif args.mode == "night_demo":
        # No GT expected; just generate from S1 via DDIM and save panels (S1 | Pred True | Pred CIR)
        for i, fname in enumerate(files[:max(1, args.save_viz_n)]):
            x_cond, x_gt, mask, Cc, Ct, _ = load_npz_as_tensors(os.path.join(args.patch_dir, fname), device)
            x0 = ddpm_ddim_generate(model, x_cond, alpha_bar, t_start=args.t_start, steps=args.ddim_steps)
            save_panel(x0, None, x_cond, None,
                       os.path.join(viz_dir, f"{i:03d}_night_panel.png"),
                       title=f"Night demo: {fname}")
        print("[DONE] NIGHT_DEMO")

    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
