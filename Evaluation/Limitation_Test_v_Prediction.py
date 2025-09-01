import os, math, argparse, numpy as np
from typing import Optional, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



# Schedules

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 1e-5, 0.999).float()

def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

def make_schedule(T: int, kind: str) -> torch.Tensor:
    if kind == "cosine":
        return cosine_beta_schedule(T)
    elif kind == "linear":
        return linear_beta_schedule(T)
    else:
        raise ValueError(f"Unknown time_schedule: {kind}")



# Dataset

class S1toS2Dataset(Dataset):
    """Reads .npz patches with arrays: inputs (C_cond,H,W), target (C_tgt,H,W), mask (H,W optional)."""
    def __init__(self, patch_dir: str, max_files: Optional[int] = None):
        files = [
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.endswith(".npz") and os.path.isfile(os.path.join(patch_dir, f))
        ]
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        x_cond = np.nan_to_num(d["inputs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x_tgt  = np.nan_to_num(d["target"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask   = d["mask"].astype(np.float32) if "mask" in d else None
        if mask is not None:
            mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x_cond), torch.from_numpy(x_tgt), (torch.from_numpy(mask) if mask is not None else None)



# Model

class UNetSmall(nn.Module):
    """
    Small UNet that consumes [x_t, cond] + integer timestep map, predicts v (C_tgt channels).
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

def v_to_x0_eps(x_t, v, alpha_bar_t):
    """
    Given:
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



# Metrics (pixel-weighted)

def _make_mask(mask, B, H, W, device, dtype):
    if mask is None:
        return torch.ones((B, 1, H, W), device=device, dtype=dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return (mask > 0).to(dtype).to(device)

@torch.no_grad()
def accumulate_sums(pred, tgt, mask):
    B, C, H, W = pred.shape
    w = _make_mask(mask, B, H, W, pred.device, pred.dtype)
    abs_sum_c = (w * (pred - tgt).abs()).sum(dim=(0, 2, 3))
    sq_sum_c  = (w * (pred - tgt) ** 2).sum(dim=(0, 2, 3))
    return abs_sum_c, sq_sum_c, w.sum()

def finalize_metrics(abs_sum_c_total, sq_sum_c_total, wsum_total, band_weights=None):
    denom = wsum_total.clamp_min(1e-8)
    mae_c = abs_sum_c_total / denom
    mse_c = sq_sum_c_total  / denom
    if band_weights is None:
        mae = mae_c.mean().item(); mse = mse_c.mean().item()
    else:
        w = torch.tensor(band_weights, dtype=mae_c.dtype, device=mae_c.device)
        w = w / w.sum().clamp_min(1e-8)
        mae = (mae_c * w).sum().item(); mse = (mse_c * w).sum().item()
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    psnr_c = torch.where(mse_c <= 1e-12, torch.full_like(mse_c, 99.0), 10.0 * torch.log10(1.0 / mse_c))
    return mae, mse, psnr, mae_c.detach().cpu().numpy(), mse_c.detach().cpu().numpy(), psnr_c.detach().cpu().numpy()



# Visualization (fixed ranges)

def compute_fixed_lo_hi_from_gt(patch_dir: str, file_list: List[str], q_low=1.0, q_high=99.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band low/high from ALL GT patches using percentiles.
    This yields stable visualization ranges shared by pred & gt.
    """
    lo, hi = None, None
    for fname in file_list:
        d = np.load(os.path.join(patch_dir, fname))
        xgt = np.nan_to_num(d["target"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)  # (C,H,W)
        C = xgt.shape[0]
        if lo is None:
            lo = np.full(C, np.inf, dtype=np.float32)
            hi = np.full(C, -np.inf, dtype=np.float32)
        for c in range(C):
            v = xgt[c].reshape(-1)
            lo[c] = min(lo[c], np.percentile(v, q_low))
            hi[c] = max(hi[c], np.percentile(v, q_high))
    for c in range(len(lo)):
        if hi[c] - lo[c] < 1e-6:
            hi[c] = lo[c] + 1.0
    return lo.astype(np.float32), hi.astype(np.float32)

def stretch_to_uint8_fixed(x_chw: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """x_chw: (C,H,W) torch -> fixed-range [lo,hi] per band -> uint8 (C,H,W)."""
    x = x_chw.detach().cpu().numpy()
    C, H, W = x.shape
    y = np.empty((C, H, W), dtype=np.uint8)
    for c in range(C):
        yc = (x[c] - lo[c]) / (hi[c] - lo[c] + 1e-8)
        yc = np.clip(yc, 0, 1)
        y[c] = (yc * 255.0).astype(np.uint8)
    return y

def to_rgb_panels_fixed(t4_u8: np.ndarray):
    """t4_u8: (4,H,W) uint8 [B2,B3,B4,B8] -> (true_rgb, cir_rgb) as HxWx3."""
    B2, B3, B4, B8 = t4_u8[0], t4_u8[1], t4_u8[2], t4_u8[3]
    true_rgb = np.dstack([B4, B3, B2])
    cir_rgb  = np.dstack([B8, B4, B3])
    return true_rgb, cir_rgb



# Sampling (v-pred)

@torch.no_grad()
def model_v(model, x_t, cond, t_idx):
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        return model(torch.cat([x_t, cond], dim=1), t_idx)

@torch.no_grad()
def sample_ddpm_v(model, cond, betas, alphas, alpha_bar, C_tgt):
    device = cond.device
    B, _, H, W = cond.shape
    x_t = torch.randn(B, C_tgt, H, W, device=device)
    for t in reversed(range(len(betas))):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        v = model_v(model, x_t, cond, t_idx)
        a_bar_t = alpha_bar[t]
        _, eps = v_to_x0_eps(x_t, v, a_bar_t)
        beta_t  = betas[t]
        alpha_t = alphas[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - a_bar_t + 1e-8)) * eps)
        if t > 0:
            x_t = mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
        else:
            x_t = mean
    return torch.clamp(x_t, 0.0, 1.0)

@torch.no_grad()
def sample_ddim_v(model, cond, alpha_bar, C_tgt, steps=250, eta=0.05, t_start=None):
    device = cond.device
    T = len(alpha_bar)
    B, _, H, W = cond.shape
    K = T - 1 if t_start is None else int(max(1, min(int(t_start), T - 1)))
    x_t = torch.randn(B, C_tgt, H, W, device=device) * torch.sqrt(1 - alpha_bar[K])
    idxs = torch.linspace(0, K, steps, device=device)
    idxs = torch.round(idxs).to(torch.long)
    idxs = torch.unique(idxs, sorted=True)
    if idxs[-1].item() != K:
        idxs = torch.unique(torch.cat([idxs, torch.tensor([K], device=device, dtype=torch.long)]), sorted=True)
    for i in reversed(range(len(idxs))):
        t = int(idxs[i].item())
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        v = model_v(model, x_t, cond, t_idx)
        a_t = alpha_bar[t]
        x0_pred, eps_pred = v_to_x0_eps(x_t, v, a_t)
        if i == 0:
            x_t = x0_pred
        else:
            t_prev = int(idxs[i - 1].item())
            a_prev = alpha_bar[t_prev]
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t + 1e-8) * (1 - a_t / a_prev).clamp_min(0))
            dir_term = torch.sqrt((1 - a_prev) - sigma**2).clamp_min(0)
            x_t = torch.sqrt(a_prev) * x0_pred + dir_term * eps_pred + sigma * torch.randn_like(x_t)
    return torch.clamp(x_t, 0.0, 1.0)



# Main evaluation

def run_eval(
    patch_dir, ckpt_path, out_dir,
    mode="ddim", T=1000, time_schedule="cosine",
    ddim_steps=250, ddim_eta=0.05, t_start=None,
    batch_size=2, base_ch=96, save_n=8, max_files=None,
    band_weights=None, seed=0,
    viz_mode="dataset_fixed", viz_q_low=1.0, viz_q_high=99.0
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # file list & probe
    file_list = sorted([f for f in os.listdir(patch_dir) if f.endswith(".npz")])
    assert len(file_list) > 0, f"No .npz files in {patch_dir}"
    if max_files is not None:
        file_list = file_list[:max_files]
    probe = np.load(os.path.join(patch_dir, file_list[0]))
    C_cond = probe["inputs"].shape[0]
    C_tgt  = probe["target"].shape[0]
    print(f"[INFO] inputs={C_cond}, target={C_tgt}")

    # schedule
    betas  = make_schedule(T, time_schedule).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # model
    model = UNetSmall(in_ch=C_cond + C_tgt, out_ch=C_tgt, base_ch=base_ch).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict): state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict): state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # data loader
    ds = S1toS2Dataset(patch_dir, max_files=max_files)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # visualization ranges
    if viz_mode == "dataset_fixed":
        viz_lo, viz_hi = compute_fixed_lo_hi_from_gt(patch_dir, file_list, q_low=viz_q_low, q_high=viz_q_high)
        print("[VIZ] dataset-fixed per-band ranges:", list(zip(viz_lo, viz_hi)))
    elif viz_mode == "unit":
        viz_lo = np.zeros(C_tgt, dtype=np.float32); viz_hi = np.ones(C_tgt, dtype=np.float32)
        print("[VIZ] unit range per band [0,1]")
    else:
        raise ValueError(f"Unknown viz_mode: {viz_mode}")

    # accumulators
    abs_sum_c_total = torch.zeros(C_tgt, device=device)
    sq_sum_c_total  = torch.zeros(C_tgt, device=device)
    wsum_total      = torch.tensor(0.0, device=device)

    saved = 0
    pbar = tqdm(loader, desc=f"Inference [{mode}/{time_schedule}]")
    for bi, (x_cond, x_gt, mask) in enumerate(pbar):
        x_cond = x_cond.to(device, non_blocking=True)
        x_gt   = x_gt.to(device, non_blocking=True)
        mask_b = mask.to(device, non_blocking=True) if mask is not None else None

        # sampling
        if mode == "ddpm":
            x_pred = sample_ddpm_v(model, x_cond, betas, alphas, alpha_bar, C_tgt)
        else:
            x_pred = sample_ddim_v(model, x_cond, alpha_bar, C_tgt,
                                   steps=ddim_steps, eta=ddim_eta, t_start=t_start)

        # metrics
        abs_c, sq_c, wsum = accumulate_sums(x_pred, x_gt, mask_b)
        abs_sum_c_total += abs_c
        sq_sum_c_total  += sq_c
        wsum_total      += wsum

        # quick batch status
        denom = wsum.clamp_min(1e-8)
        mae_b = (abs_c / denom).mean().item()
        mse_b = (sq_c  / denom).mean().item()
        psnr_b = 99.0 if mse_b <= 1e-12 else 10.0 * math.log10(1.0 / mse_b)
        pbar.set_postfix(mae=f"{mae_b:.4f}", psnr=f"{psnr_b:.2f}")

        # previews (fixed-range rendering)
        bs = x_gt.size(0)
        for b in range(bs):
            if saved >= save_n: break
            pre4 = x_pred[b].detach().cpu()
            gt4  = x_gt[b].detach().cpu()
            pre4_u8 = stretch_to_uint8_fixed(pre4, viz_lo, viz_hi)
            gt4_u8  = stretch_to_uint8_fixed(gt4,  viz_lo, viz_hi)
            pr_true, pr_cir = to_rgb_panels_fixed(pre4_u8)
            gt_true, gt_cir = to_rgb_panels_fixed(gt4_u8)
            stem = f"{mode}_{bi:04d}_{b:02d}"
            Image.fromarray(pr_true).save(os.path.join(out_dir, f"{stem}_pred_true.png"))
            Image.fromarray(gt_true).save(os.path.join(out_dir, f"{stem}_gt_true.png"))
            Image.fromarray(pr_cir ).save(os.path.join(out_dir, f"{stem}_pred_cir.png"))
            Image.fromarray(gt_cir ).save(os.path.join(out_dir, f"{stem}_gt_cir.png"))
            # raw arrays for debugging
            np.save(os.path.join(out_dir, f"{stem}_pred.npy"), pre4.numpy())
            np.save(os.path.join(out_dir, f"{stem}_gt.npy"),   gt4.numpy())
            saved += 1

    # final aggregation
    mae, mse, psnr, mae_c, mse_c, psnr_c = finalize_metrics(
        abs_sum_c_total, sq_sum_c_total, wsum_total, band_weights=band_weights
    )
    print("\n==== Dataset (pixel-weighted) ====")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"PSNR: {psnr:.3f} dB")
    names = ["B2", "B3", "B4", "B8"] if len(mae_c) == 4 else [f"Band{i}" for i in range(len(mae_c))]
    print("\n-- Per-channel --")
    for i, nm in enumerate(names):
        print(f"{nm:>3s}:  MAE={mae_c[i]:.6f}  MSE={mse_c[i]:.6f}  PSNR={psnr_c[i]:.3f} dB")
    print(f"\n[INFO] Previews & logs: {out_dir}")



# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir",   type=str, required=True)
    ap.add_argument("--ckpt",        type=str, required=True)
    ap.add_argument("--out_dir",     type=str, required=True)
    ap.add_argument("--mode",        type=str, default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--T",           type=int, default=1000)
    ap.add_argument("--time_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--ddim_steps",  type=int, default=250)
    ap.add_argument("--ddim_eta",    type=float, default=0.05, help="small noise to stabilize trajectory")
    ap.add_argument("--t_start",     type=int, default=None, help="optional: start DDIM from K (skip top noise)")
    ap.add_argument("--batch_size",  type=int, default=2)
    ap.add_argument("--base_ch",     type=int, default=96)
    ap.add_argument("--save_n",      type=int, default=8)
    ap.add_argument("--max_files",   type=int, default=None)
    ap.add_argument("--band_weights", nargs="*", type=float, default=None)
    ap.add_argument("--seed",        type=int, default=0)
    # visualization options
    ap.add_argument("--viz_mode",    type=str, default="dataset_fixed", choices=["dataset_fixed", "unit"],
                    help="dataset_fixed: per-band ranges from GT percentiles; unit: fixed [0,1] per band")
    ap.add_argument("--viz_q_low",   type=float, default=1.0, help="low percentile for dataset_fixed")
    ap.add_argument("--viz_q_high",  type=float, default=99.0, help="high percentile for dataset_fixed")
    args = ap.parse_args()

    run_eval(
        patch_dir=args.patch_dir,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        mode=args.mode,
        T=args.T,
        time_schedule=args.time_schedule,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        t_start=args.t_start,
        batch_size=args.batch_size,
        base_ch=args.base_ch,
        save_n=args.save_n,
        max_files=args.max_files,
        band_weights=args.band_weights,
        seed=args.seed,
        viz_mode=args.viz_mode,
        viz_q_low=args.viz_q_low,
        viz_q_high=args.viz_q_high
    )

