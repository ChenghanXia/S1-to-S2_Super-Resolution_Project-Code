import os
import math
import argparse
import random
import numpy as np
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# Reproducibility
def set_seed(seed: int = 1337):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Diffusion schedule (cosine)
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from "Improved DDPM" (Nichol & Dhariwal).
    Returns betas with length T in [1e-5, 0.999].
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clip(betas, 1e-5, 0.999)
    return betas.float()


@torch.no_grad()
def q_sample(x0, t_idx, noise, sqrt_alpha_bar, sqrt_1m_alpha_bar):
    """
    Forward diffusion:
      x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
    """
    return (
        sqrt_alpha_bar[t_idx].view(-1, 1, 1, 1) * x0 +
        sqrt_1m_alpha_bar[t_idx].view(-1, 1, 1, 1) * noise
    )



# Dataset
class S1toS2Dataset(Dataset):
    """
    Loads .npz patches produced by your patch script.
    Each npz must contain:
      - inputs: (C_cond, H, W)
      - target: (C_tgt,  H, W) in [0..1]
      - mask:   (H, W)  optional, 1=valid, 0=invalid
    """
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        x_cond = np.nan_to_num(d["inputs"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x_tgt  = np.nan_to_num(d["target"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask   = d["mask"].astype(np.float32) if "mask" in d else None
        if mask is not None:
            mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x_cond), torch.from_numpy(x_tgt), (
            torch.from_numpy(mask) if mask is not None else None
        )



# Model (U-Net Small)
class UNetSmall(nn.Module):
    """
    Predicts param (eps or v) with C_tgt channels.
    Input to the net is [x_t, cond, t_map], where t_map is an integer timestep channel.
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

        # Decoder (channels are carefully set to avoid mismatches)
        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv3 = conv_block(base_ch*8, base_ch*4)

        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv2 = conv_block(base_ch*4, base_ch*2)

        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv1 = conv_block(base_ch*2, base_ch)

        self.outc  = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, xt_and_cond, t_idx):
        B, _, H, W = xt_and_cond.shape
        # Integer timestep embedding as a 1-channel map (matches your training/eval style)
        t_map = t_idx.view(B, 1, 1, 1).float().repeat(1, 1, H, W)
        x = torch.cat([xt_and_cond, t_map], dim=1)

        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)

        u3 = self.up3(e4)
        d3 = torch.cat([u3, e3], dim=1)
        d3 = self.conv3(d3)

        u2 = self.up2(d3)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.conv2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.conv1(d1)

        return self.outc(d1)



# Loss / weighting helpers
def _make_mask(mask, B, H, W, device, dtype):
    """Build a (B,1,H,W) float mask tensor."""
    if mask is None:
        return torch.ones((B, 1, H, W), device=device, dtype=dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return (mask > 0).to(dtype).to(device)


def masked_mse_per_channel(pred, target, mask, band_weights=None, mask_as_weights=False):
    """
    Per-channel, pixel-weighted MSE with optional per-channel band weights.
    Returns (scalar loss, dict of per-channel losses).
    """
    B, C, H, W = pred.shape
    w = _make_mask(mask, B, H, W, pred.device, pred.dtype)
    if mask_as_weights:
        # Use the (normalized) soft mask as weights
        mean_val = torch.clamp(w.mean(), min=1e-6)
        w = w / mean_val

    se = (pred - target) ** 2 * w
    denom = w.sum(dim=(0, 2, 3)).clamp_min(1e-6).repeat(C)
    ch_losses = se.sum(dim=(0, 2, 3)) / denom

    if band_weights is not None:
        bw = band_weights.to(pred.device).view(C)
        total = (ch_losses * bw).sum() / bw.sum().clamp_min(1e-6)
    else:
        total = ch_losses.mean()

    stats = {f"ch{ci}": float(ch_losses[ci].detach().cpu()) for ci in range(C)}
    return total, stats


def snr_p2_weight(alpha_bar_t, p2_gamma: float = 1.0, p2_k: float = 1e-3):
    """
    Imagen/Stable Diffusion p2 reweighting:
      SNR(t)   = alpha_bar / (1 - alpha_bar)
      weight   = (p2_k + SNR)^(-p2_gamma)
    Given alpha_bar[t] per-sample, return per-sample weights.
    """
    snr = alpha_bar_t / torch.clamp(1.0 - alpha_bar_t, min=1e-8)
    return torch.pow(p2_k + snr, -p2_gamma)


# v-parameterization conversions (consistent with q_sample above)
def v_from_x0_eps(x0, eps, sqrt_ab, sqrt_1m_ab):
    """
    One common v definition:
      x_t = A x0 + B eps,  v = A eps - B x0
    with A = sqrt(alpha_bar), B = sqrt(1 - alpha_bar)
    """
    return sqrt_ab * eps - sqrt_1m_ab * x0


def x0_eps_from_v_x(x_t, v, sqrt_ab, sqrt_1m_ab):
    """
    Invert:
      x_t = A x0 + B eps
      v   = A eps - B x0
    =>  x0 = A * x_t - B * v
        eps = B * x_t + A * v
    """
    x0  = sqrt_ab * x_t - sqrt_1m_ab * v
    eps = sqrt_1m_ab * x_t + sqrt_ab * v
    return x0, eps



# EMA
class EMA:
    """Simple EMA over model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone() for n, p in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = (1.0 - self.decay) * param + self.decay * self.shadow[name]

    def apply_shadow(self, model):
        self.backup = model.state_dict()
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


# T sampling strategies
def sample_timesteps(T: int, B: int, mode: str, high_t_frac: float, high_t_min_ratio: float, device):
    """
    Choose timesteps for training:
      - 'uniform'   : uniformly from [0, T-1]
      - 'high_only' : uniformly from [t_min, T-1]
      - 'mix_high'  : with prob=high_t_frac pick from [t_min, T-1], else uniform
    high_t_min_ratio in [0,1): e.g. 0.6 -> start 'high' region at floor(0.6*T)
    """
    t_min = int(max(1, min(T - 1, round(high_t_min_ratio * T))))
    if mode == "uniform":
        return torch.randint(0, T, (B,), device=device)
    elif mode == "high_only":
        return torch.randint(t_min, T, (B,), device=device)
    elif mode == "mix_high":
        m = torch.rand(B, device=device)
        t = torch.empty(B, dtype=torch.long, device=device)
        hi = (m < high_t_frac)
        n_hi = int(hi.sum().item())
        n_lo = B - n_hi
        if n_hi > 0:
            t_hi = torch.randint(t_min, T, (n_hi,), device=device)
            t[hi] = t_hi
        if n_lo > 0:
            t_lo = torch.randint(0, T, (n_lo,), device=device)
            t[~hi] = t_lo
        return t
    else:
        raise ValueError(f"Unknown t_sampler: {mode}")



# Training loop
def train(
    patch_dir,
    model_path,
    T=1000,
    epochs=40,
    batch_size=4,
    lr=1e-4,
    base_ch=96,
    grad_clip=0.5,
    max_patches=None,
    weight_decay=1e-4,
    ema_decay=0.999,
    seed=1337,
    band_weights=None,
    mask_as_weights=False,
    pred_param="v", # 'eps' or 'v'
    t_sampler="mix_high", # 'uniform' | 'high_only' | 'mix_high'
    high_t_frac=0.5,
    high_t_min_ratio=0.6,
    p2_gamma=1.0,
    p2_k=1e-3,
    aux_x0_loss_w=0.02
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Probe channels
    sample_files = [f for f in os.listdir(patch_dir) if f.endswith(".npz")]
    assert len(sample_files) > 0, f"No .npz files found in {patch_dir}"
    sample = np.load(os.path.join(patch_dir, sample_files[0]))
    C_cond = sample["inputs"].shape[0]
    C_tgt  = sample["target"].shape[0]
    print(f"[INFO] Channels: cond={C_cond}, target={C_tgt}")

    # Per-channel weights for loss, if any
    bw_tensor = torch.tensor(band_weights, dtype=torch.float32) if band_weights else None

    # Schedules
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_1m_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # Data
    ds = S1toS2Dataset(patch_dir, max_files=max_patches)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # Model / optimizer / EMA
    model = UNetSmall(in_ch=C_cond + C_tgt, out_ch=C_tgt, base_ch=base_ch).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ema = EMA(model, decay=ema_decay)

    best_loss = float("inf")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    last_path = model_path.replace(".pth", "_last.pth")
    best_path = model_path.replace(".pth", "_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batches, nan_skipped = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for x_cond, x0, mask in pbar:
            x_cond, x0 = x_cond.to(device), x0.to(device)
            mask_t = mask.to(device) if mask is not None else None

            if not torch.isfinite(x_cond).all() or not torch.isfinite(x0).all():
                nan_skipped += 1
                continue

            B = x0.size(0)

            # 1) sample timesteps
            t_idx = sample_timesteps(
                T=T, B=B, mode=t_sampler,
                high_t_frac=high_t_frac, high_t_min_ratio=high_t_min_ratio,
                device=device
            )

            # 2) forward diffusion
            noise = torch.randn_like(x0)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                x_t = q_sample(x0, t_idx, noise, sqrt_alpha_bar, sqrt_1m_alpha_bar)

                # 3) build targets for chosen parameterization
                sab   = sqrt_alpha_bar[t_idx].view(B, 1, 1, 1)
                s1mab = sqrt_1m_alpha_bar[t_idx].view(B, 1, 1, 1)

                if pred_param == "eps":
                    target = noise
                elif pred_param == "v":
                    target = v_from_x0_eps(x0, noise, sab, s1mab)
                else:
                    raise ValueError("pred_param must be 'eps' or 'v'")

                # 4) predict param
                inp = torch.cat([x_t, x_cond], dim=1)
                pred = model(inp, t_idx)

                # 5) per-channel masked MSE
                base_loss, ch_stats = masked_mse_per_channel(
                    pred=pred, target=target, mask=mask_t,
                    band_weights=bw_tensor, mask_as_weights=mask_as_weights
                )

                # 6) p2 (SNR) reweighting
                p2_w = snr_p2_weight(alpha_bar[t_idx], p2_gamma=p2_gamma, p2_k=p2_k).mean().detach()
                loss = base_loss * p2_w

                # 7) small auxiliary x0 loss
                if aux_x0_loss_w > 0.0:
                    if pred_param == "eps":
                        eps_pred = pred
                        x0_pred = (x_t - s1mab * eps_pred) / torch.clamp(sab, min=1e-8)
                    else:
                        v_pred = pred
                        x0_pred, _ = x0_eps_from_v_x(x_t, v_pred, sab, s1mab)

                    aux_loss, _ = masked_mse_per_channel(
                        pred=x0_pred, target=x0, mask=mask_t,
                        band_weights=bw_tensor, mask_as_weights=mask_as_weights
                    )
                    loss = loss + aux_x0_loss_w * aux_loss

            if not torch.isfinite(loss):
                nan_skipped += 1
                continue

            # 8) optimize
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running += float(loss.detach().cpu())
            n_batches += 1

            show = {
                "loss": f"{float(loss.detach().cpu()):.4f}",
                "p2": f"{float(p2_w.cpu()):.3f}",
                "skip": nan_skipped,
            }
            show.update({k: f"{v:.4f}" for k, v in list(ch_stats.items())[:4]})
            pbar.set_postfix(**show)

        avg_loss = running / max(1, n_batches)
        print(f"→ Epoch {epoch}: avg loss = {avg_loss:.6f} (skipped {nan_skipped})")

        # Save last/best (EMA weights)
        ema.apply_shadow(model)
        torch.save(model.state_dict(), last_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved: {best_path}")
        ema.restore(model)

    # Save final EMA model
    ema.apply_shadow(model)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Final EMA model saved: {model_path}")
    ema.restore(model)


# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--max_patches", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--band_weights", nargs="*", type=float, default=None)
    ap.add_argument("--mask_as_weights", action="store_true")

    # Upgrades
    ap.add_argument("--pred_param", choices=["eps", "v"], default="v")
    ap.add_argument("--t_sampler", choices=["uniform", "high_only", "mix_high"], default="mix_high")
    ap.add_argument("--high_t_frac", type=float, default=0.5)
    ap.add_argument("--high_t_min_ratio", type=float, default=0.6)
    ap.add_argument("--p2_gamma", type=float, default=1.0)
    ap.add_argument("--p2_k", type=float, default=1e-3)
    ap.add_argument("--aux_x0_loss_w", type=float, default=0.02)

    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    train(
        patch_dir=args.patch_dir,
        model_path=args.model_path,
        T=args.T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_ch=args.base_ch,
        grad_clip=args.grad_clip,
        max_patches=args.max_patches,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        seed=args.seed,
        band_weights=args.band_weights,
        mask_as_weights=args.mask_as_weights,
        pred_param=args.pred_param,
        t_sampler=args.t_sampler,
        high_t_frac=args.high_t_frac,
        high_t_min_ratio=args.high_t_min_ratio,
        p2_gamma=args.p2_gamma,
        p2_k=args.p2_k,
        aux_x0_loss_w=args.aux_x0_loss_w
    )
