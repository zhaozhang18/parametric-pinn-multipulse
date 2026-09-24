import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def safe_torch_load(path, map_location='cpu'):
    """Load trusted project checkpoints without PyTorch's omitted-argument warning.

    The explicit weights_only=False is required because these checkpoints contain
    model_config/train_config metadata in addition to tensors. The fallback keeps
    compatibility with older PyTorch versions that do not expose weights_only.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


# ------------------------------
# Model definitions
# ------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.bn(self.conv(x))
        out = self.relu(out + residual)
        return out


class CoarseGlobalContext1D(nn.Module):
    def __init__(self, channels, bins=40):
        super().__init__()
        self.bins = int(max(4, bins))
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        pooled = F.adaptive_avg_pool1d(x, self.bins)
        context = self.net(pooled)
        context = F.interpolate(context, size=x.shape[-1], mode='linear', align_corners=False)
        return x + context


class FullPropagationCNNGeneric(nn.Module):
    def __init__(self, n_slices, hidden=40, kernel_size=11, dilations=(1,4,16,64,128), context_bins=40):
        super().__init__()
        blocks = []
        in_ch = 2
        for d in dilations:
            blocks.append(ResidualConvBlock(in_ch, hidden, kernel_size, int(d)))
            in_ch = hidden
        self.features = nn.Sequential(*blocks)
        self.context = CoarseGlobalContext1D(hidden, context_bins)
        self.linear1 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(hidden, 2 * n_slices, kernel_size=1)
        self.n_slices = int(n_slices)
        residual_mask = torch.ones(1, self.n_slices, 1, 1, dtype=torch.float32)
        residual_mask[:, 0, :, :] = 0.0
        self.register_buffer('residual_mask', residual_mask, persistent=False)

    def forward(self, x):
        h = self.features(x)
        h = self.context(h)
        h = self.relu(self.linear1(h))
        delta = self.linear2(h)
        b, _, t = delta.shape
        delta = delta.view(b, self.n_slices, 2, t)
        base = x.unsqueeze(1).expand(-1, self.n_slices, -1, -1)
        pred = base + delta * self.residual_mask.to(delta.dtype)
        return pred.reshape(b, 2 * self.n_slices, t)


class InverseResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class BaselineInverseCNN(nn.Module):
    def __init__(self, hidden=80, kernel_size=11, dilations=(1,4,16,64,128)):
        super().__init__()
        self.input_layer = nn.Conv1d(2, hidden, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList([InverseResidualBlock(hidden, kernel_size, int(d)) for d in dilations])
        self.output_norm = nn.GroupNorm(1, hidden)
        self.output_layer = nn.Conv1d(hidden, 2, kernel_size=1)

    def forward(self, waveform):
        x = self.input_layer(waveform)
        for block in self.blocks:
            x = block(x)
        delta = self.output_layer(F.gelu(self.output_norm(x)))
        return waveform + delta


# ------------------------------
# Data helpers
# ------------------------------
def build_basis(tau, centers):
    return np.stack([np.exp(-0.5 * (tau - c) ** 2) for c in centers], axis=0)


def build_initial_waveform(amplitudes, tau, centers):
    basis = build_basis(tau, centers)
    real = amplitudes @ basis
    imag = np.zeros_like(real)
    return np.stack([real, imag], axis=0).astype(np.float32), basis


def recover_amplitudes_from_reconstructed_waveform(hhat0_2xt, basis):
    y = np.asarray(hhat0_2xt[0], dtype=np.float64)
    amps, *_ = np.linalg.lstsq(basis.T, y, rcond=None)
    amps = np.clip(amps, 0.0, 1.0)
    return amps


def load_forward_model(path, device):
    payload = safe_torch_load(path, map_location=device)
    cfg = payload['model_config']
    model = FullPropagationCNNGeneric(
        n_slices=int(cfg['n_slices']),
        hidden=int(cfg['hidden']),
        kernel_size=int(cfg['kernel_size']),
        dilations=tuple(int(x) for x in cfg['dilations']),
        context_bins=int(cfg.get('context_bins', 40)),
    ).to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload


def load_inverse_model(path, device):
    payload = safe_torch_load(path, map_location=device)
    cfg = payload['model_config']
    model = BaselineInverseCNN(
        hidden=int(cfg['hidden']),
        kernel_size=int(cfg['kernel_size']),
        dilations=tuple(int(x) for x in cfg['dilations']),
    ).to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload


# ------------------------------
# Drawing helpers
# ------------------------------
def add_box(ax, x, y, w, h, text, edgecolor, facecolor='white', fontsize=12, lw=1.8, roundness=0.015):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.006,rounding_size={roundness}",
                           linewidth=lw, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)
    return patch


def add_down_arrow(ax, x1, y1, x2, y2, color='black', lw=1.6, ms=14):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                          linewidth=lw, color=color)
    ax.add_patch(arr)


def add_curve_arrow(ax, xy1, xy2, color, rad, lw=3.0, ms=18):
    arr = FancyArrowPatch(xy1, xy2, arrowstyle='Simple,tail_width=0.8,head_width=8,head_length=10',
                          connectionstyle=f'arc3,rad={rad}', color=color, linewidth=lw, alpha=0.95)
    ax.add_patch(arr)


def plot_waveform_inset(fig, parent_ax, rect, x, y, color='royalblue', lw=2.0, use_abs=True, ylim_pad=0.1):
    ax = fig.add_axes(rect)
    xx = np.asarray(x)
    yy = np.asarray(y)
    if yy.ndim == 2 and yy.shape[0] == 2:
        if use_abs:
            yy = np.sqrt(yy[0]**2 + yy[1]**2)
        else:
            yy = yy[0]
    ax.plot(xx, yy, color=color, lw=lw)
    ymin, ymax = float(np.min(yy)), float(np.max(yy))
    pad = ylim_pad * max(1e-6, ymax - ymin)
    if abs(ymax - ymin) < 1e-9:
        pad = 0.1
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin - 0.02 * max(1e-6, ymax-ymin), ymax + pad)
    ax.axis('off')
    return ax


def draw_multi_slices_box(fig, rect, tau, slices_mag, labels, color='royalblue'):
    outer = fig.add_axes(rect)
    outer.axis('off')
    n = len(slices_mag)
    lefts = np.linspace(0.04, 0.72, n)
    width = 0.22
    for i, (sig, lab) in enumerate(zip(slices_mag, labels)):
        ax = fig.add_axes([
            rect[0] + rect[2] * lefts[i],
            rect[1] + rect[3] * 0.18,
            rect[2] * width,
            rect[3] * 0.56,
        ])
        ax.plot(tau, sig, color=color, lw=1.8)
        ax.axis('off')
        outer.text(lefts[i] + width/2, 0.05, lab, ha='center', va='center', fontsize=11, transform=outer.transAxes)
    if n >= 3:
        outer.text(0.50, 0.47, '⋯', ha='center', va='center', fontsize=22, transform=outer.transAxes)
    return outer


def make_figure(output_path, forward_ckpt, inverse_ckpt, seen_csv, unseen_csv, metadata_json,
                sample_source='seen', sample_index=-1, explicit_amplitudes=None):
    device = torch.device('cpu')
    with open(metadata_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    tau = np.asarray(meta['tau'], dtype=np.float64)
    zeta = np.asarray(meta['zeta'], dtype=np.float64)
    n_slices = int(meta['n_slices'])
    centers = np.array([-28., -20., -12., -4., 4., 12., 20., 28.], dtype=np.float64)

    if explicit_amplitudes is not None:
        amps = np.array(explicit_amplitudes, dtype=np.float64).reshape(8)
        K = int(np.count_nonzero(amps > 0))
        selected_index = None
    else:
        csv_path = seen_csv if sample_source.lower() == 'seen' else unseen_csv
        df = pd.read_csv(csv_path)
        required_cols = ['K'] + [f'A{i}' for i in range(1, 9)]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f'CSV columns are incompatible: {csv_path} | missing={missing}')
        if len(df) == 0:
            raise ValueError(f'CSV contains no samples: {csv_path}')

        # sample_index < 0 means automatic representative selection.
        if sample_index is None or int(sample_index) < 0:
            # Prefer an actual K=8 waveform so the architecture figure visibly
            # demonstrates the universal multi-pulse task.
            subset = df[df['K'].astype(int) == 8]
            if len(subset) == 0:
                max_k = int(df['K'].astype(int).max())
                subset = df[df['K'].astype(int) == max_k]
            vals = subset[[f'A{i}' for i in range(1, 9)]].to_numpy(dtype=np.float64)
            # Choose a nontrivial representative: large total energy and some
            # amplitude variation, rather than the first row by accident.
            scores = vals.sum(axis=1) + 0.35 * vals.std(axis=1)
            selected_index = int(subset.index[int(np.argmax(scores))])
        else:
            requested = int(sample_index)
            if requested >= len(df) or requested < -len(df):
                safe = requested % len(df)
                print(f'[warning] requested sample index {requested} is outside 0..{len(df)-1}; using {safe}.', flush=True)
                selected_index = safe
            else:
                selected_index = requested

        row = df.iloc[int(selected_index)]
        amps = row[[f'A{i}' for i in range(1, 9)]].to_numpy(dtype=np.float64)
        K = int(row['K'])
        print(f'Using {sample_source} CSV sample: positional index={selected_index}, K={K}, rows={len(df)}', flush=True)

    x2t, basis = build_initial_waveform(amps, tau, centers)
    scale = float(meta.get('field_scale_max_abs_seen', 1.0))
    x_in = torch.from_numpy((x2t / scale)[None, ...]).to(device)

    fwd_model, fwd_payload = load_forward_model(forward_ckpt, device)
    inv_model, inv_payload = load_inverse_model(inverse_ckpt, device)

    with torch.no_grad():
        pred = fwd_model(x_in).cpu().numpy()[0]
        pred = pred.reshape(n_slices, 2, len(tau)) * scale
        terminal = pred[-1]
        recon = inv_model(torch.from_numpy((terminal / scale)[None, ...]).float()).cpu().numpy()[0] * scale
    recovered = recover_amplitudes_from_reconstructed_waveform(recon, basis)

    mag_input = np.sqrt(x2t[0]**2 + x2t[1]**2)
    mag_terminal = np.sqrt(terminal[0]**2 + terminal[1]**2)
    mag_recon = np.sqrt(recon[0]**2 + recon[1]**2)
    sample_slices = [pred[0], pred[n_slices//2], pred[-1]]
    sample_mags = [np.sqrt(s[0]**2 + s[1]**2) for s in sample_slices]

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis('off')

    # Titles and panel labels
    canvas.text(0.06, 0.95, '(a)', fontsize=20, fontweight='bold')
    canvas.text(0.52, 0.95, '(b)', fontsize=20, fontweight='bold')
    canvas.text(0.245, 0.95, 'Universal forward CNN', color='#1737c8', fontsize=24, fontweight='bold', ha='center')
    canvas.text(0.76, 0.95, 'Universal inverse CNN', color='#cc2020', fontsize=24, fontweight='bold', ha='center')

    # Top actual waveforms
    plot_waveform_inset(fig, canvas, [0.11, 0.84, 0.28, 0.09], tau, mag_input, color='#1737c8', lw=2.2)
    plot_waveform_inset(fig, canvas, [0.57, 0.84, 0.28, 0.09], tau, mag_terminal, color='#d61f1f', lw=2.0)
    canvas.text(0.25, 0.81, r'Input complex waveform $h(0,t;\,a_8)$', fontsize=14, ha='center')
    canvas.text(0.25, 0.775, rf'$a_8=[a_1,\ldots,a_8],\ K={K}$', fontsize=14, ha='center')
    canvas.text(0.70, 0.815, r'Terminal complex waveform $h(z_{out},t)$', fontsize=14, ha='center')

    # Network blocks. The number of boxes is read from each checkpoint,
    # so four-block and five-block trained models are both drawn correctly.
    xL, xR, w = 0.145, 0.57, 0.23
    block_colors = [
        ('#6aa0d8', '#edf5ff'),
        ('#e6a000', '#fff6da'),
        ('#6aa0d8', '#edf5ff'),
        ('#e6a000', '#fff6da'),
        ('#d97b7b', '#fff1f1'),
        ('#8f80d8', '#f2eeff'),
    ]

    def draw_residual_stack(x0, center_x, cfg, top_arrow_x, title_prefix='Residual Conv1D block'):
        dilations = [int(v) for v in cfg.get('dilations', [])]
        if not dilations:
            raise ValueError('Checkpoint model_config has no dilations: ' + str(cfg))
        n_blocks = len(dilations)
        # Keep the whole stack in the same vertical region regardless of block count.
        ys = np.linspace(0.69, 0.25, n_blocks).tolist()
        box_h = 0.078 if n_blocks <= 5 else 0.064
        for i, (y, dilation) in enumerate(zip(ys, dilations)):
            edge, face = block_colors[i % len(block_colors)]
            txt = (
                f'{title_prefix} {i+1}\n'
                f'{int(cfg["hidden"])} channels, kernel {int(cfg["kernel_size"])}\n'
                f'Dilation rate = {dilation}'
            )
            add_box(canvas, x0, y, w, box_h, txt, edgecolor=edge, facecolor=face, fontsize=13)
            if i == 0:
                add_down_arrow(canvas, top_arrow_x, 0.79, top_arrow_x, y + box_h)
            else:
                add_down_arrow(canvas, center_x, ys[i-1], center_x, y + box_h)
        return ys, box_h, dilations

    # Forward stack
    fcfg = dict(fwd_payload['model_config'])
    ys_f, h_f, dils_f = draw_residual_stack(xL, 0.26, fcfg, 0.25)
    add_box(canvas, 0.145, 0.18, 0.23, 0.04,
            f'Coarse global context, {int(fcfg.get("context_bins", 40))} bins',
            edgecolor='#8f80d8', facecolor='#f2eeff', fontsize=12)
    add_down_arrow(canvas, 0.26, ys_f[-1], 0.26, 0.22)

    # Bottom forward result box
    add_box(canvas, 0.12, 0.05, 0.30, 0.10,
            'Predicted complex fields\n' +
            rf'$\hat h(z_1,t),\ldots,\hat h(z_{{{n_slices}}},t)$',
            edgecolor='#aaaaaa', facecolor='white', fontsize=14)
    slice_ids = [0, n_slices // 2, n_slices - 1]
    slice_labels = [f'm = {idx + 1}' for idx in slice_ids]
    draw_multi_slices_box(fig, [0.125, 0.055, 0.29, 0.075], tau, sample_mags, slice_labels, color='#1737c8')
    add_down_arrow(canvas, 0.26, 0.18, 0.26, 0.15)

    # Inverse stack
    icfg = dict(inv_payload['model_config'])
    ys_i, h_i, dils_i = draw_residual_stack(xR, 0.685, icfg, 0.685)
    add_box(canvas, 0.61, 0.18, 0.15, 0.035, 'GroupNorm + GELU',
            edgecolor='#8f80d8', facecolor='#f2eeff', fontsize=12)
    add_down_arrow(canvas, 0.685, ys_i[-1], 0.685, 0.215)
    add_box(canvas, 0.585, 0.125, 0.20, 0.045,
            'Conv1D (1 × 1) output layer\n2 channels (Real, Imag)',
            edgecolor='#d97b7b', facecolor='#fff1f1', fontsize=12)
    add_down_arrow(canvas, 0.685, 0.18, 0.685, 0.17)

    print('Forward architecture:',
          f"hidden={fcfg.get('hidden')}, kernel={fcfg.get('kernel_size')}, dilations={dils_f}",
          flush=True)
    print('Inverse architecture:',
          f"hidden={icfg.get('hidden')}, kernel={icfg.get('kernel_size')}, dilations={dils_i}",
          flush=True)

    # Bottom inverse box with actual reconstructed waveform and recovered amplitudes
    add_box(canvas, 0.53, 0.05, 0.32, 0.10,
            r'Recovered eight-slot amplitudes $\,\hat a_8=[\hat a_1,\ldots,\hat a_8]$' + '\n' +
            r'Reconstructed input waveform $\hat h(0,t)$',
            edgecolor='#aaaaaa', facecolor='white', fontsize=14)
    plot_waveform_inset(fig, canvas, [0.565, 0.065, 0.24, 0.05], tau, mag_recon, color='#d61f1f', lw=2.0)

    # amplitude annotation
    amp_text_true = 'True: [' + ', '.join(f'{a:.2f}' for a in amps) + ']'
    amp_text_hat = 'Pred: [' + ', '.join(f'{a:.2f}' for a in recovered) + ']'
    canvas.text(0.69, 0.025, amp_text_true, ha='center', va='center', fontsize=11, color='black')
    canvas.text(0.69, 0.008, amp_text_hat, ha='center', va='center', fontsize=11, color='#b00000')

    # curved arrows and side labels
    add_curve_arrow(canvas, (0.04, 0.86), (0.05, 0.08), color='#1737c8', rad=0.18, lw=2.8, ms=18)
    canvas.text(0.035, 0.41, 'Forward\nmapping', color='#1737c8', fontsize=18, fontweight='bold', ha='center', va='center')
    add_curve_arrow(canvas, (0.93, 0.86), (0.92, 0.08), color='#d61f1f', rad=-0.18, lw=2.8, ms=18)
    canvas.text(0.95, 0.41, 'Inverse\nmapping', color='#d61f1f', fontsize=18, fontweight='bold', ha='center', va='center')

    # captions
    canvas.text(0.245, 0.0, '(a) Fully supervised universal waveform-to-propagation mapping',
                fontsize=15, ha='center', va='bottom', fontweight='bold')
    canvas.text(0.69, 0.0, '(b) Fully supervised terminal-to-input mapping',
                fontsize=15, ha='center', va='bottom', fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if output_path.lower().endswith('.png'):
        plt.savefig(output_path[:-4] + '.pdf', bbox_inches='tight')
        plt.savefig(output_path[:-4] + '.svg', bbox_inches='tight')
    plt.close(fig)


def parse_amplitudes(text):
    vals = [float(x.strip()) for x in text.split(',') if x.strip()]
    if len(vals) != 8:
        raise ValueError('Need exactly 8 amplitude values.')
    return vals


if __name__ == '__main__':
    # V3: dynamic block count and ranked checkpoint discovery.
    # Keep this script in the project root. It searches model/data files
    # recursively in their original experiment folders.
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', default='', help='Project root. Default: script folder.')
    ap.add_argument('--run-dir', default='', help='Optional exact universal run folder.')
    ap.add_argument('--sample-source', default='seen', choices=['seen', 'unseen'])
    ap.add_argument('--sample-index', type=int, default=-1, help='0-based row index; default -1 auto-selects a representative K=8 sample')
    ap.add_argument('--amplitudes', type=str, default='')
    ap.add_argument('--output', default='universal_cnn_arch_actual.png')
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else script_dir

    def load_payload_quiet(path):
        try:
            return torch.load(str(path), map_location='cpu')
        except Exception:
            return None

    def is_forward_checkpoint(path):
        payload = load_payload_quiet(path)
        if not isinstance(payload, dict):
            return False
        task = str(payload.get('task', '')).lower()
        cfg = payload.get('model_config', {})
        return task == 'forward' or (isinstance(cfg, dict) and 'n_slices' in cfg and 'dilations' in cfg)

    def is_inverse_checkpoint(path):
        payload = load_payload_quiet(path)
        if not isinstance(payload, dict):
            return False
        task = str(payload.get('task', '')).lower()
        cfg = payload.get('model_config', {})
        return task == 'inverse' or (
            isinstance(cfg, dict)
            and 'n_slices' not in cfg
            and 'hidden' in cfg
            and 'dilations' in cfg
        )

    def valid_metadata(path):
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            return isinstance(data, dict) and 'tau' in data and 'n_slices' in data
        except Exception:
            return False

    def sha256_file(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def unique_existing(paths):
        result = []
        seen = set()
        for p in paths:
            p = Path(p)
            try:
                key = str(p.resolve()).lower()
            except Exception:
                key = str(p).lower()
            if p.is_file() and key not in seen:
                result.append(p)
                seen.add(key)
        return result

    required_csv_cols = ['K'] + [f'A{i}' for i in range(1, 9)]

    def inspect_csv(path):
        try:
            df = pd.read_csv(path)
            valid = all(c in df.columns for c in required_csv_cols) and len(df) > 0
            return {'path': Path(path), 'valid': valid, 'rows': int(len(df)), 'columns': list(df.columns)}
        except Exception as exc:
            return {'path': Path(path), 'valid': False, 'rows': 0, 'error': str(exc), 'columns': []}

    def normalize_root(given):
        given = Path(given).expanduser().resolve()
        if given.name.lower() == 'universal_cnn_ddnn_full81':
            return given
        candidate = given / 'universal_CNN_DDNN_full81'
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError('--run-dir does not contain universal_CNN_DDNN_full81: ' + str(given))

    if args.run_dir:
        roots = [normalize_root(args.run_dir)]
    else:
        roots = [p for p in project_root.rglob('universal_CNN_DDNN_full81') if p.is_dir()]
        if not roots:
            raise FileNotFoundError('No universal_CNN_DDNN_full81 folder found under: ' + str(project_root))

    def find_metadata(root):
        paths = [
            root / 'shared_seen_SSFM_81planes' / 'metadata.json',
            root / 'shared_seen_SSFM_81planes' / 'metadata(1).json',
        ]
        paths.extend(root.rglob('metadata*.json'))
        return next((p for p in unique_existing(paths) if valid_metadata(p)), None)

    def checkpoint_rank(path, task_name):
        payload = load_payload_quiet(path)
        if not isinstance(payload, dict):
            return (-10**9,)
        cfg = payload.get('model_config', {})
        if not isinstance(cfg, dict):
            cfg = {}
        parts = [part.lower() for part in Path(path).parts]
        name = Path(path).name.lower()
        score = 0.0
        # Finished retraining checkpoints are preferred over selection snapshots.
        if 'final' in parts:
            score += 1000
        if 'selection' in parts:
            score -= 100
        # Prefer the large/scaled network requested for this architecture figure.
        if 'large' in name or any('large' in part for part in parts):
            score += 300
        if 'scaled' in name or any('scaled' in part for part in parts):
            score += 100
        if str(payload.get('task', '')).lower() == task_name:
            score += 200
        # When several compatible checkpoints exist, prefer the more expressive one.
        score += float(cfg.get('hidden', 0)) * 0.1
        score += len(cfg.get('dilations', [])) * 5.0
        score += float(payload.get('selected_epoch', payload.get('best_epoch', 0))) * 1e-4
        try:
            mtime = Path(path).stat().st_mtime
        except Exception:
            mtime = 0.0
        return (score, mtime)

    def find_forward(root):
        paths = []
        for sub in ['universal_CNN_large_scaled', 'universal_CNN81']:
            d = root / sub
            if d.is_dir():
                paths.extend(d.rglob('*.pt'))
        paths.extend(root.rglob('*forward*cnn81*.pt'))
        paths.extend(root.rglob('*universal_cnn81*.pt'))
        paths.extend(root.rglob('*.pt'))
        valid = [p for p in unique_existing(paths) if is_forward_checkpoint(p)]
        return max(valid, key=lambda p: checkpoint_rank(p, 'forward')) if valid else None

    def find_inverse(root):
        paths = []
        d = root / 'universal_CNN_baseline_inverse'
        if d.is_dir():
            paths.extend(d.rglob('*.pt'))
        paths.extend(root.rglob('*inverse*.pt'))
        paths.extend(root.rglob('*.pt'))
        valid = [p for p in unique_existing(paths) if is_inverse_checkpoint(p)]
        return max(valid, key=lambda p: checkpoint_rank(p, 'inverse')) if valid else None

    def csv_candidates(root, kind):
        run_root = root.parent
        exact_names = [
            f'{kind}_sparse8_combinations.csv',
            f'{kind}_sparse8_combinations_exact_copy.csv',
            f'{kind}_sparse8_combinations_exact_copy(1).csv',
        ]
        search_roots = [run_root / 'dataset', root / 'shared_seen_SSFM_81planes', run_root]
        paths = []
        for base in search_roots:
            if not base.exists():
                continue
            for name in exact_names:
                p = base / name
                if p.is_file():
                    paths.append(p)
            paths.extend(base.rglob(f'*{kind}*sparse8*combinations*.csv'))
        return unique_existing(paths)

    def choose_csv(root, kind, meta):
        target_hash = str(meta.get(f'{kind}_csv_sha256', '')).strip().lower()
        target_rows = int(meta.get('n_seen', -1)) if kind == 'seen' else -1
        inspected = [inspect_csv(p) for p in csv_candidates(root, kind)]
        inspected = [x for x in inspected if x['valid']]
        if not inspected:
            return None, {'hash_match': False, 'row_match': False, 'rows': 0}

        for item in inspected:
            item['hash_match'] = False
            if target_hash:
                try:
                    item['hash_match'] = sha256_file(item['path']).lower() == target_hash
                except Exception:
                    pass
            item['row_match'] = target_rows > 0 and item['rows'] == target_rows

        inspected.sort(key=lambda x: (
            not x['hash_match'],
            not x['row_match'],
            -x['rows'],
            'exact_copy' not in x['path'].name.lower(),
            len(x['path'].parts),
        ))
        best = inspected[0]
        return best['path'], best

    # Evaluate each experiment as one internally consistent bundle. This avoids
    # selecting a recently modified but incomplete run and then pairing it with
    # a tiny/empty CSV from another subfolder.
    bundles = []
    for root in roots:
        metadata_path = find_metadata(root)
        if metadata_path is None:
            continue
        try:
            with metadata_path.open('r', encoding='utf-8') as f:
                meta_preview = json.load(f)
        except Exception:
            continue
        forward_path = find_forward(root)
        inverse_path = find_inverse(root)
        seen_path, seen_info = choose_csv(root, 'seen', meta_preview)
        unseen_path, unseen_info = choose_csv(root, 'unseen', meta_preview)

        complete = all(x is not None for x in [forward_path, inverse_path, seen_path, unseen_path])
        score = 0
        score += 1000 if complete else 0
        score += 300 if seen_info.get('hash_match') else 0
        score += 150 if seen_info.get('row_match') else 0
        score += 200 if unseen_info.get('hash_match') else 0
        score += min(int(seen_info.get('rows', 0)), 100000) / 100000.0
        run_name = root.parent.name.lower()
        if run_name == 'universal_sparse8_k1to8_highk_try1':
            score += 500
        elif 'sparse8' in run_name and 'highk' in run_name:
            score += 200
        # modification time is only a final tie-breaker, never the main score
        try:
            mtime = root.stat().st_mtime
        except Exception:
            mtime = 0.0
        bundles.append({
            'root': root,
            'metadata': metadata_path,
            'forward': forward_path,
            'inverse': inverse_path,
            'seen': seen_path,
            'unseen': unseen_path,
            'seen_info': seen_info,
            'unseen_info': unseen_info,
            'score': score,
            'mtime': mtime,
            'complete': complete,
        })

    if not bundles:
        raise FileNotFoundError('No experiment contains a valid metadata JSON under: ' + str(project_root))

    bundle = max(bundles, key=lambda b: (b['score'], b['mtime']))
    if not bundle['complete']:
        detail = '\n'.join(
            f"  {b['root']} | forward={bool(b['forward'])} inverse={bool(b['inverse'])} "
            f"seen={bool(b['seen'])} unseen={bool(b['unseen'])}"
            for b in bundles
        )
        raise FileNotFoundError('No complete compatible CNN experiment bundle was found. Candidates:\n' + detail)

    method_root = bundle['root']
    run_root = method_root.parent
    metadata_json = bundle['metadata']
    forward_ckpt = bundle['forward']
    inverse_ckpt = bundle['inverse']
    seen_csv = bundle['seen']
    unseen_csv = bundle['unseen']

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    amps = parse_amplitudes(args.amplitudes) if args.amplitudes else None

    print('=' * 100)
    print('Selected experiment folder:')
    print(' ', method_root)
    print('Detected files:')
    print('  forward :', forward_ckpt)
    print('  inverse :', inverse_ckpt)
    print('  seen CSV:', seen_csv, f"({bundle['seen_info'].get('rows', 0)} rows)")
    print('  unseen  :', unseen_csv, f"({bundle['unseen_info'].get('rows', 0)} rows)")
    print('  metadata:', metadata_json)
    print('  output  :', output_path)
    print('=' * 100)

    make_figure(
        str(output_path),
        str(forward_ckpt),
        str(inverse_ckpt),
        str(seen_csv),
        str(unseen_csv),
        str(metadata_json),
        sample_source=args.sample_source,
        sample_index=args.sample_index,
        explicit_amplitudes=amps,
    )
    print('Saved to:', output_path)
