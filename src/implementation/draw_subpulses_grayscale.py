"""8 Gaussian sub-pulses — uniform amplitude, grayscale progressive darkening (left→right).

Two variants:
  (a) filled grayscale + black outline
  (b) outline only (each pulse has its own grayscale)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({'font.size': 14})

spatial_centers = [-28, -20, -12, -4, 4, 12, 20, 28]
# 4 amplitude levels, each exactly twice, no adjacent duplicates
# Pos: -28, -20, -12, -4,  4,  12, 20, 28
AMP_BY_POS = {
    -28: 1.00, -20: 0.25, -12: 0.75,  -4: 0.50,
      4: 1.00,  12: 0.25,  20: 0.75,  28: 0.50,
}
width = 1.2
t = np.linspace(-35, 35, 2000)

# gray_r: white=0 (lightest) → black=1 (darkest)
# 8 unique levels, t=4 is lightest, progressively darker toward both edges
# Spatial: -28, -20, -12, -4, 4, 12, 20, 28
# Gray lvl:  7,   5,   3,   1,  0,  2,  4,  6   (0=lightest, 7=darkest)
GRAY_LEVELS_BY_POS = {
    -28: 7, -20: 5, -12: 3, -4: 1,
      4: 0,  12: 2,  20: 4,  28: 6,
}
cmap = cm.gray_r
norm = Normalize(vmin=0, vmax=7)


def _setup_axes(ax):
    ax.set_xlim(-35, 35)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(spatial_centers)
    ax.set_xticklabels(spatial_centers)
    ax.set_xlabel(r'$t$', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Amplitude', labelpad=10,
                  fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    for y_level in [0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y=y_level, linestyle='--', color='gray', alpha=0.3, linewidth=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(14)


def _set_all_text_times14(fig):
    for text_obj in fig.findobj(matplotlib.text.Text):
        text_obj.set_fontfamily('Times New Roman')
        text_obj.set_fontsize(14)
        text_obj.set_fontweight('bold')


def _add_colorbar(fig, ax):
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.015, pos.height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Pulse index', fontfamily='Times New Roman', fontsize=14,
                 fontweight='bold', labelpad=8)
    cb.set_ticks(range(8))
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(14)


def _draw_pulses(ax):
    for center in spatial_centers:
        amp = AMP_BY_POS[center]
        pulse = amp * np.exp(-((t - center) / width) ** 2)
        level = GRAY_LEVELS_BY_POS[center]
        color = cmap(norm(level))
        yield center, pulse, color


# ---- Variant A: filled + black outline ----
fig_a, ax_a = plt.subplots(figsize=(12, 4))
for center, pulse, color in _draw_pulses(ax_a):
    ax_a.fill_between(t, pulse, color=color, alpha=0.92)
    ax_a.plot(t, pulse, color='black', linewidth=1.5)
_setup_axes(ax_a)
_add_colorbar(fig_a, ax_a)
_set_all_text_times14(fig_a)
fig_a.savefig(r'./subpulses_grayscale_filled.png',
              dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig_a)

# ---- Variant B: outline only (grayscale line) ----
fig_b, ax_b = plt.subplots(figsize=(12, 4))
for center, pulse, color in _draw_pulses(ax_b):
    ax_b.plot(t, pulse, color=color, linewidth=2.5)
_setup_axes(ax_b)
_add_colorbar(fig_b, ax_b)
_set_all_text_times14(fig_b)
fig_b.savefig(r'./subpulses_grayscale_outline.png',
              dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig_b)

print("Saved:")
print("  subpulses_grayscale_filled.png")
print("  subpulses_grayscale_outline.png")
