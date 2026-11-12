#!/usr/bin/env python3
"""Build redesigned conceptual framework figure for Paper 1 (Fig. 2).

Changes from v1:
- 'Infrastructure exposure' removed from Hazards tier; not a physical consequence of GW decline.
- Tier 3 split into 'Physical hazards' (Land subsidence, Saline intrusion) and
  'Socio-spatial context' (Exposure, Vulnerability) — matching the CGSRI H-E-V framework.
- 'GNSS benchmarks' removed from EO inputs; replaced with DEM/Administrative data
  as the analytical source for the Vulnerability component.
- EO inputs now carry component labels (→ GW Hazard, etc.) to show data-to-component mapping.
- Driver arrows now land at their corresponding x-position on Core Process, not all at centre.
- Consistent palette, font family, and line weights throughout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGS = Path('outputs/figures')
FIGS.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
C_CLIMATE  = '#1A6FAA'  # blue
C_URBAN    = '#2E8B57'  # green
C_ABSTRACT = '#7B3F00'  # brown
C_CORE     = '#8B0000'  # dark red
C_SUBSID   = '#8B4513'  # sienna
C_SALINE   = '#2C6E8A'  # teal
C_EXPOSURE = '#4A5A8A'  # slate blue
C_VULN     = '#6A3880'  # muted purple
C_CGSRI    = '#A50026'  # deep red
C_EO       = '#2E5B8A'  # steel blue
C_DIV      = '#CCCCCC'  # dividers
C_ARROW    = '#444444'
C_ARROW_LO = '#7799BB'  # lighter arrows (effects → CGSRI, EO → CGSRI)

FONT   = 'DejaVu Sans'
FS_ROW =  8.0   # row labels (left margin)
FS_DRV =  9.0   # driver boxes
FS_CORE=  9.0   # core process
FS_HAZ =  8.5   # hazard boxes
FS_CTX =  8.5   # context (exposure/vuln) boxes
FS_SUB =  6.5   # sub-labels inside boxes
FS_CGSR= 10.0   # CGSRI title
FS_CSUB=  8.0   # CGSRI sub-label
FS_EO  =  7.5   # EO boxes
FS_CMP =  6.0   # component labels below EO boxes


def draw_box(ax, x, y, w, h, label, sublabel='', color='#4477AA',
             fontsize=9, subsize=7.0, lw=1.2):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.09',
                          fc=color, ec='#333333', lw=lw, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                fontfamily=FONT, zorder=4)
        ax.text(x + w/2, cy - 0.30, sublabel, ha='center', va='center',
                fontsize=subsize, color='#FFE0E0', fontfamily=FONT,
                style='italic', zorder=4, multialignment='center')
    else:
        ax.text(x + w/2, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                fontfamily=FONT, multialignment='center', zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))


# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14.0, 9.0))
plt.rcParams.update({'font.family': FONT, 'font.size': 9})
ax.set_xlim(-1.2, 13.5)   # negative left margin gives room for row labels
ax.set_ylim(0, 9.0)
ax.axis('off')

# ════════════════════════════════════════════════════════════════
# TIER 1 — Drivers   (y 7.3–8.2)
# ════════════════════════════════════════════════════════════════
T1_Y, T1_H, T1_W = 7.30, 0.85, 3.0

drivers = [
    (0.55,  'Climate\nvariability',       C_CLIMATE),
    (5.0,   'Urban\ngrowth',              C_URBAN),
    (9.45,  'Groundwater\nabstraction',   C_ABSTRACT),
]
for xd, lbl, col in drivers:
    draw_box(ax, xd, T1_Y, T1_W, T1_H, lbl, color=col, fontsize=FS_DRV)

# ════════════════════════════════════════════════════════════════
# TIER 2 — Core Process   (y 5.75–6.7)
# ════════════════════════════════════════════════════════════════
T2_Y, T2_H = 5.75, 0.90
draw_box(ax, 2.5, T2_Y, 8.5, T2_H,
         'Groundwater storage decline / aquifer stress',
         color=C_CORE, fontsize=FS_CORE)

# Arrows: each driver → its x-position on Core Process
for xd, _, _ in drivers:
    ax_mid = xd + T1_W / 2
    cx = max(2.5, min(11.0, ax_mid))   # clamp to Core Process box x-range
    draw_arrow(ax, ax_mid, T1_Y, cx, T2_Y + T2_H, color='#333333', lw=1.5)

# ════════════════════════════════════════════════════════════════
# TIER 3 — Physical hazards (left) + Socio-spatial context (right)
#          (y 4.05–4.95)
# ════════════════════════════════════════════════════════════════
T3_Y, T3_H = 4.05, 0.85

# Physical hazards
hazards = [
    (0.55, 'Land\nsubsidence', C_SUBSID),
    (3.45, 'Saline\nintrusion', C_SALINE),
]
CORE_LEFT, CORE_RIGHT = 2.5, 11.0  # Core Process box x-bounds
for xh, lbl, col in hazards:
    draw_box(ax, xh, T3_Y, 2.5, T3_H, lbl, color=col, fontsize=FS_HAZ)
    mid_x = xh + 1.25
    # Arrow starts from bottom of Core Process (clamped to its x-range) → top of hazard box
    start_x = max(CORE_LEFT + 0.05, min(CORE_RIGHT - 0.05, mid_x))
    draw_arrow(ax, start_x, T2_Y, mid_x, T3_Y + T3_H, color='#666666', lw=1.2)

# Socio-spatial context (right half)
context = [
    (6.9,  'Exposure',
     C_EXPOSURE,
     'Population · buildings\nroads · coastal proximity'),
    (9.9,  'Vulnerability',
     C_VULN,
     'Water access · elevation\ninformal settlements'),
]
for xc, lbl, col, sub in context:
    draw_box(ax, xc, T3_Y, 2.8, T3_H, lbl, sublabel=sub,
             color=col, fontsize=FS_CTX, subsize=FS_SUB)

# Sub-group labels beneath tier 3
ax.text(2.8,  T3_Y - 0.28, 'Physical hazards',
        ha='center', va='center', fontsize=6.8, color='#666666',
        style='italic', fontfamily=FONT)
ax.text(10.15, T3_Y - 0.28, 'Socio-spatial context',
        ha='center', va='center', fontsize=6.8, color='#666666',
        style='italic', fontfamily=FONT)

# Thin dashed separator between hazards and context
ax.axvline(6.3, ymin=(T3_Y - 0.05)/9.0, ymax=(T3_Y + T3_H + 0.05)/9.0,
           color='#AAAAAA', lw=0.7, ls='--', zorder=0)

# ════════════════════════════════════════════════════════════════
# TIER 4 — Integration / CGSRI   (y 2.3–3.5)
# ════════════════════════════════════════════════════════════════
T4_Y, T4_H = 2.30, 1.10
draw_box(ax, 0.9, T4_Y, 11.7, T4_H,
         'Coastal Groundwater–Subsidence Risk Index (CGSRI)',
         sublabel='GW Hazard  ·  Subsidence Hazard  ·  Climate Stress  ·  Exposure  ·  Vulnerability',
         color=C_CGSRI, fontsize=FS_CGSR, subsize=FS_CSUB)

# Arrows: Physical hazards → CGSRI
for xh, _, _ in hazards:
    draw_arrow(ax, xh + 1.25, T3_Y, xh + 1.25, T4_Y + T4_H,
               color=C_ARROW_LO, lw=1.1)

# Arrows: Context → CGSRI
for xc, _, _, _ in context:
    draw_arrow(ax, xc + 1.4, T3_Y, xc + 1.4, T4_Y + T4_H,
               color=C_ARROW_LO, lw=1.1)

# ════════════════════════════════════════════════════════════════
# TIER 5 — EO Inputs   (y 0.25–1.25)
# ════════════════════════════════════════════════════════════════
T5_Y, T5_H, T5_W = 0.25, 0.90, 2.25

eo_inputs = [
    (0.30,  'GRACE-FO\nGLDAS',             '→ GW Hazard'),
    (2.75,  'Sentinel-1\nOPERA-DISP',      '→ Subsidence Hazard'),
    (5.20,  'CHIRPS\nTerraClimate',        '→ Climate Stress'),
    (7.65,  'GHSL · OSM\nWorldPop',        '→ Exposure'),
    (10.10, 'DEM · Census\nAdm. data',     '→ Vulnerability'),
]
for xi, lbl, cmp in eo_inputs:
    draw_box(ax, xi, T5_Y, T5_W, T5_H, lbl, color=C_EO, fontsize=FS_EO)
    # Component label below the box
    ax.text(xi + T5_W/2, T5_Y - 0.22, cmp,
            ha='center', va='top', fontsize=FS_CMP, color='#5577AA',
            style='italic', fontfamily=FONT)
    draw_arrow(ax, xi + T5_W/2, T5_Y + T5_H, xi + T5_W/2, T4_Y,
               color=C_ARROW_LO, lw=0.9)

# ════════════════════════════════════════════════════════════════
# Row labels (left margin)
# ════════════════════════════════════════════════════════════════
row_labels = [
    (-1.15, T1_Y + T1_H/2,       'Drivers'),
    (-1.15, T2_Y + T2_H/2,       'Core\nprocess'),
    (-1.15, T3_Y + T3_H/2,       'Hazards /\ncontext'),
    (-1.15, T4_Y + T4_H/2,       'Integration'),
    (-1.15, T5_Y + T5_H/2,       'EO\ninputs'),
]
for xl, yl, lbl in row_labels:
    ax.text(xl, yl, lbl, fontsize=FS_ROW, color='#444444', style='italic',
            va='center', ha='left', fontfamily=FONT, multialignment='left')

# ════════════════════════════════════════════════════════════════
# Horizontal dividers
# ════════════════════════════════════════════════════════════════
for yd in [7.10, 5.55, 3.85, 2.15, 1.20]:
    ax.axhline(yd, xmin=0.072, xmax=0.995,
               color=C_DIV, lw=0.5, ls='--', zorder=0)

# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════
out = FIGS / 'fig00_conceptual_framework.png'
fig.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}  ({out.stat().st_size // 1024} KB)')
