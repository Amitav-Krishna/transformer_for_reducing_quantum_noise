"""
Generate transformer architecture diagram with two subfigures:
(a) Architecture flow
(b) Global attention pattern
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Colors (pastel)
colors = {
    'input': '#DAE8FC',    # light blue
    'token': '#D5E8D4',    # light green
    'embed': '#FFF2CC',    # light yellow
    'encoder': '#FFD9B3',  # light orange
    'bottleneck': '#F8CECC',  # light red
    'decoder': '#FFD9B3',  # light orange
    'output': '#DAE8FC',   # light blue
}

# ============ Subfigure (a): Architecture ============
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 16)
ax1.axis('off')
ax1.set_aspect('equal')

# Box parameters
box_width = 6
box_height = 1.2
x_center = 5
spacing = 1.8

# Layers from top to bottom
layers = [
    ('32 × 32 Density Matrix', colors['input'], 14),
    ('1024 Tokens', colors['token'], 12.2),
    ('Linear (2 → 32)', colors['embed'], 10.4),
    ('Encoder (4× Self-Attn)', colors['encoder'], 8.6),
    ('Bottleneck (32 → 16 → 32)', colors['bottleneck'], 6.8),
    ('Decoder (4× Cross-Attn)', colors['decoder'], 5.0),
    ('Linear (32 → 2)', colors['embed'], 3.2),
    ('Reconstructed ρ̂', colors['input'], 1.4),
]

boxes = {}
for label, color, y in layers:
    box = FancyBboxPatch(
        (x_center - box_width/2, y - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=color,
        edgecolor='gray',
        linewidth=1.5
    )
    ax1.add_patch(box)
    ax1.text(x_center, y, label, ha='center', va='center', fontsize=10, fontweight='medium')
    boxes[label] = (x_center, y)

# Draw arrows between layers
arrow_style = dict(arrowstyle='->', color='gray', lw=1.5, mutation_scale=15)

for i in range(len(layers) - 1):
    y_start = layers[i][2] - box_height/2
    y_end = layers[i+1][2] + box_height/2
    ax1.annotate('', xy=(x_center, y_end), xytext=(x_center, y_start),
                 arrowprops=arrow_style)

# Add side labels
ax1.text(x_center + box_width/2 + 0.3, (layers[0][2] + layers[1][2])/2, 'flatten',
         fontsize=8, color='gray', va='center')
ax1.text(x_center + box_width/2 + 0.3, (layers[6][2] + layers[7][2])/2, 'reshape',
         fontsize=8, color='gray', va='center')

# Cross-attention arrow (encoder to decoder)
enc_y = layers[3][2]
dec_y = layers[5][2]
ax1.annotate('',
             xy=(x_center + box_width/2 + 0.8, dec_y),
             xytext=(x_center + box_width/2 + 0.8, enc_y),
             arrowprops=dict(arrowstyle='->', color='#E67300', lw=1.5,
                           connectionstyle='arc3,rad=0.3', linestyle='--'))
ax1.text(x_center + box_width/2 + 1.3, (enc_y + dec_y)/2, 'memory',
         fontsize=8, color='gray', va='center')

ax1.text(x_center, 0.2, '(a) Architecture', ha='center', fontsize=11, fontweight='bold')

# ============ Subfigure (b): Global Attention ============
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 16)
ax2.axis('off')
ax2.set_aspect('equal')

# Draw 8x8 grid representing density matrix
grid_size = 8
cell_size = 0.5
grid_x = 5 - (grid_size * cell_size) / 2
grid_y = 10

# Draw grid cells
for i in range(grid_size):
    for j in range(grid_size):
        rect = plt.Rectangle(
            (grid_x + i * cell_size, grid_y + j * cell_size),
            cell_size, cell_size,
            facecolor='white',
            edgecolor='lightgray',
            linewidth=0.5
        )
        ax2.add_patch(rect)

# Highlight specific cells
highlight_cells = [
    (0, 7, colors['input']),   # top-left area
    (3, 4, colors['bottleneck']),  # middle
    (6, 1, colors['token']),   # bottom-right area
]

for i, j, color in highlight_cells:
    rect = plt.Rectangle(
        (grid_x + i * cell_size, grid_y + j * cell_size),
        cell_size, cell_size,
        facecolor=color,
        edgecolor='gray',
        linewidth=1
    )
    ax2.add_patch(rect)

# Draw attention arrows between highlighted cells
def cell_center(i, j):
    return (grid_x + i * cell_size + cell_size/2,
            grid_y + j * cell_size + cell_size/2)

c1 = cell_center(0, 7)
c2 = cell_center(3, 4)
c3 = cell_center(6, 1)

# Curved arrows showing attention
arrow_props = dict(arrowstyle='->', color='#3366CC', lw=2,
                   connectionstyle='arc3,rad=0.2', mutation_scale=12)
ax2.annotate('', xy=c2, xytext=c1, arrowprops=arrow_props)
ax2.annotate('', xy=c3, xytext=c1,
             arrowprops=dict(arrowstyle='->', color='#3366CC', lw=2,
                           connectionstyle='arc3,rad=0.3', mutation_scale=12))
ax2.annotate('', xy=c3, xytext=c2,
             arrowprops=dict(arrowstyle='->', color='#CC3333', lw=2,
                           connectionstyle='arc3,rad=0.2', mutation_scale=12))

# Grid border
border = plt.Rectangle(
    (grid_x, grid_y),
    grid_size * cell_size, grid_size * cell_size,
    facecolor='none',
    edgecolor='gray',
    linewidth=1.5
)
ax2.add_patch(border)

# Labels
ax2.text(5, grid_y - 0.5, r'Density Matrix $\rho$', ha='center', fontsize=10)

# Token sequence
ax2.annotate('', xy=(5, grid_y - 1.5), xytext=(5, grid_y - 0.8),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
ax2.text(5, grid_y - 2.0, r'$[\rho_{00}, \rho_{01}, \ldots, \rho_{ij}, \ldots, \rho_{31,31}]$',
         ha='center', fontsize=9)

# Explanation box
box = FancyBboxPatch(
    (1.5, grid_y - 4.5), 7, 1.5,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#F5F5F5',
    edgecolor='lightgray',
    linewidth=1
)
ax2.add_patch(box)
ax2.text(5, grid_y - 3.75, 'Each token attends globally to all\n1024 elements in the sequence',
         ha='center', va='center', fontsize=9)

ax2.text(5, 0.2, '(b) Global attention pattern', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/transformer_architecture.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/transformer_architecture.png', bbox_inches='tight', dpi=300)
print("Saved transformer_architecture.pdf and transformer_architecture.png")
