import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from configs import config
from model.RPCANetPP import RPCANet
from datasets.vs_ds import Normalized, PadImg, STARE_MEAN, STARE_STD

# config
weight_path = config.weight_file
image_path = "test.png"
save_path = "./figures/stage_vis.png"
stage_indices = None
threshold = config.thre

if __name__ == '__main__':
    model = RPCANet(
        in_c = config.image_channel,
        hid_c = config.channel_expansion,
        lo = config.lo,
        ld = config.ld,
        stage_num = config.stage_num
    ).to(config.device)
    model.load_state_dict(torch.load(weight_path, map_location = config.device))
    model.eval()

    img = np.array(Image.open(image_path).convert('I'), dtype = np.float32)
    img = PadImg(Normalized(img, STARE_MEAN, STARE_STD))
    inp = torch.from_numpy(np.ascontiguousarray(img[None, None, :, :])).to(config.device)

    stage_D, stage_O = [], []
    with torch.no_grad():
        Dk, Ok = inp, torch.zeros_like(inp)
        Bh = torch.zeros(inp.shape[0], config.channel_expansion, inp.shape[2], inp.shape[3]).to(config.device)
        Bc = torch.zeros_like(Bh)

        for stage in model.net:
            Dk, Ok, Bh, Bc = stage(Dk, Ok, Bh, Bc)
            stage_D.append(Dk.squeeze().cpu().numpy())
            stage_O.append(torch.sigmoid(Ok).squeeze().cpu().numpy())

    n_stage = len(stage_D)
    idx = stage_indices if stage_indices is not None else list(range(n_stage))

    D_pix = [stage_D[i] * STARE_STD + STARE_MEAN for i in idx]
    O_prob = [stage_O[i] for i in idx]

    img_pix = img * STARE_STD + STARE_MEAN

    n_row = len(idx)
    fig, axes = plt.subplots(n_row, 2, figsize = (10, 4.2 * n_row))
    if n_row == 1:
        axes = axes[None, :]

    for r, (d, o) in enumerate(zip(D_pix, O_prob)):
        ax_d, ax_o = axes[r, 0], axes[r, 1]

        ax_d.imshow(d, cmap = 'gray', vmin = 0, vmax = 255)
        ax_d.set_title(f"Stage {idx[r]} : D (restored)")
        ax_d.axis('off')

        im = ax_o.imshow(o, cmap = 'jet', vmin = 0, vmax = 1)
        ax_o.set_title(f"Stage {idx[r]} : O (object prob.)  >{threshold}: {(o > threshold).mean():.3f}")
        ax_o.axis('off')
        fig.colorbar(im, ax = ax_o, fraction = 0.046, pad = 0.02)

    ax_in = axes[0, 0].inset_axes([0.02, 0.62, 0.36, 0.36])
    ax_in.imshow(img_pix, cmap = 'gray', vmin = 0, vmax = 255)
    ax_in.set_title("input", fontsize = 8, pad = 2)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_edgecolor('red'); sp.set_linewidth(1.2)

    plt.subplots_adjust(left = 0.02, right = 0.98, top = 0.98, bottom = 0.02,
                        wspace = 0.06, hspace = 0.10)
    plt.savefig(save_path, dpi = 120)
