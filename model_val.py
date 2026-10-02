import torch
import numpy as np
from PIL import Image

from configs import config
from model.RPCANetPP import RPCANet
from datasets.vs_ds import Normalized, PadImg, STARE_MEAN, STARE_STD

weight_path = config.weight_file
image_path = "test.png"

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
    img = torch.from_numpy(np.ascontiguousarray(img[None, None, :, :])).to(config.device)

    with torch.no_grad():
        D_pred, O_pred = model(img)

    D_pred = D_pred.squeeze().cpu().numpy() * STARE_STD + STARE_MEAN
    O_pred = torch.sigmoid(O_pred).squeeze().cpu().numpy()

    Image.fromarray(D_pred.astype(np.uint8)).save("./figures/pred_D.png")
    Image.fromarray((O_pred * 255).astype(np.uint8)).save("./figures/pred_O.png")
