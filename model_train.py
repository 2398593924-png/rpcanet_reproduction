import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs import config

from torch.optim import Adam
from utils.SoftIOU import SoftIOU
from utils.GetLoder import GetLoader
from model.RPCANetPP import RPCANet
from utils.metrics import compute_iou, compute_f1, compute_mae

from torch.optim.lr_scheduler import PolynomialLR

if __name__ == '__main__':
    
    TrainLoader, ValLoader = GetLoader(
        path = config.dataset_path,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        pin_memory = config.pin_memory,
        split_ratio = config.split_ratio
    )

    model = RPCANet(
        in_c = config.image_channel,
        hid_c = config.channel_expansion,
        lo = config.lo,
        ld = config.ld,
        stage_num = config.stage_num
    ).to(config.device)

    criterion = SoftIOU(sigma = config.sigma)
    optimizer = Adam(
        params = model.parameters(),
        lr = config.learning_rate
    )
    scheduler = PolynomialLR(
        optimizer = optimizer,
        total_iters = len(TrainLoader) * config.epochs,
        power = 0.9
    )
    best_iou = -1.0
    train_loss, val_loss_hist = [], []
    iou_hist, f1_hist, mae_hist = [], [], []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch_idx, (img, label) in enumerate(TrainLoader):
            img = img.to(config.device)
            label = label.to(config.device)
            optimizer.zero_grad()

            pred_D, pred_O = model(img)

            loss = criterion(pred_O, label, pred_D, img)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
        
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{config.epochs}] "
                    f"Iter [{batch_idx}/{len(TrainLoader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        #========================================================================================================
        model.eval()
        val_loss = 0.0
        iou_sum, f1_sum, mae_sum = 0.0, 0.0, 0.0
        n_batches = 0

        with torch.no_grad():
            for img, label in ValLoader:
                img = img.to(config.device)
                label = label.to(config.device)

                pred_D, pred_O = model(img)
                loss = criterion(pred_O, label, pred_D, img)
                val_loss += loss.item()

                iou_sum += compute_iou(pred_O, label)
                f1_sum  += compute_f1(pred_O, label)
                mae_sum += compute_mae(pred_D, img)

                n_batches += 1

        val_loss /= max(n_batches, 1)
        iou_avg = iou_sum / max(n_batches, 1)
        f1_avg  = f1_sum  / max(n_batches, 1)
        mae_avg = mae_sum / max(n_batches, 1)

        print(f"[Eval] Epoch {epoch} | "
            f"Loss: {val_loss:.4f} | "
            f"IoU: {iou_avg:.4f} | "
            f"F1: {f1_avg:.4f} | "
            f"MAE: {mae_avg:.4f}")

        train_loss.append(train_loss_sum / max(len(TrainLoader), 1))
        val_loss_hist.append(val_loss)
        iou_hist.append(iou_avg)
        f1_hist.append(f1_avg)
        mae_hist.append(mae_avg)

        if iou_avg > best_iou:
            best_iou = iou_avg
            torch.save(model.state_dict(), "./weight/best.pth")
        torch.save(model.state_dict(), "./weight/last.pth")

    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_loss, label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, val_loss_hist, color='orange', label='Val Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Val Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, f1_hist, label='F1')
    axes[1, 0].plot(epochs, iou_hist, label='IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('F1 / IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, mae_hist, color='green', label='MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('train_curve.png', dpi=150)