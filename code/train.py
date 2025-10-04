import torch
import numpy as np
import pandas as pd
from MLPEncoder import MLPEncoder
from utils import *
from CLLoss import Contrastive_Loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import os
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_num", type=int, default=5, help="Number of bins divided (default=5)")
    parser.add_argument("--device", type=str, default="cpu", help="Device(s) Used for Training (default='cpu')")
    parser.add_argument("--seed", type=int, default=2024, help="Random Seed (default=2024)")

    args = parser.parse_args()
    bin_num = args.bin_num
    device = args.device
    seed = args.seed

    set_seed(seed)

    path = './data/parkinsons_updrs.data'
    data = pd.read_csv(path, sep=',')
    data = data.values

    X = data[:, 6:]
    y = data[:, 4:6]

    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.4893, random_state=seed)
    sort_idx = train_X[:, 14] ## Selected Binning Feature

    ### Equal-Width Binning ###
    sort_point = cal_uniformed_sort_point(bin_num, sort_idx)

    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/MLPEncoder'):
        os.mkdir('./model/MLPEncoder')
    if not os.path.exists(f'./model/MLPEncoder/{bin_num}_bins'):
        os.mkdir(f'./model/MLPEncoder/{bin_num}_bins')
    folder_path = f'./model/MLPEncoder/{bin_num}_bins'

    train_bin_idx = cal_bin_idx(train_X[:, 14], sort_point)
    train_bin_idx = np.expand_dims(train_bin_idx, axis=1)
    test_bin_idx = cal_bin_idx(test_X[:, 14], sort_point)
    test_bin_idx = np.expand_dims(test_bin_idx, axis=1)

    train_X = np.concatenate((train_bin_idx, train_X), axis=1)
    test_X = np.concatenate((test_bin_idx, test_X), axis=1)

    train_data = np.concatenate((train_X, train_Y), axis=1)
    test_data = np.concatenate((test_X, test_Y), axis=1)

    ss = StandardScaler()
    train_data[:, 1:] = ss.fit_transform(train_data[:, 1:])
    test_data[:, 1:] = ss.transform(test_data[:, 1:])

    train_X = train_data[:, 0:17]
    train_Y = train_data[:, 17:]
    test_X = test_data[:, 0:17]
    test_Y = test_data[:, 17:]

    test_bin_idx = torch.tensor(test_X[:, 0], dtype=torch.int)
    test_X = torch.tensor(test_X[:, 1:], dtype=torch.float)

    temprature = 1.0
    epochs = 10
    steps = 20
    model = MLPEncoder(16, 16, loss_function=Contrastive_Loss(temp=temprature, device=device), num_layers=1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(device)

    # train and validation ##
    step_list = []
    train_loss = []
    val_plt_loss = []
    val_loss = []
    kf = KFold(n_splits=epochs, shuffle=False)
    epoch = 0
    for train_idx, val_idx in kf.split(train_X):
        print(f"Epoch {epoch+1} started!")
        train_X_fold, val_X_fold = train_X[train_idx], train_X[val_idx]

        train_bin_idx = torch.tensor(train_X_fold[:, 0], dtype=torch.int)
        val_bin_idx = torch.tensor(val_X_fold[:, 0], dtype=torch.int)

        train_alpha = cal_alpha(bin_num, train_bin_idx)
        val_alpha = cal_alpha(bin_num, val_bin_idx)

        sample_embed = torch.tensor(train_X_fold[:, 1:], dtype=torch.float)
        val_X = torch.tensor(val_X_fold[:, 1:], dtype=torch.float)

        train_alpha = train_alpha.to(device)
        train_bin_idx = train_bin_idx.to(device)
        val_bin_idx = val_bin_idx.to(device)
        sample_embed = sample_embed.to(device)
        val_X = val_X.to(device)
        val_alpha = val_alpha.to(device)

        for step in range(steps):
            model.train()
            h, loss, train_bin_centr = model(sample_embed, train_bin_idx, train_alpha, bin_num)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if not (step+1) % 20:
                print(f"Global step {epoch*steps + step + 1} Finished, Training Loss={loss}.\nh={h}")
                train_loss.append(loss.detach().cpu().numpy())

            model.eval()
            stopper_count = 0
            with torch.no_grad():
                val_sample_centr = cal_centr_by_train(val_bin_idx, train_bin_centr)
                h, _, _ = model(val_X, val_bin_idx, val_alpha, bin_num)
                loss_fc = Contrastive_Loss(temp=temprature, device=device)
                train_bin_centr = train_bin_centr.to(device)
                val_sample_centr = val_sample_centr.to(device)
                loss = loss_fc(h, train_bin_centr, val_sample_centr, val_alpha)
                if not (step+1) % 20:
                    print(f"In global step {epoch*steps + step + 1}, Validation Loss={loss}")
                    step_list.append(epoch*steps + step + 1)
                    val_plt_loss.append(loss.detach().cpu().numpy())
                if (len(val_loss) >= 10 and loss < min(val_loss)):
                    torch.save(model, f"{folder_path}/{bin_num}_bins.pth")
                if (len(val_loss) >= 10 and loss > min(val_loss[step - 10:])):
                    stopper_count += 1
                    if (stopper_count >= 10):
                        print("Validation loss has not decreased for 10 consecutive steps. Stopping training.")
                        break
                else:
                    stopper_count = 0
                val_loss.append(loss)
        print(f"Epoch {epoch+1} finished!")
        epoch += 1

    save_dir = folder_path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_curve.png")

    plt.figure(figsize=(8, 6), dpi=150)

    plt.plot(step_list, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5, marker='o')
    plt.plot(step_list, val_plt_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker='s')

    plt.title("Training and Validation Loss", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Steps", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"✅ Loss Fig is saved to: {save_path}")



if __name__ == "__main__":
    main()