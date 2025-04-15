import os
import time
import torch
import numpy as np

# from RCML.model import RCML

from utils.model import LLM_RC
from args import parameter_parser
from utils.Dataloader import create_multi_view_data, load_data
from utils.loss import ce_loss, get_dc_loss


args = parameter_parser()
EPOCHS = args.epochs
# BATCH_SIZE = args.batch_size
# texthead = args.texthead
annealing_epoch = args.annealing_epoch
dataset_name = args.dataset_name
out_path = f"{dataset_name}/outputs"
SAVE_PATH = f"save_model/{dataset_name}/{args.dc_loss}_best_model.pth"
DATA_PATH = f"datasets/{out_path}/multi_view-concise-pro.npz"


os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
if not os.path.exists(DATA_PATH):
    create_multi_view_data(args)

# 
args = parameter_parser()
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

# print(f"🚀 Using device: {device}")
data, n_class, idx_train, idx_val, idx_test, logger = load_data(args, f"datasets/{out_path}/")
print(f'train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}')
input_dims = [data.X[0].shape[1], data.X[1].shape[1]]  # (200, 384), (200, 130)
num_classes = len(np.unique(data.Y))
labels = data.Y

# model = build_rcml(input_dims, num_classes)
model = LLM_RC(data=data, num_classes=num_classes, dropout=args.dropout, hid=args.hid).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

x1 = data.X[0][idx_train].to(device) # shape ([20, 384])
x2 = data.X[1][idx_train].to(device) # shape ([20, 130])
target = labels[idx_train].to(device)

x1_val = data.X[0][idx_val].to(device)
x2_val = data.X[1][idx_val].to(device)
y_val = labels[idx_val].to(device)

x1_test = data.X[0][idx_test].to(device)
x2_test = data.X[1][idx_test].to(device)
y_test = labels[idx_test].to(device)

# print(target)
# Training
best_acc = 0.0
acc_val = 0.0

patience = args.patience
no_improve_counter = 0

for epoch_i in range(EPOCHS):
    t0_epoch, t0_batch = time.time(), time.time()
    total_loss, batch_loss, batch_counts = 0, 0, 0
    model.train()
    # for step, batch in enumerate(train_loader):
    alpha_1, alpha_2, alpha_a = model(x1, x2)

    if args.wo_view1:
        loss = ce_loss(target, alpha_2, num_classes, epoch_i, annealing_epoch)
                #  + \ args.dc_loss * get_dc_loss([alpha_2], device=device)
    else:
        loss = ce_loss(target, alpha_1, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_2, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_a, num_classes, epoch_i, annealing_epoch) + \
                    args.dc_loss * get_dc_loss([alpha_1,alpha_2], device=device)
        
    loss = torch.mean(loss)
    loss.backward()

    batch_loss += loss.item()
    total_loss += loss.item()

    optimizer.step()
        
    # avg_train_loss = total_loss / len(x1)

    # ----- Validation -----
    if epoch_i % 50 == 0:
        model.eval()
        with torch.no_grad():
            _, _, alpha_val = model(x1_val, x2_val)
            probs_val = alpha_val / torch.sum(alpha_val, dim=1, keepdim=True)
            preds_val = torch.argmax(probs_val, dim=1)
            correct_val = (preds_val == y_val).sum().item()
            acc_val = correct_val / len(y_val)

        print(f"Epoch {epoch_i}: Train Loss={loss.item():.4f}, Val Acc={acc_val:.4f}")

    # 保存最好模型
    if acc_val > best_acc:
        best_acc = acc_val
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved at epoch {epoch_i}, acc = {acc_val:.4f}")
        no_improve_counter = 0  # 重置
    else:
        no_improve_counter += 1
        # print(f"⚠️  No improvement in val acc for {no_improve_counter} eval(s)")

    if patience > 0 and no_improve_counter >= patience:
        print(f"🛑 Early stopping triggered at epoch {epoch_i}")
        break

# ----------- TEST -------------
model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
model.eval()


with torch.no_grad():
    alpha_1, alpha_2, alpha_a = model(x1_test, x2_test)
    probs = alpha_a / torch.sum(alpha_a, dim=1, keepdim=True)  # normalize alpha to probability
    preds = torch.argmax(probs, dim=1)
    correct = (preds == y_test).sum().item()
    acc = correct / len(y_test)

print(f"🎯 Test Accuracy: {acc:.4f}")


