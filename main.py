import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score
# from RCML.model import RCML
from sklearn.metrics import precision_score, f1_score
from utils.model import LLM_RC
from args import parameter_parser
from utils.Dataloader import create_multi_view_data, load_data
from utils.loss import ce_loss, get_dc_loss, get_soft_dc_loss
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, classification_report, balanced_accuracy_score, confusion_matrix, roc_auc_score

args = parameter_parser()
EPOCHS = args.epochs
# BATCH_SIZE = args.batch_size
# texthead = args.texthead
annealing_epoch = args.annealing_epoch
dataset_name = args.dataset_name
# out_path = f"{dataset_name}"
# 
args = parameter_parser()
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

model_dir = f"save_model/{dataset_name}"
    
os.makedirs(model_dir, exist_ok=True)

# 使用run_id创建唯一的保存路径
if args.run_id:
    SAVE_PATH = f"{model_dir}/{args.dc_loss}_{args.hid}_best_model_{args.run_id}.pth"
else:
    SAVE_PATH = f"{model_dir}/{args.dc_loss}_{args.hid}_best_model.pth"

DATA_PATH = f"datasets/{dataset_name}/multi_view-{args.embedding_type}-{args.views}.npz"


os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
if not os.path.exists(DATA_PATH):
    create_multi_view_data(args)


# print(f"🚀 Using device: {device}")
data, n_class, idx_train, idx_val, idx_test, logger = load_data(args, f"datasets/{dataset_name}/")
print(f'train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}')
# input_dims = [data.X[0].shape[1], data.X[1].shape[1]]  # (200, 384), (200, 130)
num_classes = len(np.unique(data.Y))
labels = data.Y

# model = build_rcml(input_dims, num_classes)
model = LLM_RC(data=data, num_classes=num_classes, dropout=args.dropout, hid=args.hid).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# test_save_np = {f'view_{i}': x[idx_test].cpu().numpy() for i, x in enumerate(data.X)}
# np.savez('test.npz', **test_save_np)

# idx_test = [4696,]
x_train = [x[idx_train].to(device) for x in data.X]
target = labels[idx_train].to(device)

x_val = [x[idx_val].to(device) for x in data.X]
y_val = labels[idx_val].to(device)

if not args.no_test:
    x_test = [x[idx_test].to(device) for x in data.X]
    y_test = labels[idx_test].to(device)


# print(target)
# Training
best_acc = 0.0
acc_val = 0.0

patience = args.patience
no_improve_counter = 0

criterion = torch.nn.CrossEntropyLoss()


for epoch_i in range(EPOCHS):
    t0_epoch, t0_batch = time.time(), time.time()
    total_loss, batch_loss, batch_counts = 0, 0, 0
    model.train()
    results = model(x_train)

    if len(results) == 2:
        alpha_raw, alpha_a = results
        alpha_2 = alpha_raw
        alpha_1 = alpha_raw
        
    elif len(results) == 3: 
        alpha_1, alpha_2, alpha_a = results
        # alpha_raw = alpha_2  
    else: 
        alpha_1, alpha_2, alpha_raw, alpha_a = results
        # print(len(results))
    if len(results) == 2:
        # loss =  ce_loss(target, alpha_raw, num_classes, epoch_i, annealing_epoch) 
        loss = criterion(results[-1], target) + args.dc_loss * get_soft_dc_loss([alpha_1,alpha_2, alpha_raw,], device=device)
    elif len(results) == 3:
        loss = ce_loss(target, alpha_1, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_2, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_a, num_classes, epoch_i, annealing_epoch) + \
                args.dc_loss * get_soft_dc_loss([alpha_1,alpha_2, alpha_a,], device=device)
    else:
        loss = ce_loss(target, alpha_1, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_2, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_raw, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_a, num_classes, epoch_i, annealing_epoch) + \
                    args.dc_loss * get_soft_dc_loss([alpha_1,alpha_2, alpha_raw,], device=device)
        # loss =     ce_loss(target, alpha_a, num_classes, epoch_i, annealing_epoch) + \
        #             args.dc_loss * get_soft_dc_loss([alpha_1,alpha_2, alpha_raw,], device=device)
        # loss = criterion(results[-1], target) + args.dc_loss * get_soft_dc_loss([alpha_1,alpha_2, alpha_raw,], device=device)
        
    loss = torch.mean(loss)
    loss.backward()

    # batch_loss += loss.item()
    # total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    # avg_train_loss = total_loss / len(x_train[0])

    # ----- Validation -----
    if epoch_i % 50 == 0:
        model.eval()
        with torch.no_grad():
            results_val = model(x_val)
            alpha_val = results_val[-1]  # 获取最后一个结果，即组合结果
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

# 测试阶段 - 只在不使用no_test时执行
if not args.no_test:
    try:
        # 尝试加载同一结构的模型
        model_new = LLM_RC(data=data, num_classes=num_classes, dropout=args.dropout, hid=args.hid).to(device)
        model_new.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
        model = model_new
    except Exception as e:
        print(f"Warning: Could not load saved model. Using current model. Error: {str(e)}")
    
    model.eval()
    with torch.no_grad():
        results_test = model(x_test)
        # alpha_a = results_test[-1]  # 获取最后一个结果
        alpha_a = results_test[-1]  # 获取最后一个结果
        probs = alpha_a / torch.sum(alpha_a, dim=1, keepdim=True)
        preds = torch.argmax(probs, dim=1)
        # correct = (preds == y_test).sum().item()
        # acc = correct / len(y_test)
        acc = accuracy_score(y_test.cpu(), preds.cpu())
        # print(f"🎯 Test Accuracy: {acc:.4f}")
        precision_test = precision_score(y_test.cpu(), preds.cpu(), average='macro', zero_division=0)
        precision_test2 = precision_score(y_test.cpu(), preds.cpu(), average='micro', zero_division=0)
        f1_test = f1_score(y_test.cpu(), preds.cpu(), average='macro', zero_division=0)
        f1_test2 = f1_score(y_test.cpu(), preds.cpu(), average='micro', zero_division=0)

        dict = classification_report(y_test.cpu(), preds.cpu(), digits=4,zero_division=0,output_dict=True)
        # np.save("agent1results.npy", results_test[1].detach().cpu().numpy())
        # np.save("agent2results.npy", results_test[2].detach().cpu().numpy())
       
        # # === Vis
        # results_val = model(x_val)
        # alpha_f = results_val[0]  # flow agent
        # alpha_c = results_val[1]  # context agent
    
    AP = dict['weighted avg']['precision']
    AR = dict['weighted avg']['recall']
    AF1 = dict['weighted avg']['f1-score']
    # print(pd.DataFrame(dict))
    print('===weighted avg===')
    print(f'Precision: {AP}')
    print(f'Recall: {AR}')
    print(f'F1-score: {AF1}')


    print(f"🎯 Test Accuracy: {acc:.4f}, Test PR: {precision_test:.4f}-{precision_test2:.4f}, Test F1: {f1_test:.4f}-{f1_test2:.4f}")
    cm = confusion_matrix(y_test.cpu(), preds.cpu())
    print("Confusion Matrix:")
    print(pd.DataFrame(cm))
else:
    print(f"✅ Training completed. Best validation accuracy: {best_acc:.4f}")
    print(f"Test phase skipped (--no_test flag used)")


