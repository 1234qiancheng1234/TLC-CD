from dataloader import Dataset_self
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm
from utils.utils import *

# 关键修改1: 使用正确的AMP API导入
from torch.cuda.amp import autocast, GradScaler

# 关键修改2: 确保模型有reset_memory方法
from model.HRSICD.HRSICD import HRSICD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 关键修改3: 增强的混合损失函数
class EnhancedHybridLoss(nn.Module):
    """改进的混合损失函数，增加平衡权重"""

    def __init__(self, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2, alpha=0.25, gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, pred_logits, target):
        """Focal Loss，处理类别不平衡"""
        pred_prob = torch.sigmoid(pred_logits)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        return (focal_weight * bce_loss).mean()

    def forward(self, pred_logits, target):
        # 1. BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target)

        # 2. Dice损失
        pred_prob = torch.sigmoid(pred_logits)
        pred_flat = pred_prob.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

        # 3. Focal损失
        focal_loss = self.focal_loss(pred_logits, target)

        # 4. 加权求和
        total_loss = (
                self.bce_weight * bce_loss +
                self.dice_weight * dice_loss +
                self.focal_weight * focal_loss
        )

        return total_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train LNN-CD Model')
    parser.add_argument('--train_path', default='data/shuguang/train', help='train data path')
    parser.add_argument('--val_path', default='data/shuguang/val', help='val data path')
    parser.add_argument('--train_batch_size', type=int, default=8, help='train batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='validate batch size')
    parser.add_argument('--work_dir', default='result_lnn_debug', help='the dir to save checkpoint and logs')
    parser.add_argument('--epoch', type=int, default=100, help='Total Epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--model_name', default='LNN_CD_DEBUG', help='Model name for saving')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
    # 新增：辅助损失权重
    parser.add_argument('--aux_weight', type=float, default=0.3, help='Weight for auxiliary losses (deep supervision)')
    return parser.parse_args()


def calculate_metrics(pred_prob, gt, threshold=0.5):
    """计算分类指标"""
    pred_binary = (pred_prob > threshold).float()

    # 展平
    pred_flat = pred_binary.view(-1)
    gt_flat = gt.view(-1)

    # 计算混淆矩阵
    TP = ((pred_flat == 1) & (gt_flat == 1)).sum().item()
    FP = ((pred_flat == 1) & (gt_flat == 0)).sum().item()
    FN = ((pred_flat == 0) & (gt_flat == 1)).sum().item()
    TN = ((pred_flat == 0) & (gt_flat == 0)).sum().item()

    # 计算指标
    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1e-8)
    precision = TP / max(TP + FP, 1e-8)
    recall = TP / max(TP + FN, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = TP / max(TP + FP + FN, 1e-8)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'pos_ratio': gt_flat.sum().item() / max(gt_flat.numel(), 1)
    }


def validate_with_multiple_thresholds(net, val_loader, loss_func, device, use_amp=False):
    """使用多个阈值进行验证，找到最佳阈值"""
    net.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image1_val, image2_val, gt_val in val_loader:
            # 重置记忆状态
            if hasattr(net, 'reset_memory'):
                net.reset_memory()

            image1_val = image1_val.to(device, dtype=torch.float32)
            image2_val = image2_val.to(device, dtype=torch.float32)
            gt_val = gt_val.to(device, dtype=torch.float32)

            # 验证时不需要辅助输出（with_aux=False，默认）
            with autocast(enabled=use_amp):
                pred_logits_val = net(image1_val, image2_val)
                loss_val = loss_func(pred_logits_val, gt_val)

            val_loss += loss_val.item()

            # 收集概率和标签
            pred_prob = torch.sigmoid(pred_logits_val).cpu()
            all_preds.append(pred_prob)
            all_labels.append(gt_val.cpu())

    # 合并所有批次
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 尝试多个阈值
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_metrics = None
    best_threshold = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        metrics = calculate_metrics(all_preds, all_labels, threshold=thresh)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics
            best_threshold = thresh

    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss, best_metrics, best_threshold, all_preds, all_labels


if __name__ == "__main__":
    args = parse_args()

    # 创建工作目录
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # 数据加载
    print("加载训练数据...")
    data_loader_train = torch.utils.data.DataLoader(
        dataset=Dataset_self(args.train_path),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print("加载验证数据...")
    data_loader_val = torch.utils.data.DataLoader(
        dataset=Dataset_self(args.val_path),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"训练集大小: {len(data_loader_train.dataset)}")
    print(f"验证集大小: {len(data_loader_val.dataset)}")

    # 检查数据集的类别分布
    print("\n检查数据集类别分布...")
    pos_count = 0
    total_pixels = 0

    for _, _, gt in data_loader_train:
        pos_count += gt.sum().item()
        total_pixels += gt.numel()

    pos_ratio = pos_count / total_pixels
    print(f"训练集正样本比例: {pos_ratio:.4%}")

    # 初始化模型
    net = HRSICD().to(device)

    # 检查模型是否有reset_memory方法
    if not hasattr(net, 'reset_memory'):
        print("警告: 模型没有reset_memory方法，液态神经元的记忆不会被重置！")
        net.reset_memory = lambda: None

    # 根据正样本比例调整损失函数权重
    if pos_ratio < 0.1:  # 高度不平衡
        loss_func = EnhancedHybridLoss(bce_weight=0.4, dice_weight=0.3, focal_weight=0.3, alpha=0.75)
    elif pos_ratio < 0.3:
        loss_func = EnhancedHybridLoss(bce_weight=0.5, dice_weight=0.3, focal_weight=0.2, alpha=0.5)
    else:
        loss_func = EnhancedHybridLoss(bce_weight=0.6, dice_weight=0.4, focal_weight=0.0)

    # 优化器配置
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # 使用OneCycleLR调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epoch,
        steps_per_epoch=len(data_loader_train),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # 使用AMP
    scaler = GradScaler(enabled=args.use_amp)

    # 监控指标初始化
    best_val_loss = float('inf')
    best_iou = 0.0
    best_f1 = 0.0

    # 创建保存目录
    save_path = os.path.join(args.work_dir, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 日志文件
    log_file = os.path.join(save_path, f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')
    with open(log_file, 'w') as f:
        f.write('Epoch\tLR\tTrain_Loss\tVal_Loss\tAccuracy\tF1\tIoU\tPrecision\tRecall\tThreshold\tPos_Ratio\n')

    print(f"\n开始训练LNN-CD模型，共{args.epoch}个epoch")
    print(f"设备: {device}, 批次大小: {args.train_batch_size}")
    print(f"使用混合精度: {args.use_amp}, 梯度裁剪: {args.grad_clip}")
    print(f"优化器: AdamW, 初始学习率: {args.lr}")
    print(f"损失函数: 增强混合损失 (BCE+Dice+Focal)")
    print(f"辅助损失权重: {args.aux_weight}")
    print(f"模型保存路径: {save_path}")
    print("-" * 60)

    for epoch in range(1, args.epoch + 1):
        # ========== 训练阶段 ==========
        net.train()
        train_loss = 0
        train_samples = 0

        pbar = tqdm(data_loader_train, desc=f'Epoch {epoch}/{args.epoch} [Train]')
        for batch_idx, (image1, image2, gt) in enumerate(pbar):
            # 每个批次前重置记忆状态
            net.reset_memory()

            image1 = image1.to(device, dtype=torch.float32)
            image2 = image2.to(device, dtype=torch.float32)
            gt = gt.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            # 使用AMP进行混合精度前向传播
            with autocast(enabled=args.use_amp):
                # 启用辅助输出
                pred_logits, aux2, aux1 = net(image1, image2, with_aux=True)

                # 上采样辅助输出到与gt相同的尺寸
                aux2_up = F.interpolate(aux2, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                aux1_up = F.interpolate(aux1, size=gt.shape[-2:], mode='bilinear', align_corners=False)

                # 计算主损失和辅助损失
                loss_main = loss_func(pred_logits, gt)
                loss_aux2 = loss_func(aux2_up, gt)
                loss_aux1 = loss_func(aux1_up, gt)

                # 总损失 = 主损失 + 辅助损失权重 * (aux2 + aux1)
                total_loss = loss_main + args.aux_weight * (loss_aux2 + loss_aux1)

            # 混合精度反向传播
            scaler.scale(total_loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # 学习率调度
            scheduler.step()

            batch_size = image1.size(0)
            train_loss += loss_main.item() * batch_size  # 记录主损失（便于与验证损失对比）
            train_samples += batch_size

            # 更新进度条
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                'loss': f'{loss_main.item():.4f}',
                'aux_loss': f'{(loss_aux2.item() + loss_aux1.item()):.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 调试模式
            if args.debug and batch_idx % 5 == 0:
                with torch.no_grad():
                    pred_prob = torch.sigmoid(pred_logits)
                    pred_binary = (pred_prob > 0.5).float()

                    batch_pos_ratio = gt.sum().item() / gt.numel()
                    pred_pos_ratio = pred_binary.sum().item() / pred_binary.numel()

                    print(f"[训练批次 {batch_idx}] 标签正样本: {batch_pos_ratio:.3%}, "
                          f"预测正样本: {pred_pos_ratio:.3%}, 主损失: {loss_main.item():.4f}")

        avg_train_loss = train_loss / train_samples

        # ========== 验证阶段 ==========
        avg_val_loss, val_metrics, best_threshold, _, _ = validate_with_multiple_thresholds(
            net, data_loader_val, loss_func, device, args.use_amp
        )

        avg_accuracy = val_metrics['accuracy']
        avg_precision = val_metrics['precision']
        avg_recall = val_metrics['recall']
        avg_f1 = val_metrics['f1']
        avg_iou = val_metrics['iou']
        pos_ratio = val_metrics['pos_ratio']

        current_lr = optimizer.param_groups[0]['lr']

        # ========== 保存策略 ==========
        save_models = []

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_path, f'best_loss_epoch{epoch}_loss{avg_val_loss:.4f}.pth')
            save_models.append(('最佳损失', model_path))

        if avg_iou > best_iou:
            best_iou = avg_iou
            model_path = os.path.join(save_path, f'best_iou_epoch{epoch}_iou{avg_iou:.4f}.pth')
            save_models.append(('最佳IoU', model_path))

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            model_path = os.path.join(save_path, f'best_f1_epoch{epoch}_f1{avg_f1:.4f}.pth')
            save_models.append(('最佳F1', model_path))

        if epoch % 5 == 0:
            model_path = os.path.join(save_path, f'checkpoint_epoch{epoch}.pth')
            save_models.append(('定期检查点', model_path))

        for reason, model_path in save_models:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'accuracy': avg_accuracy,
                'f1': avg_f1,
                'iou': avg_iou,
                'precision': avg_precision,
                'recall': avg_recall,
                'threshold': best_threshold,
                'pos_ratio': pos_ratio
            }, model_path)
            print(f"  ✓ 保存模型 ({reason}): {os.path.basename(model_path)}")

        # ========== 打印和记录日志 ==========
        print(f"\nEpoch {epoch:04d}/{args.epoch}:")
        print(f"  LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  最佳阈值: {best_threshold:.3f} | 正样本比例: {pos_ratio:.3%}")
        print(f"  Accuracy: {avg_accuracy:.4f} | F1: {avg_f1:.4f} | IoU: {avg_iou:.4f}")
        print(f"  Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f}")
        print(f"  最佳指标: Loss={best_val_loss:.4f}, IoU={best_iou:.4f}, F1={best_f1:.4f}")

        with open(log_file, 'a') as f:
            f.write(f'{epoch}\t{current_lr:.2e}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t'
                    f'{avg_accuracy:.4f}\t{avg_f1:.4f}\t{avg_iou:.4f}\t{avg_precision:.4f}\t'
                    f'{avg_recall:.4f}\t{best_threshold:.3f}\t{pos_ratio:.4f}\n')

        print("-" * 60)

        if epoch > 20 and avg_val_loss > best_val_loss * 1.5:
            print(f"⚠️  警告：验证损失显著增加，可能过拟合。考虑早停。")

        if epoch > 10 and avg_f1 < 0.2:
            print(f"⚠️  警告：F1分数过低 ({avg_f1:.4f})，模型可能没有学习到有效特征。")
            print(f"考虑：1. 检查数据质量 2. 调整损失函数 3. 简化模型结构")

    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳IoU: {best_iou:.4f}")
    print(f"最佳F1: {best_f1:.4f}")
    print(f"所有结果保存在: {save_path}")
    print(f"日志文件: {log_file}")