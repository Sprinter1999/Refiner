import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.utils import compute_accuracy, mkdirs
from algorithms.symmetricCE import SCELoss


def train_net_fedrefiner(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", logger=None, is_warmup=True):
    """FedRefiner本地训练函数"""
    net.cuda()
    
    # 只使用SGD优化器
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)
    
    num_classes = net.l3.out_features if hasattr(net, 'l3') else args.num_class
    
    for epoch in range(epochs):
        epoch_loss_collector = []
        
        for batch_idx, (x, target, idx) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            target = target.long()
            
            _, feat, out = net(x)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            
            if is_warmup:
                # 第一阶段：使用SCE进行预热
                sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
                loss = sce_criterion(out, target)
            else:
                # 第二阶段：GMM识别噪声样本 + 标签修正 + 对比损失
                loss = train_net_fedrefiner_stage2(net, x, target, out, feat, args, num_classes)
            
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())
        
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    
    net.to('cpu')
    return None


def train_net_fedrefiner_stage2(net, x, target, out, feat, args, num_classes):
    """FedRefiner第二阶段训练逻辑"""
    # 计算交叉熵损失用于GMM
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    ce_losses = ce_loss_fn(out, target).detach().cpu().numpy().reshape(-1, 1)
    
    # 如果batch size太小，直接使用SCE
    if len(ce_losses) < 2:
        sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
        return sce_criterion(out, target)
    
    # 使用GMM识别噪声样本
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(ce_losses)
    labels_gmm = gmm.predict(ce_losses)
    means = gmm.means_.flatten()
    clean_label = np.argmin(means)
    clean_mask = torch.tensor(labels_gmm == clean_label, dtype=torch.bool, device=out.device)
    noisy_mask = ~clean_mask
    
    # 计算模型预测置信度
    probs = F.softmax(out, dim=1)
    confidence, predicted_labels = torch.max(probs, dim=1)
    
    # 标签修正：对高置信度的噪声样本进行标签修正
    tao = getattr(args, 'tao', 0.8)  # 置信度阈值
    corrected_targets = target.clone()
    
    # 对于噪声样本且置信度高于阈值的样本，使用模型预测作为新标签
    high_conf_noisy_mask = noisy_mask & (confidence > tao)
    corrected_targets[high_conf_noisy_mask] = predicted_labels[high_conf_noisy_mask]
    
    # 计算SCE损失（使用修正后的标签）
    sce_criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
    sce_loss = sce_criterion(out, corrected_targets)
    
    # 计算对比损失（基于TriTAN中的loss_2）
    contrastive_loss = compute_contrastive_loss(feat, corrected_targets, args)
    
    # 总损失
    contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
    total_loss = sce_loss + contrastive_weight * contrastive_loss
    
    return total_loss


def compute_contrastive_loss(feats, targets, args):
    """计算对比损失，基于TriTAN中的loss_2"""
    # 确保特征维度正确
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)
    
    # 归一化特征
    feats = F.normalize(feats, p=2, dim=1)
    
    # 计算特征相似度矩阵
    sim_mat = torch.matmul(feats, feats.t())
    
    # 构建正负样本掩码
    pos_mask = targets.expand(targets.shape[0], targets.shape[0]).t() == targets.expand(targets.shape[0], targets.shape[0])
    neg_mask = ~pos_mask
    
    # 硬负样本挖掘
    hard_neg_mask = neg_mask & (sim_mat > 0.5)
    pos_mask = pos_mask & (sim_mat < (1 - 1e-5))
    
    # 计算对比损失
    pos_pair = sim_mat[pos_mask]
    neg_pair = sim_mat[hard_neg_mask]
    
    if len(pos_pair) > 0 and len(neg_pair) > 0:
        pos_loss = torch.sum(-pos_pair + 1)
        neg_loss = torch.sum(neg_pair)
        contrastive_loss = (pos_loss + neg_loss) / targets.shape[0]
    else:
        contrastive_loss = torch.tensor(0.0, device=feats.device, requires_grad=True)
    
    return contrastive_loss


def fedrefiner_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger):
    """FedRefiner算法主函数"""
    best_test_acc = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_loss = float('inf')
    record_test_acc_list = []
    record_f1_list = []
    record_precision_list = []
    record_recall_list = []
    record_loss_list = []
    
    # 获取预热轮数
    warmup_rounds = getattr(args, 'fedrefiner_warmup', 5)
    
    # 第一阶段：预热阶段（使用SCE）
    logger.info("=== FedRefiner Stage 1: Warmup with SCE ===")
    print("=== FedRefiner Stage 1: Warmup with SCE ===")
    for round in range(warmup_rounds):
        logger.info(f"FedRefiner Warmup Round {round}")
        print(f"FedRefiner Warmup Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        
        # 本地训练（预热阶段）
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            train_net_fedrefiner(net_id, net, global_model, train_dl_local, test_dl, args.epochs, 
                               args.lr, args.optimizer, args, device, logger, is_warmup=True)
        
        # 模型聚合
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        
        global_w = None
        for idx, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if idx == 0:
                global_w = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
            else:
                for k in global_w:
                    global_w[k] += net_para[k] * fed_avg_freqs[idx]
        
        global_model.load_state_dict(global_w)
        global_model.cuda()
        
        # 测试
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        record_f1_list.append(f1)
        record_precision_list.append(precision)
        record_recall_list.append(recall)
        record_loss_list.append(avg_loss)
        global_model.to('cpu')
        
        # 更新最佳结果
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        logger.info(f'Warmup Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        print(f'Warmup Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
    
    logger.info("=== FedRefiner Stage 2: GMM + Label Correction + Contrastive Learning ===")
    print("=== FedRefiner Stage 2: GMM + Label Correction + Contrastive Learning ===")
    
    # 第二阶段：GMM + 标签修正 + 对比学习
    for round in range(warmup_rounds, n_comm_rounds):
        logger.info(f"FedRefiner Stage 2 Round {round}")
        print(f"FedRefiner Stage 2 Round {round}")
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        global_w = global_model.state_dict()
        
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        
        # 本地训练（第二阶段）
        for net_id, net in nets_this_round.items():
            train_dl_local = train_local_dls[net_id]
            # 直接调用第二阶段的训练逻辑
            net.cuda()
            # 只使用SGD优化器
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
            num_classes = net.l3.out_features if hasattr(net, 'l3') else args.num_class
            
            for epoch in range(args.epochs):
                for batch_idx, (x, target, idx) in enumerate(train_dl_local):
                    x, target = x.cuda(), target.cuda()
                    optimizer.zero_grad()
                    target = target.long()
                    
                    _, feat, out = net(x)
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)
                    
                    # 第二阶段：GMM识别噪声样本 + 标签修正 + 对比损失
                    loss = train_net_fedrefiner_stage2(net, x, target, out, feat, args, num_classes)
                    
                    loss.backward()
                    optimizer.step()
            
            net.to('cpu')
        
        # 模型聚合
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
        
        if args.disco:  # Discrepancy-aware collaboration
            from disco import get_distribution_difference, disco_weight_adjusting
            distribution_difference = get_distribution_difference(traindata_cls_counts, participation_clients=party_list_this_round, metric=args.measure_difference, hypo_distribution=global_dist)
            fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs, distribution_difference, args.disco_a, args.disco_b)
        
        global_w = None
        for idx, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if idx == 0:
                global_w = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
            else:
                for k in global_w:
                    global_w[k] += net_para[k] * fed_avg_freqs[idx]
        
        global_model.load_state_dict(global_w)
        global_model.cuda()
        
        # 测试
        test_acc, f1, precision, recall, conf_matrix, avg_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
        record_test_acc_list.append(test_acc)
        record_f1_list.append(f1)
        record_precision_list.append(precision)
        record_recall_list.append(recall)
        record_loss_list.append(avg_loss)
        global_model.to('cpu')
        
        # 更新最佳结果
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if f1 > best_f1:
            best_f1 = f1
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        logger.info(f'Stage 2 Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        logger.info(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        print(f'Stage 2 Round {round}: Acc={test_acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Loss={avg_loss:.4f}')
        print(f'Best so far: Acc={best_test_acc:.4f}, F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, Loss={best_loss:.4f}\n')
        
        # 保存模型
        if args.save_model:
            import os
            mkdirs(args.modeldir + 'fedrefiner/')
            torch.save(global_model.state_dict(), args.modeldir + 'fedrefiner/' + 'globalmodel' + args.log_file_name + '.pth')
    
    # 计算最后10轮的平均值
    def last_k_avg(lst, k=10):
        return np.mean(lst[-k:]) if len(lst) >= k else np.mean(lst)
    
    avg_acc = last_k_avg(record_test_acc_list)
    avg_f1 = last_k_avg(record_f1_list)
    avg_precision = last_k_avg(record_precision_list)
    avg_recall = last_k_avg(record_recall_list)
    avg_loss = last_k_avg(record_loss_list)
    
    logger.info(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    print(f'Last 10 rounds average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Loss={avg_loss:.4f}')
    
    return record_test_acc_list, best_test_acc
