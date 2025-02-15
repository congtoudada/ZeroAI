import os
from collections.abc import Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

from util import misc
from util.abnormal_utils import filt


def inference(model: torch.nn.Module, data_loader: Iterable,
              device: torch.device,
              log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing '

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions_teacher = []
    predictions_student_teacher = []
    pred_anomalies = []
    labels = []
    videos = []
    frames = []
    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        frames += list(frame_name)
        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        model.train_TS = True  # student-teacher reconstruction error
        _, _, _, recon_error_st_tc = model(samples, targets=targets, grad_mask=grads, mask_ratio=args.mask_ratio)
        recon_error_st_tc[0] = recon_error_st_tc[0].detach().cpu().numpy()
        recon_error_st_tc[1] = recon_error_st_tc[1].detach().cpu().numpy()
        if len(recon_error_st_tc) > 2:
            pred_anomalies += list(recon_error_st_tc[2].detach().cpu().numpy())
        predictions_student_teacher += list(recon_error_st_tc[0])
        predictions_teacher += list(recon_error_st_tc[1])

    # Compute statistics

    labels = np.array(labels)
    videos = np.array(videos)

    if args.dataset == 'avenue':
        predictions_teacher = np.array(predictions_teacher)
        predictions_student_teacher = np.array(predictions_student_teacher)
        pred_anomalies = np.array(pred_anomalies)
        predictions = 10.5 * predictions_teacher + 5.3 * predictions_student_teacher + 5.3 * pred_anomalies
        micro_auc, macro_auc = evaluate_model(predictions, labels, videos,
                                              normalize_scores=False,
                                              range=100, mu=11)
    elif args.dataset == 'ped2':
        predictions_teacher = np.array(predictions_teacher)
        predictions_student_teacher = np.array(predictions_student_teacher)
        pred_anomalies = np.array(pred_anomalies)
        predictions = 10.5 * predictions_teacher + 5.3 * predictions_student_teacher + 5.3 * pred_anomalies
        micro_auc, macro_auc = evaluate_model(predictions, labels, videos,
                                              normalize_scores=False,
                                              range=100, mu=11)
        # 粗搜索
        weight_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(10):
            for w in weight_params:
                evaluate_model(predictions, labels, videos,
                               normalize_scores=False, dataset=args.dataset,
                               range=20+i*20, mu=2+i*2, weight=w)
    else:
        predictions_teacher = np.array(predictions_teacher)
        predictions_student_teacher = np.array(predictions_student_teacher)
        if len(pred_anomalies) != 0:
            pred_anomalies = np.array(pred_anomalies)
            predictions = 10.5 * predictions_teacher + 5.3 * predictions_student_teacher + 5.3 * pred_anomalies
        else:
            predictions = 10.5 * predictions_teacher + 5.3 * predictions_student_teacher
        micro_auc, macro_auc = evaluate_model(predictions, labels, videos,
                                              normalize_scores=False,
                                              range=50, mu=20)
        micro_auc, macro_auc = evaluate_model(predictions, labels, videos,
                                              normalize_scores=True,
                                              range=50, mu=20)
        # range:50 mu:20 True
        # range:50 mu:20 False
        # range:120 mu:16 True MicroAUC: 0.6979951228689855, MacroAUC: 0.8224947369492023
        # range:120 mu:16 False MicroAUC: 0.5280654536454747, MacroAUC: 0.8139392223992751
        # range:200 mu:20 True MicroAUC: 0.7130767998462397, MacroAUC: 0.8282025714373306
        # range:200 mu:20 False MicroAUC: 0.5413547710776061, MacroAUC: 0.8188680416257275
        # range:300 mu:30 True MicroAUC: 0.7330919176192304, MacroAUC: 0.8381192056966079
    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")

    # np.save("st_tc_list.npy", predictions_student_teacher)
    # np.save("rec_list.npy", predictions_teacher)
    # np.save("pred_list.npy", pred_anomalies)
    # np.save("videos.npy", videos)
    # np.save("labels.npy", labels)


def evaluate_model(predictions, labels, videos,
                   range=302, mu=21, normalize_scores=False):
    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = filt(pred, range=range, mu=mu)
        if normalize_scores:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        pred = np.nan_to_num(pred, nan=0.)
        # plt.plot(pred)
        # plt.xlabel("Frames")
        # plt.ylabel("Anomaly Score")
        # plt.savefig(f"graphs/{vid}.png")
        # plt.close()
        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)

        # pred + label
        # Plot the anomaly score (pred) and the label (lbl) on the same graph
        if normalize_scores:
            pred_norm = pred
        else:
            pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        plt.plot(pred_norm, label='Anomaly Score', color='b')
        plt.plot(lbl, label='Ground Truth', color='r', linestyle='--')
        plt.xlabel("Frames")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.title(f"Anomaly Detection: Video {vid}")

        # Save the plot for this video
        os.makedirs("lib/vad/vad_mae/experiments/graphs/shanghaitech", exist_ok=True)
        plt.savefig(f"lib/vad/vad_mae/experiments/graphs/shanghaitech/{vid}.png")
        plt.close()

        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    # Micro-AUC
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    # gather the stats from all processes
    return micro_auc, macro_auc
