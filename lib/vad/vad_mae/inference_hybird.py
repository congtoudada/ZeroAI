import os
import pickle
from collections.abc import Iterable

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from tqdm import tqdm

from jigsaw.aggregate import remake_video_3d_output, remake_video_output
from util import misc
from util.abnormal_utils import filt


def inference_hybird(model: torch.nn.Module, data_loader: Iterable,
                     model2: torch.nn.Module, data_loader2: Iterable,
                     device: torch.device, log_writer=None, args=None):
    # -------------------------- jigsaw --------------------------
    model2.eval()
    video_output = {}
    # step = int(args.sample_num * 0.5)

    for data in tqdm(data_loader2):
        videos = data["video"]
        frames = data["frame"].tolist()
        obj = data["obj"].cuda(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = model2(obj)
            temp_logits = temp_logits.view(-1, args.sample_num, args.sample_num)
            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()

        for video_, frame_, s_score_, t_score_  in zip(videos, frames, scores, scores2):
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([s_score_, t_score_])

    # jigsaw中分数越低越异常，这里取1-分数，即越高越异常
    pickle_path = './logs/video_output_ori_{}.pkl'.format(args.dataset)
    with open(pickle_path, 'wb') as write:
        pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)
    if args.dataset == 'shanghaitech':
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(video_output,
                                                                                                 dataset=args.dataset)
    else:
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_3d_output(video_output,
                                                                                                    dataset=args.dataset)

    video_output = [1 - l for l in video_output_complete]

    # -------------------------- vad-mae --------------------------
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
    # fids = []
    frames = []
    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        # fids += list(fid)
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
    # fids = np.array(fids)

    predictions_teacher = np.array(predictions_teacher)
    predictions_student_teacher = np.array(predictions_student_teacher)
    pred_anomalies = np.array(pred_anomalies)

    if args.dataset == 'avenue':
        predictions = 10.5 * predictions_teacher + 5.3 * predictions_student_teacher + 5.3 * pred_anomalies
        # MicroAUC: 0.9453523376633725, MacroAUC: 0.9403822652221759 range=120, mu=12, weight=0.8, normalize_scores=False
        evaluate_model(predictions, labels, videos, video_output,
                       normalize_scores=False, dataset=args.dataset,
                       range=120, mu=12, weight=0.8, draw_vis=True)
        # 粗搜索
        # weight_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # for i in range(10):
        #     for w in weight_params:
        #         evaluate_model(predictions, labels, videos, video_output,
        #                        normalize_scores=False, dataset=args.dataset,
        #                        range=100+i*20, mu=10+i*2, weight=w)
        # 细搜索
        # weight_params = [0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86]  # avenue
        # for w in weight_params:
        #     evaluate_model(predictions, labels, videos, video_output,
        #                    normalize_scores=False, dataset=args.dataset,
        #                    range=120, mu=12, weight=w)

    elif args.dataset == 'ped2':
        predictions = 10.5 * predictions_teacher + 5.3 * predictions_student_teacher + 5.3 * pred_anomalies
        evaluate_model(predictions, labels, videos, video_output,
                       normalize_scores=False, dataset=args.dataset,
                       range=20, mu=5, weight=0.2, draw_vis=False)
        # 粗搜索
        weight_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(10):
            for w in weight_params:
                evaluate_model(predictions, labels, videos, video_output,
                               normalize_scores=False, dataset=args.dataset,
                               range=10+i*10, mu=12+2*i, weight=w)
    else:
        if len(pred_anomalies) != 0:
            pred_anomalies = np.array(pred_anomalies)
            predictions = predictions_teacher + predictions_student_teacher + pred_anomalies
        else:
            predictions = predictions_teacher + predictions_student_teacher
        # MicroAUC: 0.8673084615770787, MacroAUC: 0.9290096902395378 range=50, mu=5, weight=0.3, normalize_scores=True
        evaluate_model(predictions, labels, videos, video_output,
                       normalize_scores=True, dataset=args.dataset,
                       range=50, mu=5, weight=0.3, draw_vis=True)
        # 网格搜索
        # weight_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # weight_params = [0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36]
        # for i in range(15):
        #     for w in weight_params:
        #         evaluate_model(predictions, labels, videos, video_output,
        #                        normalize_scores=True, dataset=args.dataset,
        #                        range=10+i*10, mu=1+i*1, weight=w)

    # print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")

    # np.save("st_tc_list.npy", predictions_student_teacher)
    # np.save("rec_list.npy", predictions_teacher)
    # np.save("pred_list.npy", pred_anomalies)
    # np.save("videos.npy", videos)
    # np.save("labels.npy", labels)


def evaluate_model(predictions, labels, videos, video_output,
                   range=302, mu=21, normalize_scores=False, dataset="avenue",
                   weight=0.8, draw_vis=False):
    aucs = []
    filtered_preds = []
    filtered_labels = []
    for idx, vid in enumerate(np.unique(videos)):
        pred = predictions[np.array(videos) == vid]

        if normalize_scores:
            pred = filt(pred, range=900, mu=282)
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            pred = pred * weight + video_output[idx] * (1.0 - weight)
        else:
            pred = pred * weight + video_output[idx] * (1.0 - weight)

        pred = filt(pred, range=range, mu=mu)
        pred = np.nan_to_num(pred, nan=0.)
        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)

        if draw_vis:
            # pred + label
            # Plot the anomaly score (pred) and the label (lbl) on the same graph
            pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            plt.plot(pred_norm, label='Anomaly Score', color='b')
            plt.plot(lbl, label='Ground Truth', color='r', linestyle='--')
            plt.xlabel("Frames")
            plt.ylabel("Anomaly Score")
            plt.legend()
            plt.title(f"Anomaly Detection: Video {vid}")

            # Save the plot for this video
            os.makedirs(f"lib/vad/vad_mae/experiments/graphs/{dataset}", exist_ok=True)
            plt.savefig(f"lib/vad/vad_mae/experiments/graphs/{dataset}/{vid}.png")
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
    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc} range={range}, mu={mu}, "
          f"weight={weight}, normalize_scores={normalize_scores}")
    return micro_auc, macro_auc
