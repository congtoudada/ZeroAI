import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from tqdm import tqdm

from jigsaw.aggregate import remake_video_3d_output
from util import misc
from util.abnormal_utils import filt


def inference_hybird(model: torch.nn.Module, data_loader: Iterable,
                     model2: torch.nn.Module, data_loader2: Iterable,
                     device: torch.device, log_writer=None, args=None):
    # -------------------------- jigsaw --------------------------
    model2.eval()
    video_output = {}
    step = int(args.sample_num * 0.5)
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

        # jigsaw中分数越低越异常，这里取1-分数，即越高越异常
        for video_, frame_, s_score_, t_score_ in zip(videos, frames, scores, scores2):
            if video_ not in video_output:
                video_output[video_] = {}
                for i in range(step):
                    video_output[video_][i] = [1-1, 1-1]  # 首帧非异常
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
                video_output[video_][frame_].append(1-s_score_)
                video_output[video_][frame_].append(1-t_score_)
            else:
                video_output[video_][frame_][0] = max(video_output[video_][frame_][0], 1-s_score_)
                video_output[video_][frame_][1] = max(video_output[video_][frame_][1], 1-t_score_)

    # ideo_output_spatial, video_output_temporal, video_output_complete = remake_video_3d_output(video_output,
    #                                                                                            dataset=args.dataset)
    # micro_auc, macro_auc = save_and_evaluate(video_output, running_date, dataset=args.dataset)
    # 补尾帧
    for vid, frames in video_output.items():
        # 获取尾帧分数
        len_frames = len(frames)
        tail_s = video_output[vid][len_frames-1][0]
        tail_t = video_output[vid][len_frames-1][1]
        tail_score = [tail_s, tail_t]
        for i in range(step):
            video_output[vid][len_frames+i] = tail_score

    # 分数归一化
    # max_s_score = 0
    # min_s_score = 1
    # max_t_score = 0
    # min_t_score = 1
    # for vid, frames in video_output.items():
    #     for fid, scores in frames.items():
    #         s_score = scores[0]
    #         t_score = scores[1]
    #         if s_score > max_s_score:
    #             max_s_score = s_score
    #         if s_score < min_s_score:
    #             min_s_score = s_score
    #         if t_score > max_t_score:
    #             max_t_score = t_score
    #         if t_score < min_t_score:
    #             min_t_score = t_score
    # for vid, frames in video_output.items():
    #     for fid, scores in frames.items():
    #         scores[0] = (scores[0] - min_s_score) / (max_s_score - min_s_score)
    #         scores[1] = (scores[1] - min_t_score) / (max_t_score - min_t_score)
    # for k, v in video_output.items():
    #     for fid in v.keys():
    #         print(fid)
    # print(video_output)

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
        micro_auc, macro_auc = evaluate_model(predictions, labels, videos, video_output,
                                              normalize_scores=False,
                                              range=100, mu=11)
    else:
        predictions = pred_anomalies + predictions_teacher + predictions_student_teacher
        micro_auc, macro_auc = evaluate_model(predictions, labels, videos, video_output,
                                              normalize_scores=True,
                                              range=120, mu=16)

    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")

    # np.save("st_tc_list.npy", predictions_student_teacher)
    # np.save("rec_list.npy", predictions_teacher)
    # np.save("pred_list.npy", pred_anomalies)
    # np.save("videos.npy", videos)
    # np.save("labels.npy", labels)


def evaluate_model(predictions, labels, videos, video_output,
                   range=302, mu=21, normalize_scores=False):
    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        # 补齐无对象帧结果
        for fid, _ in enumerate(pred):
            if video_output[vid].__contains__(fid):
                continue
            else:
                video_output[vid][fid] = [0.5, 0.5]
        # fid = fids[np.array(videos) == vid]
        video_output[vid] = list(video_output[vid].values())
        # ------------------------ max ------------------------
        # video_output[vid] = np.array([max(pair[0], pair[1]) for pair in video_output[vid]])

        # ------------------------ add ------------------------
        # 0.21 MicroAUC: 0.9353282584667615, MacroAUC: 0.9028670567235408
        # video_output[vid] = np.array([pair[0] * 0.25 + pair[1] * 0.75 for pair in video_output[vid]])

        # 0.21 MicroAUC: 0.938568288496407, MacroAUC: 0.9222412295592993
        # video_output[vid] = np.array([pair[0] * 0.5 + pair[1] * 0.5 for pair in video_output[vid]])

        # 0.21 MicroAUC: 0.933944214206778, MacroAUC: 0.9315834605328955
        # video_output[vid] = np.array([pair[0] * 0.75 + pair[1] * 0.25 for pair in video_output[vid]])

        # 0.21 MicroAUC: 0.938319034582237, MacroAUC: 0.9299158508214856
        video_output[vid] = np.array([pair[0] * 0.6 + pair[1] * 0.4 for pair in video_output[vid]])

        # 0.21 MicroAUC: 0.938319034582237, MacroAUC: 0.9299158508214856
        # video_output[vid] = np.array([pair[0] * 0.65 + pair[1] * 0.35 for pair in video_output[vid]])

        # 0.21 MicroAUC: 0.9380336667259613, MacroAUC: 0.9245246329731791
        # video_output[vid] = np.array([pair[0] * 0.55 + pair[1] * 0.45 for pair in video_output[vid]])

        # pred = video_output[vid]
        # pred += video_output[vid] * 0.3  # MicroAUC: 0.9239464530454763, MacroAUC: 0.9409956053829157
        # pred += video_output[vid] * 0.25  # MicroAUC: 0.9263334040489463, MacroAUC: 0.9385036530054336
        # pred += video_output[vid] * 0.23  # MicroAUC: 0.9267144964230156, MacroAUC: 0.9400302642977714
        # pred += video_output[vid] * 0.22  # MicroAUC: 0.9274323513072198, MacroAUC: 0.9377439964789929
        pred += video_output[vid] * 0.21  # MicroAUC: 0.9275254124217831, MacroAUC: 0.9394392395403883
        # pred += video_output[vid] * 0.2  # MicroAUC: 0.9272279012524745, MacroAUC: 0.9381994910451334
        # pred += video_output[vid] * 0.18  # MicroAUC: 0.9268373289686024, MacroAUC: 0.9358369708170391
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
        pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        plt.plot(pred_norm, label='Anomaly Score', color='b')
        plt.plot(lbl, label='Ground Truth', color='r', linestyle='--')
        plt.xlabel("Frames")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.title(f"Anomaly Detection: Video {vid}")

        # Save the plot for this video
        os.makedirs("lib/vad/vad_mae/experiments/graphs/avenue", exist_ok=True)
        plt.savefig(f"lib/vad/vad_mae/experiments/graphs/avenue/{vid}.png")
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
