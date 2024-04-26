from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from collections import defaultdict

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )
    #indices是每行元素从小到大排序的索引数组
    indices = np.argsort(distmat, axis=1) #距离由小到大排序;axis=1表示按行排序;
    #print(indices.shape) #[3368, 15913]
    #print(indices)
    #最终matches矩阵展示了查询样本与图库样本身份ID之间的匹配情况
    #print(g_pids[indices].shape, q_pids[:, np.newaxis].shape) #(3368, 15913) (3368, 1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        #具体细节
        #g_pids[indices]: 这部分会根据indices索引从g_pids中取出对应的元素，生成一个新的与indices形状相同的数组。[3368, 15913]
        #q_pids[:, np.newaxis]: 这部分会将一维数组q_pids转换为二维数组[3368,1]，使其可以与g_pids[indices]进行比较.
        #(g_pids[indices] == q_pids[:, np.newaxis]): 这部分将g_pids[indices]与q_pids[:, np.newaxis]进行逐元素比较，
            #比较时遵循广播原则,[3368,1]广播为[3368, 15913]  
            #若相等则返回True，否则返回False。这样就生成了一个布尔类型的匹配矩阵。
        #.astype(np.int32): 这部分将布尔类型的匹配矩阵转换为整数类型，True转为1，False转为0。这样就便于后续的计算和处理。

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    #for循环，遍历查询样本的索引，范围是0到num_q-1。
    for q_idx in range(num_q):
        # get query pid and camid
        #获取当前查询样本的身份ID和摄像头ID。
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        #从排序后的索引数组中获取当前查询样本对应的图库样本索引(取出indices的第q_idx行)
            #例如，当q_idx==0时,如果distmat的第0行是[0.5, 0.2, 0.8, 0.3, 0.1]，
            #那么order就是[4, 1, 3, 0, 2]，表示的是distmat第0行的值从小到大排序后的索引。
        order = indices[q_idx]
        #删除对于当前查询样本在相同摄像头下的相同身份的图库样本
            #创建一个布尔数组remove，用来标记与当前查询样本具有相同身份和摄像头ID的图库样本。
            #使用np.invert函数得到保留的图库样本的布尔数组keep，其中True表示保留，False表示删除。
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        
        # #pzy1
        # # Find the first match to remove  
        # first_match_index = np.where(remove)[0][0] if np.any(remove) else None
        # # Create a new keep array
        # keep = np.ones_like(remove, dtype=bool)
        # # Set the first match to remove
        # if first_match_index is not None:
        #     keep[first_match_index] = False
        
        keep = np.invert(remove)
        keep = np.ones_like(keep, dtype=bool) #pzy2
        # compute cmc curve
        #从 matches 矩阵中提取出与当前查询样本 q_idx 相关的匹配信息，
            #并通过 keep 布尔数组过滤掉那些需要被移除的样本。
            #结果 raw_cmc 是一个二进制向量，其中值为1的位置表示正确的匹配。
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        #如果没有任何正确匹配的情况，跳过当前查询样本的处理。
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        #计算raw_cmc的累积和，得到CMC曲线。
            #在这个累积和中，第i个元素表示的是在前i个样本中有多少个样本被正确匹配。
        #print(raw_cmc.shape) #比如(15911,)
        #print(raw_cmc)
        cmc = raw_cmc.cumsum()
        #print(cmc.shape)   #比如(15911,)
        #print(cmc)
        #将CMC曲线中大于1的值截断为1，确保曲线的性质。
            #这时候cmc第一个是1的位置是i,则表示rank(i+1)命中.
            #准确率很高 所以一般i都为0
        cmc[cmc > 1] = 1
        #print(cmc.shape)   #比如(15911,)
        #print(cmc)
        
        #将截断后的CMC曲线结果添加到all_cmc列表中，仅保留前max_rank个值。 
        #错误理解❌:rank20的意思是找到20个正确匹配的时候,正例占比正例+负例的比例,
            #比如找到第20个正例的时候一共找到了20个正例3个负例,则rank20是20/20+3=86.9%
        #正确理解✅:对于该query,rank20的意思是只要距离最小的前20个样本中有正例,rank20就是100%
            #当100个query中有98个rank20为100%,有两个query前20个样本中没有找到正例时,
            #该测试集rank20的值为98%
        all_cmc.append(cmc[:max_rank])
        #print(len(all_cmc),len(all_cmc[0])) #1、2……3368 ;50
        #增加有效查询样本的计数。
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        #
        num_rel = raw_cmc.sum()
        #print(num_rel)
        #计算raw_cmc的累积和。这样，对于每个位置i，tmp_cmc[i]就是查询样本在前i+1个位置中匹配上的次数。
        tmp_cmc = raw_cmc.cumsum()
        #print(tmp_cmc)
        #计算每个位置的准确率，得到一个临时的CMC曲线。这里的准确率是指在前i+1个位置中找到正确匹配的比例。
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        #print(tmp_cmc)
        #将每个位置的准确率乘以对应位置的匹配情况，得到修正后的CMC曲线。
            #在这里，tmp_cmc数组表示的是在每个位置的准确率，
            #而raw_cmc数组表示的是在每个位置是否有正确的匹配（如果有，值为1，否则为0）
            #这样的结果是，新的tmp_cmc数组中，只有那些有正确匹配的位置的准确率被保留，其它位置的值都为0。
            #AP的定义就是在每个正确匹配的位置i处的准确率的平均值
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        #print(tmp_cmc)
        #计算平均准确率（AP）
        AP = tmp_cmc.sum() / num_rel
        #将计算得到的平均准确率添加到all_AP列表中。
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    
    #类型转换,避免整数除法
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    #先行平均
    #print(all_cmc.sum(0).shape) #(50,)
    all_cmc = all_cmc.sum(0) / num_valid_q
    #print(all_cmc.shape) #(50,)
    #print(all_AP)
    mAP = np.mean(all_AP)
    #print(all_cmc,mAP)
    return all_cmc, mAP


def evaluate_py(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )
    else:
        return eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )


def evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    use_metric_cuhk03=False,
    use_cython=True
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
    else:
        return evaluate_py(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
