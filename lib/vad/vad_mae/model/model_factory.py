from functools import partial

from torch import nn

from vad_mae.model.mae_cvt import MaskedAutoencoderCvT


def mae_cvt_patch16(mask_ratio=0.5, pred_cls=True, **kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=16, embed_dim=256, depth=3, num_heads=4, mask_ratio=mask_ratio,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4, pred_cls=pred_cls,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_cvt_patch8(mask_ratio=0.5, pred_cls=True, **kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=8, embed_dim=256, depth=3, num_heads=4, mask_ratio=mask_ratio,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4, pred_cls=pred_cls,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model