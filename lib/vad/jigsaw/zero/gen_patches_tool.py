import torch
from torchvision.ops import roi_align


# TO-DO: compare the RoiAlign with Crop+Resize
def crop_object(self, frame_img, bbox, i, size=(64, 64)):
    # frame_img : C * D * H * W
    shape = frame_img.shape
    bbox = torch.from_numpy(bbox[i, :4]).float()
    frame_img = frame_img.reshape(1, -1, shape[2], shape[3])
    frame_img = roi_align(frame_img, [bbox.unsqueeze(dim=0)], output_size=size)
    frame_img = frame_img.reshape(-1, shape[1], size[0], size[1])
    return frame_img
