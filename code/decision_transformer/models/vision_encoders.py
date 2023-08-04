import clip
import open_clip
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from perceiver import PerceiverResampler
from torch import einsum, nn


class VisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def forward(self, observations):
        image_tensor = observations.to(self.device)
        image_features = self.model.encode_image(image_tensor)
        return image_features
    


class PerceiverVisionEncoder(nn.Module):
    """
    Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
    Args:
        vision_x (torch.Tensor): Vision input
            shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

    rearrange code based on https://github.com/dhansmair/flamingo-mini
    """
    
    def __init__(self):
        super().__init__()
        self.vision_encoder, _, self.image_processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai")
        self.vision_encoder = self.vision_encoder.visual
        self.vis_dim = open_clip.get_model_config("ViT-L-14")["vision_cfg"]["width"]
        self.perceiver = PerceiverResampler(dim=self.vis_dim)

    def forward(self, vision_x):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)

        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        return vision_x
