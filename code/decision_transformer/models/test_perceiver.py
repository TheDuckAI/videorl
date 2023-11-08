import open_clip
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn



from perceiver import PerceiverResampler


vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")

vision_encoder = vision_encoder.visual
vision_encoder.output_tokens = True

vis_dim=open_clip.get_model_config("ViT-L-14")["vision_cfg"]["width"]
perceiver = PerceiverResampler(dim=vis_dim)

            
            
batch = 5
num_images = 27
channels = 3
height = 224
width = 224
input_data = torch.randn((batch, num_images, 1, channels, height, width))
# vision_x (torch.Tensor): Vision input
#     shape (B, T_img, F, C, H, W) with F=1


if False:
    import torch
    from PIL import Image
    import open_clip
    
    image = image_processor(Image.open("dog.jpg")).unsqueeze(0)
    print(image.shape)
    
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = vision_encoder.encode_image(image)
        print(image_features.shape)


def encode_vision_x(vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        print(vision_x.shape)
        with torch.no_grad():
            vision_x, tokens = vision_encoder(vision_x)
            #We might want the -2 instead by the way.
            print(tokens.shape)#batch x frames x 768..
        
        vision_x = rearrange(tokens, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        print(vision_x.shape)#Put back in original shape
        vision_x = perceiver(vision_x)
        print(vision_x.shape)

        # for layer in lang_encoder._get_decoder_layers():
        #     layer.condition_vis_x(vision_x)

encode_vision_x(input_data)
