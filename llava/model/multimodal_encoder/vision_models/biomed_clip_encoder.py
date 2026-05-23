import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig

# BiomedCLIP uses ViT-B/16 at 224x224 with standard CLIP normalization.
BIOMED_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOMED_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
BIOMED_IMAGE_SIZE = 224
BIOMED_PATCH_SIZE = 16
BIOMED_HIDDEN_DIM = 768  # ViT-B/16 pre-projection width


class BiomedCLIPVisionTower(nn.Module):
    """
    BiomedCLIP teacher encoder (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

    Requires: pip install open-clip-torch
    Checkpoint key:  'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

    Returns patch tokens of shape [B, 196, 768], without the CLS token.
    Use the adapters in move_llava_llama.py to align to student dims (1024, 576).
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args = args
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_vision = getattr(args, 'freeze_vision', True)

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "BiomedCLIP teacher requires open-clip-torch. "
                "Install it before using --encoder_teachers biomed_clip."
            ) from exc

        # CLIPImageProcessor at 224 carries BiomedCLIP's mean/std for process_images().
        self.image_processor = CLIPImageProcessor(
            do_resize=True,
            size={'shortest_edge': BIOMED_IMAGE_SIZE},
            crop_size={'height': BIOMED_IMAGE_SIZE, 'width': BIOMED_IMAGE_SIZE},
            do_center_crop=True,
            do_normalize=True,
            image_mean=BIOMED_IMAGE_MEAN,
            image_std=BIOMED_IMAGE_STD,
        )

        model, _, _ = open_clip.create_model_and_transforms(self.vision_tower_name)
        self.vision_tower = model.visual
        # output_tokens=True makes forward return (pooled_cls, patch_tokens).
        self.vision_tower.output_tokens = True

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def feature_select(self, patch_tokens, cls_token):
        """
        Select features based on mm_vision_select_feature.
        BiomedCLIP ViT-B/16 has 196 patch tokens plus 1 CLS token.
        """
        if self.select_feature == 'patch':
            return patch_tokens
        elif self.select_feature == 'cls_patch':
            # cls_token from timm is already [B, 1, D]; cat directly
            return torch.cat((cls_token, patch_tokens), dim=1)
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

    def forward(self, images):
        if images is None:
            return None

        # Ensure images have shape [B, C, H, W].
        if type(images) is list:
            images = torch.stack(images)
        
        images = images.to(device=self.device, dtype=self.dtype)

        # In timm/open_clip, n=1 is the final layer and n=2 is the penultimate layer.
        # Convert LLaVA's usually negative select_layer into timm's n-from-end value.
        n_from_block = abs(self.select_layer) if self.select_layer < 0 else (12 - self.select_layer)

        # Access the timm trunk directly so return_prefix_tokens is forwarded reliably.
        intermediate_layers = self.vision_tower.trunk.get_intermediate_layers(
            images,
            n=n_from_block,
            return_prefix_tokens=True
        )

        patch_tokens, cls_token = intermediate_layers[0]
        
        image_features = self.feature_select(patch_tokens, cls_token)

        return image_features, None

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.num_patches, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        """Mock config for callers that expect a Hugging Face-style .config."""
        return CLIPVisionConfig(
            hidden_size=BIOMED_HIDDEN_DIM,
            image_size=BIOMED_IMAGE_SIZE,
            patch_size=BIOMED_PATCH_SIZE,
            num_hidden_layers=12
        )

    @property
    def hidden_size(self):
        return BIOMED_HIDDEN_DIM

    @property
    def num_patches_per_side(self):
        return BIOMED_IMAGE_SIZE // BIOMED_PATCH_SIZE

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
