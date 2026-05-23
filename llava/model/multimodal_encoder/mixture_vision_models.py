import os
import torch
from copy import deepcopy
from .vision_models.eva_vit import EVAVITVisionTower
from .vision_models.sam_encoder import SAMVisionTower
from .vision_models.pix2struct_encoder import Pix2StructLargeVisionTower
from .vision_models.convnext_encoder import ConvNextVisionTower
from .vision_models.biomed_clip_encoder import BiomedCLIPVisionTower
from .builder import build_vision_tower

def init_vision_teachers(config):
    vision_teachers = {}

    # CLIP
    if 'clip' in getattr(config, 'encoder_teachers', []):
        clip_args = deepcopy(config) 
        clip_teacher = build_vision_tower(clip_args)
        vision_teachers['CLIP'] = clip_teacher

    # BiomedCLIP
    if 'biomed_clip' in getattr(config, 'encoder_teachers', []):
        biomed_args = deepcopy(config)
        biomed_args.freeze_vision = getattr(config, 'freeze_vision_teacher', True)
        biomed_args.mm_vision_select_layer = getattr(config, 'biomed_clip_select_layer', -2)
        biomed_args.mm_vision_select_feature = getattr(config, 'biomed_clip_select_feature', 'patch')
        biomed_vision_tower = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        biomed_teacher = BiomedCLIPVisionTower(biomed_vision_tower, biomed_args)
        vision_teachers['BiomedCLIP'] = biomed_teacher

    # SAM
    if 'sam' in getattr(config, 'encoder_teachers', []):
        sam_args = deepcopy(config)
        sam_args.freeze_vision = False
        sam_args.input_image_size = 1024
        sam_args.add_pixel_shuffle = True
        sam_args.vision_tower_pretrained_from = './checkpoints/sam/sam_vit_l'
        sam_teacher = SAMVisionTower("SAM-L", sam_args)
        sam_teacher.load_model()
        vision_teachers['SAM'] = sam_teacher

    # EVA-02
    if 'eva' in getattr(config, 'encoder_teachers', []):
        eva_args = deepcopy(config)
        eva_args.input_image_size = 1024
        eva_args.freeze_vision = False
        eva_args.vision_tower_pretrained_from = './checkpoints/EVA/eva02/det/eva02_L_coco_det_sys_o365.pth'
        eva_teacher = EVAVITVisionTower("eva02-l-16", eva_args)     
        eva_teacher.load_model()
        vision_teachers['EVA'] = eva_teacher

    # Pix2struct
    if 'pix2struct' in getattr(config, 'encoder_teachers', []):
        pix_args = deepcopy(config)
        #pix_args.freeze_vision = True
        pix_args.input_image_size = 1024
        pix_args.freeze_vision = False
        pix_args.do_resize = True
        pix_args.de_normalize = True
        pix_args.vision_tower_pretrained_from = './checkpoints/pix2struct'
        pix_teacher = Pix2StructLargeVisionTower("pix2struct-large", pix_args)     
        pix_teacher.load_model()
        vision_teachers['Pix2Struct'] = pix_teacher
    
    # ConvNeXt
    if 'convnext' in getattr(config, 'encoder_teachers', []):
        convnext_args = deepcopy(config)
        convnext_args.freeze_vision = False
        convnext_args.input_image_size = 1024
        convnext_vision_tower = "convnext_xxlarge.clip_laion2b_soup" # hardcode
        convnext_args.vision_tower_pretrained_from = './checkpoints/convnext'
        convnext_teacher = ConvNextVisionTower(convnext_vision_tower, 
                                                        convnext_args)
        convnext_teacher.load_model()      
        vision_teachers['ConvNeXt'] = convnext_teacher
    
    return vision_teachers
