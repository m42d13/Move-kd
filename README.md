## MoVE-KD modified for medical VQA

Pretrain data: LlaVA-Pretrain + MedMNIST
    playground/data/LLaVA-Pretrain/llava_medmnist_tiny.json (small-scaled version -currently used)

Finetune data: RadImageNet
    radimagenet_llava_instruct_balanced_10k_shuffled.json (multi-turn)
    radimagenet_llava_instruct_balanced_10k_singleturn_formatfix.json (single-turn)
