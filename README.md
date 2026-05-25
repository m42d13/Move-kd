# MoVE-KD modified for medical VQA

### Pretrain:  
    bash Move-kd/scripts/move-kd/pretrain_med.sh` 

Modify `--data_path` for your custom json file. 

Modify `--image_folder` for you custom dataset.

Modify `--output_dir to` store the pretrained weights.

### Finetune:
    bash Move-kd/scripts/move-kd/finetune_med.sh 

Modify `--data_path` for your custom json file. 

Modify `--image_folder` for you custom dataset.

Modify `--pretrain_mm_mlp_adapter`, `--pretrain_mole_encoder`, `--pretrain_encoder_adapter` to point to the respective `.bin` files in the pretrain `--output_dir` folder.

Modify `--output_dir` to store the finetuned weights.

### Pretrain data: LlaVA-Pretrain + MedMNIST
`playground/data/LLaVA-Pretrain/llava_medmnist_tiny.json` **(small-scaled version -currently used)**.

### Finetune data: RadImageNet
`radimagenet_llava_instruct_balanced_10k_shuffled.json` **(multi-turn)**.

`radimagenet_llava_instruct_balanced_10k_singleturn_formatfix.json` **(single-turn)**.
