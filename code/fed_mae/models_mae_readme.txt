bcz the models_mae.py wasnt suitable for classification part, we modified it. so we preserved the backup here and wrote the new one.
it has a flag and to do some jobs for classification or training. 

Purpose of Modifying models_mae.py for Classification

The original models_mae.py file defines the Masked Autoencoder (MAE) architecture used for unsupervised pretraining, where the model learns to reconstruct missing image patches. However, this model lacks a classification head and cannot be directly used for downstream tasks like classification.

To fine-tune the pretrained MAE on a classification task (e.g., distinguishing between two classes of medical images), we added the following functionality:
Modifications

Add a use_class_head Flag

Allows toggling between pretraining and classification modes using the same model.

Keeps the architecture reusable and backward-compatible.

Add num_classes Parameter

Specifies the number of output classes when using the model for classification.

Add a Classification Head (self.cls_head)

A simple nn.Linear(embed_dim, num_classes) layer.

Takes the [CLS] token output from the encoder and maps it to class logits.

Modify the forward() Function

When use_class_head=True, skips the decoder and returns classification logits.

When use_class_head=False, follows the standard MAE pipeline (encode ? decode ? reconstruct ? compute loss).


CUDA_VISIBLE_DEVICES=1 nohup python code/fed_mae/run_mae_pretrain_FedAvg.py   --model mae_vit_base_patch16   --data_set Retina   --data_path ./data/Retina   --split_type split_1   --mask_ratio 0.6   --batch_size 256   --accum_iter 1   --blr 1e-3   --warmup_epochs 5   --E_epoch 1   --max_communication_rounds 2400   --num_local_clients -1   --output_dir ./out_pretrain_retina2_35K   --log_dir ./logs_pretrain_retina2_35K   --resume ./out_pretrain_retina2_35K/checkpoint-1599.pth   > pretrain_resume_retina_35K.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 nohup python code/fed_mae/run_mae_pretrain_FedAvg.py   --model mae_vit_base_patch16   --data_set Retina   --data_path ./data/Retina   --split_type split_1   --mask_ratio 0.6   --batch_size 128   --accum_iter 1   --blr 1e-3   --warmup_epochs 5   --E_epoch 1   --max_communication_rounds 2400   --num_local_clients -1   --output_dir ./out_pretrain_retina_12K   --log_dir ./logs_pretrain_retina_12K  



nohup python code/fed_mae/run_mae_pretrain_FedAvg.py \
  --model mae_vit_large_patch16 \
  --data_set Retina \
  --data_path ./data/Retina \
  --split_type split_1 \
  --mask_ratio 0.6 \
  --batch_size 256 \
  --accum_iter 1 \
  --blr 1e-3 \
  --warmup_epochs 5 \
  --E_epoch 1 \
  --max_communication_rounds 1600 \
  --num_local_clients -1 \
  --output_dir ./out_pretrain_retina2_35K \
  --log_dir ./logs_pretrain_retina2_35K \
  > pretrain_retina_35K.log 2>&1 & 
