nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/cta/users/undergrad2/SSL-FL /cta/users/undergrad2/miniconda3/envs/ssfl/bin/python -u /cta/users/undergrad2/SSL-FL/code/fed_mae/run_class_finetune_FedAvg.py   --data_set Retina   --data_path /cta/users/undergrad2/SSL-FL/data/Retina   --split_type split_1   --n_clients 5   --num_local_clients -1   --E_epoch 1   --max_communication_rounds 300   --batch_size 256   --blr 3e-3   --layer_decay 0.75   --weight_decay 0.05   --warmup_epochs 5   --model vit_base_patch16   --nb_classes 2   --global_pool   --resume /cta/users/undergrad2/SSL-FL/out_finetune_retina2/checkpoint-199.pth   --output_dir /cta/users/undergrad2/SSL-FL/out_finetune_retina2_resume   --log_dir /cta/users/undergrad2/SSL-FL/out_finetune_retina2_resume/tb > /cta/users/undergrad2/SSL-FL/finetune_retina2_resume2.log 2>&1 &

or you can aslo use this 
history | grep run_class_finetune_FedAvg.py



nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/cta/users/undergrad2/SSL-FL /cta/users/undergrad2/miniconda3/envs/ssfl/bin/python -u /cta/users/undergrad2/SSL-FL/code/fed_mae/run_class_finetune_FedAvg.py   --data_set Retina   --data_path /cta/users/undergrad2/SSL-FL/data/Retina   --split_type split_1   --n_clients 5   --num_local_clients -1   --E_epoch 1   --max_communication_rounds 300   --batch_size 256   --blr 3e-3   --layer_decay 0.75   --weight_decay 0.05   --warmup_epochs 5   --model vit_base_patch16   --nb_classes 2   --global_pool   --finetune /cta/users/undergrad2/SSL-FL/out_pretrain_retina_35K/checkpoint-119.pth   --output_dir /cta/users/undergrad2/SSL-FL/out_finetune_retina_35K   --log_dir /cta/users/undergrad2/SSL-FL/out_finetune_retina_35K/tb > /cta/users/undergrad2/SSL-FL/finetune_retina35K.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/cta/users/undergrad2/SSL-FL /cta/users/undergrad2/miniconda3/envs/ssfl/bin/python -u /cta/users/undergrad2/SSL-FL/code/fed_mae/run_class_finetune_FedAvg.py   --data_set Retina   --data_path /cta/users/undergrad2/SSL-FL/data/Retina   --split_type split_1   --n_clients 5   --num_local_clients -1   --E_epoch 1   --max_communication_rounds 300   --batch_size 256   --blr 3e-3   --layer_decay 0.75   --weight_decay 0.05   --warmup_epochs 5   --model vit_large_patch16   --nb_classes 2   --global_pool   --finetune /cta/users/undergrad2/SSL-FL/pretrain_retinalarge_35K/checkpoint-799.pth   --output_dir /cta/users/undergrad2/SSL-FL/out_finetune_retinalarge_35K   --log_dir /cta/users/undergrad2/SSL-FL/out_finetune_retinalarge_35K/tb > /cta/users/undergrad2/SSL-FL/finetune_retinalarge35K.log 2>&1 &


env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/cta/users/undergrad2/SSL-FL /cta/users/undergrad2/miniconda3/envs/ssfl/bin/python -u /cta/users/undergrad2/SSL-FL/code/fed_mae/run_class_finetune_FedAvg.py --data_set Retina --data_path /cta/users/undergrad2/SSL-FL/data/Retina --split_type split_1 --n_clients 5 --num_local_clients -1 --E_epoch 1 --max_communication_rounds 300 --batch_size 64 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 5 --model vit_large_patch16 --nb_classes 2 --global_pool --finetune /cta/users/undergrad2/SSL-FL/pretrain_retinalarge_35K/checkpoint-799.pth --output_dir /cta/users/undergrad2/SSL-FL/out_finetune_retinalarge_35K --log_dir /cta/users/undergrad2/SSL-FL/out_finetune_retinalarge_35K/tb



python code/fed_mae/run_class_finetune_FedAvg.py   --data_set Retina   --data_path /cta/users/undergrad2/SSL-FL/data/Retina   --split_type split_1   --n_clients 5   --num_local_clients -1   --E_epoch 1   --max_communication_rounds 300   --batch_size 256   --blr 3e-3   --layer_decay 0.75   --weight_decay 0.05   --warmup_epochs 5   --model vit_base_patch16   --nb_classes 2   --global_pool   --finetune /cta/users/undergrad2/SSL-FL/out_pretrain_retina_3/checkpoint-119.pth   --output_dir /cta/users/undergrad2/SSL-FL/out_finetune_retina_35K   --log_dir /cta/users/undergrad2/SSL-FL/out_finetune_retina_35K/tb



#################################
#######if we want to train on  central 


nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/cta/users/undergrad2/SSL-FL \
/cta/users/undergrad2/miniconda3/envs/ssfl/bin/python -u \
/cta/users/undergrad2/SSL-FL/code/fed_mae/run_class_finetune_central.py \
--model vit_base_patch16 \
--finetune /cta/users/undergrad2/SSL-FL/out_pretrain_retina2_35K/checkpoint-1599.pth \
--data_set Retina \
--data_path /cta/users/undergrad2/SSL-FL/data/Retina \
--batch_size 64 \
--blr 5e-4 \
--weight_decay 0.1 \
--drop_path 0.3 \
--mixup 0.8 \
--cutmix 1.0 \
--warmup_epochs 5 \
--epochs 100 \
--nb_classes 2 \
--global_pool \
--output_dir /cta/users/undergrad2/SSL-FL/out_finetune_retinabase_35K_central \
--log_dir /cta/users/undergrad2/SSL-FL/out_finetune_retinabase_35K_central/tb \
> /cta/users/undergrad2/SSL-FL/finetune_retinabase_35K_central.log 2>&1 &


##################################










        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)
        
        if self.args.data_set == 'Retina':
            #img = np.load(path)
            try:
                img = np.load(path, allow_pickle=False)
            except ValueError:
                img = np.load(path, allow_pickle=True)

            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))
        
        
        if img.ndim < 3:
            img = np.concatenate((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target
        
        
the error i never noticed was 151 about mismatch when loading the finetuning code. because i tracked the checkpoint and found its keys using a code like 

python - <<'PY'
import torch, sys
ckpt_path = "/cta/users/undergrad2/SSL-FL/out_pretrain_retina2_35K/checkpoint-1599.pth"
ck = torch.load(ckpt_path, map_location='cpu')
state = ck.get('model', ck)
print("Checkpoint top keys:", list(ck.keys())[:20])
print("State dict sample keys:", list(state.keys())[:40])

# Count encoder.* prefix and sample shapes
enc_keys = [k for k in state.keys() if k.startswith("encoder.")]
print("encoder.* keys:", len(enc_keys))
for k in ['encoder.pos_embed','pos_embed','encoder.patch_embed.proj.weight','patch_embed.proj.weight']:
    if k in state: print(k, state[k].shape)
PY

so we noticed there is no key like encoder. Your checkpoint has no "encoder.*" prefix.

All weights (e.g. pos_embed, patch_embed.proj.weight, blocks.0.*) are already in the "plain" ViT style.

That’s why your old filtering logic:enc_state = {k.replace("encoder.",""): v for k,v in state.items() if k.startswith("encoder.")}
ended up keeping nothing (because k.startswith("encoder.") was always false).
? Result: 151 missing params when loading.
so we are gonna fix it using another code, which we name the previous code which was working but with the rror 151 as _1 then modify the finetuning code
        