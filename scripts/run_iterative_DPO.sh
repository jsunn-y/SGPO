for protein in GB1 #TrpB CreiLOV
do  
    CUDA_VISIBLE_DEVICES=1 python iterativeBO.py pretrained_ckpt=causalLM_finetune/$protein data=$protein model=causalLM problem=protein_DPO algorithm=DPO 
done