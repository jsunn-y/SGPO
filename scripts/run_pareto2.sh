for protein in GB1
do  

    #CUDA_VISIBLE_DEVICES=1 python pareto.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_classifier_discrete model=mdlm algorithm=cls_guidance_discrete

    CUDA_VISIBLE_DEVICES=1 python pareto.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein_classifier_discrete algorithm=daps_discrete

    CUDA_VISIBLE_DEVICES=1 python pareto.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein_NOS_discrete algorithm=NOS_discrete

    CUDA_VISIBLE_DEVICES=1 python pareto.py pretrained_ckpt=causalLM_finetune/$protein data=$protein model=causalLM problem=protein_DPO algorithm=DPO

    CUDA_VISIBLE_DEVICES=1 python pareto.py pretrained_ckpt=causalLM_finetune/TrpB data=TrpB model=causalLM problem=protein_DPO_weighted algorithm=DPO
    
done