for protein in GB1 #CreiLOV #TrpB
do  

    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python pareto_NOS_hyperparameter.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_NOS_continuous algorithm=NOS_continuous

    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python pareto_NOS_hyperparameter.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein_NOS_discrete algorithm=NOS_discrete

done

