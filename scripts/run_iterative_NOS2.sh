for protein in GB1 #TrpB CreiLOV #GB1
do  
    HYDRA_FULL_ERROR=1 python iterativeBO.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_NOS_continuous algorithm=NOS_continuous

    HYDRA_FULL_ERROR=1 python iterativeBO.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_NOS_discrete model=mdlm algorithm=NOS_discrete
done