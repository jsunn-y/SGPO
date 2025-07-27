for protein in GB1 #TrpB, CreiLOV
do  

    python pareto_NOS_hyperparameter.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_NOS_continuous algorithm=NOS_continuous

    python pareto_NOS_hyperparameter.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein_NOS_discrete algorithm=NOS_discrete

done

