for protein in GB1 #TrpB CreiLOV
do  

    python pareto.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_classifier_continuous algorithm=cls_guidance_continuous

    python pareto.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_NOS_continuous algorithm=NOS_continuous

    python pareto.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_classifier_continuous algorithm=daps_continuous

    python pareto.py pretrained_ckpt=d3pm_finetune/$protein data=$protein problem=protein_classifier_discrete model=d3pm algorithm=cls_guidance_discrete

    python pareto.py pretrained_ckpt=d3pm_finetune/$protein data=$protein model=d3pm problem=protein_classifier_discrete algorithm=daps_discrete

done

