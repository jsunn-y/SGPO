<<<<<<< HEAD
<<<<<<< HEAD
for protein in TrpB CreiLOV GB1
=======
for protein in GB1 #TrpB CreiLOV
>>>>>>> 032bd50 (bug fixes and running new analysis for GB1)
=======
for protein in GB1 #TrpB CreiLOV
>>>>>>> 72d9410 (bug fixes and running new analysis for GB1)
do  

    python pareto.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_classifier_continuous algorithm=cls_guidance_continuous

    python pareto_NOS_hyperparameter.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_NOS_continuous algorithm=NOS_continuous

    python pareto.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_classifier_continuous algorithm=daps_continuous

    python pareto.py pretrained_ckpt=d3pm_finetune/$protein data=$protein problem=protein_classifier_discrete model=d3pm algorithm=cls_guidance_discrete

    python pareto.py pretrained_ckpt=d3pm_finetune/$protein data=$protein model=d3pm problem=protein_classifier_discrete algorithm=daps_discrete

<<<<<<< HEAD
<<<<<<< HEAD
    python pareto.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_classifier_discrete model=mdlm algorithm=cls_guidance_discrete

    python pareto.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein_classifier_discrete algorithm=daps_discrete

    python pareto_NOS_hyperparameter.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein_NOS_discrete algorithm=NOS_discrete

done

for protein in TrpB CreiLOV GB1
do  

    python pareto.py pretrained_ckpt=causalLM_finetune/$protein data=$protein model=causalLM problem=protein_DPO algorithm=DPO
    
done
=======
done

>>>>>>> 032bd50 (bug fixes and running new analysis for GB1)
=======
done

>>>>>>> 72d9410 (bug fixes and running new analysis for GB1)
