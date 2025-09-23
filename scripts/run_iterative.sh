for protein in TrpB CreiLOV GB1
do  
    #D3PM + CG + Ensemble
    # python iterativeBO.py pretrained_ckpt=d3pm_finetune/$protein data=$protein problem=protein_classifier_discrete model=d3pm algorithm=cls_guidance_discrete

    #D3PM + CG + GP
    # CUDA_VISIBLE_DEVICES=1 python iterativeBO.py pretrained_ckpt=d3pm_finetune/$protein data=$protein problem=protein_classifier_discrete_GP model=d3pm algorithm=cls_guidance_discrete

    #MDLM + CG + Ensemble
    python iterativeBO.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_classifier_discrete model=mdlm algorithm=cls_guidance_discrete

    #MDLM + CG + GP
    # CUDA_VISIBLE_DEVICES=1 python iterativeBO.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_classifier_discrete_GP model=mdlm algorithm=cls_guidance_discrete

    # #D3PM + DAPS + Ensemble
    # python iterativeBO.py pretrained_ckpt=d3pm_finetune/$protein data=$protein model=d3pm problem=protein_classifier_discrete algorithm=daps_discrete

    # #D3PM + DAPS + GP
    # # CUDA_VISIBLE_DEVICES=1 python iterativeBO.py pretrained_ckpt=d3pm_finetune/$protein data=$protein model=d3pm problem=protein_classifier_discrete_GP algorithm=daps

    # #MDLM + DAPS + Ensemble
    python iterativeBO.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_classifier_discrete model=mdlm algorithm=daps_discrete

    # #MDLM + DAPS + GP
    # # CUDA_VISIBLE_DEVICES=1 python iterativeBO.py pretrained_ckpt=mdlm/$protein data=$protein problem=protein_classifier_discrete_GP model=mdlm algorithm=daps

    # Continuous + NOS baseline
    python iterativeBO.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein_NOS_continuous algorithm=NOS_continuous
done

for protein in TrpB CreiLOV GB1
do  
    # ARLM + DPO
    python iterativeBO.py pretrained_ckpt=causalLM_finetune/$protein data=$protein model=causalLM problem=protein_DPO algorithm=DPO 
done