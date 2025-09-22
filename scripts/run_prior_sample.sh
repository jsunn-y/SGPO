for protein in TrpB CreiLOV GB1
do  

    python prior_sample.py pretrained_ckpt=continuous/$protein data=$protein model=continuous

    python prior_sample.py pretrained_ckpt=continuous_ESM/$protein data=$protein model=continuous

    python prior_sample.py pretrained_ckpt=d3pm/$protein data=$protein model=d3pm
    
    python prior_sample.py pretrained_ckpt=d3pm_finetune/$protein data=$protein model=d3pm

    python prior_sample.py pretrained_ckpt=udlm/$protein data=$protein model=udlm

    python prior_sample.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm

    python prior_sample.py pretrained_ckpt=causalLM_finetune/$protein data=$protein model=causalLM 

done
