for protein in TrpB CreiLOV GB1
do  

    python sample.py pretrained_ckpt=continuous/$protein data=$protein model=continuous problem=protein

    python sample.py pretrained_ckpt=continuous_ESM/$protein data=$protein model=continuous problem=protein

    python sample.py pretrained_ckpt=d3pm/$protein data=$protein model=d3pm problem=protein

    python sample.py pretrained_ckpt=d3pm_finetune/$protein data=$protein model=d3pm problem=protein

    python sample.py pretrained_ckpt=mdlm/$protein data=$protein model=mdlm problem=protein

    python sample.py pretrained_ckpt=udlm/$protein data=$protein model=udlm problem=protein

    python sample.py pretrained_ckpt=causalLM_finetune/$protein data=$protein model=causalLM problem=protein

done
