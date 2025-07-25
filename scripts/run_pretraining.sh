for protein in GB1 #TrpB CreiLOV
do  

    CUDA_VISIBLE_DEVICES=1 python pretrain.py pretrain_model=continuous data=$protein

    #python pretrain.py pretrain_model=continuous_ESM data=$protein

    #python pretrain.py pretrain_model=d3pm data=$protein

    # python pretrain.py pretrain_model=d3pm_finetune data=$protein

    #python pretrain.py pretrain_model=udlm data=$protein

    CUDA_VISIBLE_DEVICES=1 python pretrain.py pretrain_model=mdlm data=$protein

    CUDA_VISIBLE_DEVICES=1 python pretrain.py pretrain_model=causalLM_finetune data=$protein

done