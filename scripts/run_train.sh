# python main.py mode=train name=kp_R50_FPN_ITR_2e4 num_gpus=2 max_iters=20000

# python main.py mode=train name=kp_R50_FPN_LR_0.02_ITR_2e4 num_gpus=2 learning_rate=0.02 max_iters=20000

# python main.py mode=train name=kp_R50_FPN_LR_0.002_ITR_2e4 num_gpus=2 learning_rate=0.002 max_iters=20000

# python main.py mode=train name=kp_R50_FPN_LR_0.002_ITR_5e4 num_gpus=2 learning_rate=0.002 max_iters=50000

names=(kp_R50_FPN_ITR_2e4 kp_R50_FPN_LR_0.02_ITR_2e4 kp_R50_FPN_LR_0.002_ITR_2e4 kp_R50_FPN_LR_0.002_ITR_5e4)

for name in "${names[@]}"; do

    python main.py mode=test name=eval_kp_R50_FPN_ITR_2e4 checkpoint="ckpt/${name}/model_final.pth"

done
