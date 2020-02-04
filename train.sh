python main_reid.py train --trainset duke_sct --testset duke_sct --save_dir path_to_save/duke_sct_mcnl --max_epoch 200 --eval_step 20 --model_name="distance" --num_gpu 2 --train_batch 256 --num_instances 8 --sampler="randomidentitycamera" --margin1 0.1 --margin2 0.1 --optstep="100,200"

python main_reid.py train --trainset duke_sct --testset duke_sct --save_dir path_to_save/duke_sct_trip --max_epoch 200 --eval_step 20 --model_name="triplet"  --num_gpu 2 --train_batch 256 --num_instances 8 --sampler="randomidentity" --margin 0.3 --optstep="100,200"

python main_reid.py train --trainset market_sct --testset market_sct --save_dir path_to_save/market_sct_mcnl --max_epoch 200 --eval_step 20 --model_name="distance" --num_gpu 2 --train_batch 240 --num_instances 8 --sampler="randomidentitycamera" --margin1 0.1 --margin2 0.1 --optstep="100,200"

python main_reid.py train --trainset market_sct --testset market_sct --save_dir path_to_save/market_sct_trip --max_epoch 200 --eval_step 20 --model_name="triplet"  --num_gpu 2 --train_batch 240 --num_instances 8 --sampler="randomidentity" --margin 0.3 --optstep="100,200"
