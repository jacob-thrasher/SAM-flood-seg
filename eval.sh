source ~/miniconda3/bin/activate
conda activate erie

python eval.py --model floodseg_bbox_k1_42  --do_finetune_weights 0    --output_name eval_WVFlood
python eval.py --model floodseg_point_k1_42 --do_finetune_weights 0    --output_name eval_WVFlood
python eval.py --model floodseg_point_k3_42 --do_finetune_weights 0    --output_name eval_WVFlood
python eval.py --model floodseg_each_k3_42  --do_finetune_weights 0    --output_name eval_WVFlood
python eval.py --model sen1flood11_bbox_k1_42  --do_finetune_weights 0 --output_name eval_WVFlood
python eval.py --model sen1flood11_point_k1_42 --do_finetune_weights 0 --output_name eval_WVFlood
python eval.py --model sen1flood11_point_k3_42 --do_finetune_weights 0 --output_name eval_WVFlood
python eval.py --model sen1flood11_each_k3_42  --do_finetune_weights 0 --output_name eval_WVFlood
python eval.py --model floodseg_bbox_k1_42  --do_finetune_weights 1    --output_name eval_WVFlood
python eval.py --model floodseg_point_k1_42 --do_finetune_weights 1    --output_name eval_WVFlood
python eval.py --model floodseg_point_k3_42 --do_finetune_weights 1    --output_name eval_WVFlood
python eval.py --model floodseg_each_k3_42  --do_finetune_weights 1    --output_name eval_WVFlood
python eval.py --model sen1flood11_bbox_k1_42  --do_finetune_weights 1 --output_name eval_WVFlood
python eval.py --model sen1flood11_point_k1_42 --do_finetune_weights 1 --output_name eval_WVFlood
python eval.py --model sen1flood11_point_k3_42 --do_finetune_weights 1 --output_name eval_WVFlood
python eval.py --model sen1flood11_each_k3_42  --do_finetune_weights 1 --output_name eval_WVFlood