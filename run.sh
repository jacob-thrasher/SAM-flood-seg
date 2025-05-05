source ~/miniconda3/bin/activate
conda activate erie

# python main.py --seed 42 --region_select bbox --dataset floodseg
# python main.py --seed 42 --region_select point --k 1 --dataset floodseg
# python main.py --seed 42 --region_select point --k 3 --dataset floodseg
# python main.py --seed 42 --region_select each --k 3 --dataset floodseg

python main.py --seed 42 --region_select bbox --dataset sen1flood11 
python main.py --seed 42 --region_select point --k 1 --dataset sen1flood11
python main.py --seed 42 --region_select point --k 3 --dataset sen1flood11
python main.py --seed 42 --region_select each --k 3 --dataset sen1flood11