for i in {0..9}
do
   echo "Working on $i th fold."
   python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42
done
