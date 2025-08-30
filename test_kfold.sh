for i in {0..9}
do
    echo "Working on $i th fold."
    version_dir="lightning_logs/version_$((80+$i))/checkpoints"
    
    # Check if the version directory exists
    if [ -d "$version_dir" ]; then
        # Find all .ckpt files in the directory
        for ckpt_file in "$version_dir"/last.ckpt; do
            # Check if the file exists (handles case where no .ckpt files found)
            if [ -f "$ckpt_file" ]; then
                echo "Evaluating checkpoint: $(basename "$ckpt_file")"
                python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 500 --loss bce --devices 0 --seed 42 --no_ndcg_loss --no_logistic_loss --test --ckpt_path "$ckpt_file"

            else
                echo "No .ckpt files found in $version_dir"
                break
            fi
        done
    else
        echo "Directory $version_dir does not exist, skipping fold $i"
    fi
done