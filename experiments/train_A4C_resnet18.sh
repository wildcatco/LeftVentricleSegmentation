for lr in 0.001 0.0003 0.0001 0.00003
do
    for wd in 0.01 0.001 0.0001
    do
        python train.py --dataset A4C\
                        --encoder resnet18\
                        --num-workers 4\
                        --batch-size 4\
                        --lr $lr\
                        --wd $wd\
                        --max-epochs 60
    done
done