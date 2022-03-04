exp="batch_size_study"
batch_size="2000 20000 200000 2000000"
optimize_every="1 10 100 2000000"

for bs in $batch_size;
do
    PYTHONPATH=../../../ python ../train_her.py --env FetchPickAndPlace-v1 --num_envs 8 --parent_folder ./results/batch$bs --her future_4 --optimize_every $(($bs/2000)) --batch_size $bs --prefix batch$bs
done