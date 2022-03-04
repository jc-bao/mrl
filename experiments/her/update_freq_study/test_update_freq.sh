exp="update_freq_study"
opt_every="1 10 100 1000"

for opt in $opt_every;
do
    PYTHONPATH=../../../ python ../train_her.py --env FetchPickAndPlace-v1 --num_envs 8 --parent_folder ./results/opt$opt --her future_4 --optimize_every $opt --optimize_times $opt --prefix opt$opt
done