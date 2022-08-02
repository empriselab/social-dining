#!/bin/bash
#SBATCH -J tcn_model                         # Job name
#SBATCH -o slurm_outs/tcn_model_%j.out                  # output file (%j expands to jobID)
#SBATCH -e slurm_outs/tcn_model_%j.err                  # error log file (%j expands to jobID)
#SBATCH -c 6                                 # Number of CPUs requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64000                           # server memory requested (per node)
#SBATCH -t 8:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=bhattacharjee                      # Request partition
#SBATCH --nodelist=bhattacharjee-compute-01  # Set specific node
#SBATCH --gres=gpu:1                 # Type/number of GPUs needed

# Must have one of each type of feature for these architectures...

# 70:30
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features gaze headpose speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features headpose speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features gaze speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features gaze headpose --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features gaze --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features headpose --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --features speaking --global_features None

# 10-fold
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features gaze headpose speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features headpose speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features gaze speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features gaze headpose --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features gaze --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features headpose --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "10-fold" --features speaking --global_features None

# loso-session
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze headpose speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features headpose speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze speaking --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze headpose --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features headpose --global_features None
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features speaking --global_features None


# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body --global_features None --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features face --global_features None --frame_length 90

# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose speaking --global_features None --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face speaking gaze --global_features None --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face speaking headpose --global_features None --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose --global_features None --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features speaking face gaze headpose --global_features None --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features speaking body gaze headpose --global_features None --frame_length 90


# With global features
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze speaking 
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features headpose speaking 
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze headpose speaking



# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze headpose 
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features gaze 
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features headpose 
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features speaking 

# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body 
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features face 

# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face speaking gaze
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face speaking headpose
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features speaking face gaze headpose
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features speaking body gaze headpose 


# Take the everything model and put it at 15fps
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose speaking --frame_length 90
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose speaking --frame_length 90 --use_ssp 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose speaking --frame_length 90 --use_ssp 2

# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split "loso-session" --features body face gaze headpose --frame_length 90 --use_ssp 2


# This is for the robot! 15fps ssp
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split 70:30 --features body face gaze headpose speaking --frame_length 90 --use_ssp 1 --save_scalers 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split 70:30 --features body face gaze headpose speaking --frame_length 90 --use_ssp 1 --no_scaling 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split 70:30 --features body face speaking --frame_length 90 --use_ssp 1 --no_scaling 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --model global --training_split 70:30 --features speaking --frame_length 90 --no_scaling 1 --global_features None --patience 50



# Let's run the other versions of TCN which require audio and video, but not necessarily global features
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features body face gaze headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features body face gaze speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features body face headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features body headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features body gaze speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features face headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features face gaze speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features body speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality --training_split "loso-session" --features face speaking

# Run for per_modality_seq version
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features body face gaze headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features body face gaze speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features body face headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features body headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features body gaze speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features face headpose speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features face gaze speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features body speaking
# python /home/aa2375/social-dining/src/models/tcn_model.py --model per_modality_seq --training_split "loso-session" --features face speaking



# 70:30 variants
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 0 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 0 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 0 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 1 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 1 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 1 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/tcn_model.py --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/tcn_model.py --training_split loso-session --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 100


# python /home/aa2375/social-dining/src/models/tcn_model.py --training_split "loso-session" --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 100
# python /home/aa2375/social-dining/src/models/tcn_model.py --training_split "loso-session" --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/tcn_model.py --training_split "loso-session" --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 25
# python /home/aa2375/social-dining/src/models/tcn_model.py --training_split "loso-session" --features face gaze headpose body speaking --frame_length 90 --use_ssp 2 --num_global_feats_repeat 1

python /home/aa2375/social-dining/src/models/tcn_model.py --training_split "loso-session" --features face gaze headpose body speaking --frame_length 90 --use_ssp 0 --num_global_feats_repeat 100
