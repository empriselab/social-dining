#!/bin/bash
#SBATCH -J ipaznet_model                         # Job name
#SBATCH -o slurm_outs/ipaznet_%j.out                  # output file (%j expands to jobID)
#SBATCH -e slurm_outs/ipaznet_%j.err                  # error log file (%j expands to jobID)
#SBATCH -c 6                                 # Number of CPUs requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64000                           # server memory requested (per node)
#SBATCH -t 8:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=bhattacharjee                      # Request partition
#SBATCH --nodelist=bhattacharjee-compute-01  # Set specific node
#SBATCH --gres=gpu:1                 # Type/number of GPUs needed

# 70:30
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features gaze headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features gaze speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features gaze headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features gaze 
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features speaking

# 10-fold
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features gaze headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features gaze speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features gaze headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features gaze 
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "10-fold" --features speaking

# loso-session
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features speaking


# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face


# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face gaze headpose speaking
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face speaking gaze
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face speaking headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face gaze headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features speaking face gaze headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features speaking body gaze headpose



# IGNORE. PROBABLY DELETE
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face gaze 

# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body gaze
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face headpose
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features 




# Double number of filters

# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose speaking --filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features headpose speaking  -filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze speaking --filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose --filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze --filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features headpose --filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features speaking --filter_scale 2 --frame_length 90


# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body --filter_scale 2 --frame_length 90
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face --filter_scale 2 --frame_length 90

#python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face gaze headpose speaking --filter_scale 2 --frame_length 90 --no_scaling 1
#python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face speaking gaze --filter_scale 2 --frame_length 90 --no_scaling 1
#python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face speaking headpose --filter_scale 2 --frame_length 90 --no_scaling 1
#python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features body face gaze headpose --filter_scale 2 --frame_length 90 --no_scaling 1
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features speaking face gaze headpose --filter_scale 2 --frame_length 90 --no_scaling 1
#python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features speaking body gaze headpose --filter_scale 2 --frame_length 90 --no_scaling 1

# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2


# All feature SSP trainings

#  python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 1
#  python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 50
#  python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 1 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 1 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 1 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 100

# 70:30 variants
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 1 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 1 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 1 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 1
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 50
# python /home/aa2375/social-dining/src/models/interleaved_net.py --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 100


# seems like 100 works better. run loso versions of this.
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 2 --num_global_feats_repeat 100

# no ssp variant
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100

# no ssp feature ablations
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face gaze headpose speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100
# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face gaze headpose body --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100

# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features gaze headpose speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100
python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --num_global_feats_repeat 100


# python /home/aa2375/social-dining/src/models/interleaved_net.py --training_split "loso-session" --features face gaze headpose body speaking --filter_scale 2 --frame_length 90 --no_scaling 1 --use_ssp 0 --global_features None