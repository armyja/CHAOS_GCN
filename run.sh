TRAINING_ROOT='/home/ubuntu/MyFiles/GCN'
TRAINING_ROOT='/media/jeffrey/D/CHAOS'
RESUME=${TRAINING_ROOT}/snapshots/main_GCN_All_20190402_124203_2001.pkl
RESUME='0'
CURRENT_FOLD=0
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
python train.py \
--resume ${RESUME} \
--current_fold ${CURRENT_FOLD} \
--batch_size 4 \
--epoch 100 \
--learning_rate 1e-3 \
--root_dir ${TRAINING_ROOT} \
--timestamp ${TRAINING_TIMESTAMP}  \
2>&1 | tee ${TRAINING_ROOT}/logs/${TRAINING_TIMESTAMP}.txt