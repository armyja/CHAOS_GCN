TRAINING_ROOT='/home/ubuntu/MyFiles/GCN'
#TRAINING_ROOT='/media/jeffrey/D/CHAOS'
RESUME=${TRAINING_ROOT}/snapshots/main_GCN_All_20190521_050103FCN_GCN_ALL_10000.pkl
RESUME='0'
CURRENT_FOLD=0
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')FCN_GCN_ALL
python train.py \
--resume ${RESUME} \
--current_fold ${CURRENT_FOLD} \
--batch_size 4 \
--epoch 50 \
--learning_rate 1e-3 \
--root_dir ${TRAINING_ROOT} \
--timestamp ${TRAINING_TIMESTAMP} \
2>&1 | tee ${TRAINING_ROOT}/logs/${TRAINING_TIMESTAMP}.txt
