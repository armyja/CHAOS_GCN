TRAINING_ROOT='/home/ubuntu/MyFiles/GCN'
RESUME=${TRAINING_ROOT}/snapshots/main_GCN_All_20190331_054658_35001.pkl
RESUME='0'
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
python train.py \
--resume ${RESUME} \
--batch_size 8 \
--epoch 12 \
--learning_rate 1e-3 \
--root_dir ${TRAINING_ROOT} \
--timestamp ${TRAINING_TIMESTAMP}  \
2>&1 | tee ${TRAINING_ROOT}/logs/${TRAINING_TIMESTAMP}.txt