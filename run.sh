TRAINING_ROOT='/home/ubuntu/MyFiles/GCN'
RESUME='main_GCN_20190330_124553_17001.pkl'
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
python train.py \
--resume ${TRAINING_ROOT}/snapshots/${RESUME} \
--batch_size 8 \
--epoch 100 \
--learning_rate 1e-4 \
--root_dir ${TRAINING_ROOT} \
--timestamp ${TRAINING_TIMESTAMP}  \
2>&1 | tee ${TRAINING_ROOT}/logs/${TRAINING_TIMESTAMP}.txt