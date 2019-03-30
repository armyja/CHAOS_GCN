TRAINING_ROOT='/media/jeffrey/D/CHAOS'
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
python train.py --timestamp ${TRAINING_TIMESTAMP}  2>&1 | tee ${TRAINING_ROOT}/logs/${TRAINING_TIMESTAMP}.txt