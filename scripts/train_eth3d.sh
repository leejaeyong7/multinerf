MOUNT_DIR="/home/lee896/data/$(hostname -s)"
DATA_DIR="${MOUNT_DIR}/eth3d_processed"
CHECKPOINT_DIR="/${MOUNT_DIR}/eth3d_outputs_multinerf"
mkdir -p $CHECKPOINT_DIR
scene=$1
rm -r "${CHECKPOINT_DIR}/${scene}/"
python -m train --gin_configs=configs/eth3d.gin \
                --gin_bindings="Config.data_dir = '${DATA_DIR}/${scene}/'" \
                --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${scene}/'" \
                --gin_bindings="Config.full_training = False" \
python -m write --gin_configs=configs/eth3d.gin \
                --gin_bindings="Config.data_dir = '${DATA_DIR}/${scene}/'" \
                --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${scene}/'" \
                --gin_bindings="Config.full_training = False" \

# after training, upload results to jyl.kr
zip -r ${CHECKPOINT_DIR}/${scene}/${scene}.zip ${CHECKPOINT_DIR}/${scene}/results
scp -r ${CHECKPOINT_DIR}/${scene}/${scene}.zip jyl:/mnt/data1/cluster_results/multinerf/
