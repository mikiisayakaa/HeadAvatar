PYTHON=python
DEVICE=0

echo "Device ID = $DEVICE"
names=("165")


for name in "${names[@]}"; do
    # echo "Training single frame for subject $name"    
    DATA_DIR=./data/${name}
    BASE_DIR=./output/new/new_${name}_base
    # $PYTHON train.py -s ${DATA_DIR} -m ${BASE_DIR} --eval --gpuID ${DEVICE}

    # echo "Generating training frames for subject $name"
    # $PYTHON get_training_frames.py --sample ${name}

    # echo "Training multi frame for subject $name"
    MULTI_DIR=./output/new/new_${name}_seq_0510
    echo "NAME = '$name'"
    echo "DATA_DIR = '$DATA_DIR'"
    $PYTHON train_seq.py -s ${DATA_DIR} -m ${MULTI_DIR} -l ${BASE_DIR} --eval --gpuID ${DEVICE}
done
