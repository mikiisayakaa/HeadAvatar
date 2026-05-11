PYTHON=python
DEVICE=0

echo "Device ID = $DEVICE"
names=("074" "104" "140" "165" "175" "210" "218" "238" "253" "264" "302" "306")

for name in "${names[@]}"; do
    # echo "Training single frame for subject $name"    
    DATA_DIR=./data/${name}
    BASE_DIR=./output/new/new_${name}_seq
    # $PYTHON train.py -s ${DATA_DIR} -m ${BASE_DIR} --eval --gpuID ${DEVICE}

    # echo "Generating training frames for subject $name"
    # $PYTHON get_training_frames.py --sample ${name} --num_frames 20 --test

    echo "Testing for subject $name"
    echo "NAME = '$name'"
    echo "DATA_DIR = '$DATA_DIR'"
    $PYTHON test.py -s ${DATA_DIR} -l ${BASE_DIR} --gpuID ${DEVICE} --train
done
