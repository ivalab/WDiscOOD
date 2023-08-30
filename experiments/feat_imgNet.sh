ARCH=$1     # resnet50 or vit_b or resnet50_clip or resnet50_supcon or vit_b16_clip or vit_b32_clip


if [[ "$ARCH" == "resnet50_supcon" ]]; then
    LOAD_FILE=./pretrained_models/ImageNet/supcon.pth 
else
    LOAD_FILE=None
fi


python feat_extraction.py \
        --id_dset imagenet \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch $ARCH \
        --large_scale \
        --id_train_num 200000 \
        --rerun \
        --test_bs 16 \
        --load_file ${LOAD_FILE}
