ARCH=$1     # resnet50 or vit_b or resnet50_supcon or resnet50_clip
FEAT_SPACE=0
SCORE=WDiscOOD
NUM_DISC=1000
RES_WEIGHT=5
SCORE_G=ClsEucl
SCORE_H=CenterEucl
FEAT_NORM=0
LOAD_FILE=None

# load file
if [[ $ARCH == "resnet50_supcon" ]]; then
    LOAD_FILE=./pretrained_models/ImageNet/supcon.pth
    FEAT_NORM=1
elif [[ "$ARCH" == "vit_b" ]]; then
    FEAT_NORM=1
    NUM_DISC=512
    RES_WEIGHT=1
fi

# Try
 
echo ""
echo ""
echo "==================================== Test WDiscOOD - Imagnet, "${ARCH}" ================================="


python test_feat_disc.py \
    --id_dset imagenet \
    --id_train_num 200000 \
    --large_scale \
    --ood_dsets textures sun places inat imagenet_o openimage_o \
    --arch ${ARCH} \
    --load_file ${LOAD_FILE} \
    --score ${SCORE} \
    --feat_norm ${FEAT_NORM} \
    --feat_space ${FEAT_SPACE} \
    --num_disc ${NUM_DISC} \
    --res_dist_weight ${RES_WEIGHT} \
    --score_g ${SCORE_G} \
    --score_h ${SCORE_H}
