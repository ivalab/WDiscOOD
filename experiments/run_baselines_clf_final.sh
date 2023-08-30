# Run the baseline models and the scoring methods

# settings
ARCH=$1    # resnet50, vit_b

SCORE=${2}
# choices: [MSP, Energy, MahaVanilla, KNN, VIM, Residual, ReAct, ODIN]

# load file
if [[ $ARCH == "vit_b" ]]; then
    FEAT_NORM=1
else
    FEAT_NORM=0
fi



# get started
echo ""
echo ""
echo "=============================== ImageNet Benchmark: Method - ${SCORE}; ARCH - ${ARCH} =============================="

if [[ "$SCORE" == "KNN" ]]; then
    python test_feat_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together \
        --feat_norm 1

elif [[ "$SCORE" == "ReAct" ]]; then
    python test_feat_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together 

elif [[ "$SCORE" == "MahaVanilla" ]]; then
    python test_feat_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --feat_norm ${FEAT_NORM}

elif [[ "$SCORE" == "Residual" ]]; then
    python test_feat_logit_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "VIM" ]]; then
    python test_feat_logit_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

else
    python test_baselines.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --test_bs 20
fi