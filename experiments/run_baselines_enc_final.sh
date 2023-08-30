# Run the baseline models and the scoring methods

# settings
ARCH=${1}    # resnet50_clip or resnet50_supcon

SCORE=${2}
# choices: [MahaVanilla, KNN]


# get started
echo ""
echo ""
echo "=============================== ImageNet SupCon Benchmark: ${SCORE} score =============================="

if [[ "$SCORE" == "KNN" ]]; then
    python test_feat_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --feat_space 0 \
        --run_together
        #--feat_norm 1 \       # KNN will always normalize the feature (since it even normalizes ResNet50_CE features), so no need to provide this parameter
elif [[ "$SCORE" == "MahaVanilla" ]]; then
    if [[ "${ARCH}" == "resnet50_supcon" ]]; then
        FEAT_NORM=1
    else
        FEAT_NORM=0
    fi
    python test_feat_ood.py \
        --id_dset imagenet \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --feat_space 0 \
        --feat_norm ${FEAT_NORM} 
fi