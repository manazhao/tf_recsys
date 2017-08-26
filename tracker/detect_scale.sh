#!/bin/bash
set -e

source ${HOME}/git/tf_recsys/scripts/common.sh

IMAGE=$1
for size in 200 250 300 350 400 450 500; do
  SCALED_IMAGE="$(suffix_filename ${IMAGE} _${size})"
  echo ${SCALED_IMAGE}
  convert -quality 100 -resize ${size}x${size} ${IMAGE} ${SCALED_IMAGE}
  python peopledetect.py ${SCALED_IMAGE}
done
