#!/bin/bash

PYTHON=python
SEED=1

ARCH="ViT-B/32"
TEST_ATTACK_TYPE="pgd"

if [[ "${ARCH}" == "RN50" ]]; then
  TEST_EPS=4
  TEST_NUMSTEPS=10
  TEST_STEPSIZE=1

elif [[ "${ARCH}" == ViT* ]]; then 
  TEST_EPS=4
  TEST_NUMSTEPS=100
  TEST_STEPSIZE=1 
else
  echo "Unknown ARCH=${ARCH}" >&2
  exit 1
fi
ARCH_SAFE="${ARCH//\//-}"

BATCH_SIZE=128
MAC_EPS=8 
MAC_NUMSTEPS=4
TAU_TEMP=0.01
TAU_THRES=0.7
NUM_VIEWS=2

TEST_SET="Caltech101 DTD Flower102 Pets UCF101 Aircraft eurosat Cars SUN397 Food101 A V I R K" 

BASE_OUTDIR="MAC_results"
STAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUTDIR="${BASE_OUTDIR}/${STAMP}"
mkdir -p "${ROOT_OUTDIR}"

SCRIPT="code/mac.py"

OUTDIR="${ROOT_OUTDIR}/${ARCH_SAFE}"
mkdir -p "${OUTDIR}"

${PYTHON} "${SCRIPT}" \
  --batch_size ${BATCH_SIZE} \
  --test_attack_type "${TEST_ATTACK_TYPE}" \
  --test_eps ${TEST_EPS} \
  --test_numsteps ${TEST_NUMSTEPS} \
  --test_stepsize ${TEST_STEPSIZE} \
  --outdir "${OUTDIR}" \
  --seed ${SEED} \
  --mac_eps ${MAC_EPS} \
  --mac_numsteps ${MAC_NUMSTEPS} \
  --tau_temp ${TAU_TEMP} \
  --tau_thres ${TAU_THRES} \
  --num_views ${NUM_VIEWS} \
  --arch "${ARCH}" \
  --test_set ${TEST_SET}



