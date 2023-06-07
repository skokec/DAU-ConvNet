#!/bin/bash

# This script builds dau-conv package for various TensorFlow and Python version
#
# Below is defined a list of all tensorflow builds (TF_BUILDS) and python builds (PYTHON_BUILDS)
# for which DAU-ConvNet package will be build. 
#
# This script performs:
#  1. For all combinations of TensorFlow na Python version perform build using a prepared docker file
#
#  2. After all images are build it performs the following tests:
#    - integirety check by running "import dau_conv" within container
#    - quick unit test by running test "python tests/dau_conv_test.py DAUConvTest.test_DAUConvQuick
#
#  3. Wheel packages (.whl) are stored to the same location of this script.

#DOCKER_EXEC=nvidia-docker
DOCKER_EXEC=docker

DAU_VERSION=1.0
DOCKER_IMG_NAME=dau-convnet
UNITTEST_DOCKER=0
DOCKER_HUB_REPO=""

# python build numbers
PYTHON_BUILDS=(3.7 )

# list of all TensorFlow version with corresponding nvidia/cuda image version 
# version and base image str are seperated by semicolumn (;)
TF_BUILDS=("1.15.5;nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04" \
           "1.14.0;nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04")



for i in "$@"
do
case $i in
    --dau-version=*)
    DAU_VERSION="${i#*=}"
    shift # past argument
    ;;
    --docker-basename=*)
    DOCKER_IMG_NAME="${i#*=}"
    shift # past argument
    ;;
    --python-builds=*)
    PYTHON_BUILDS=""
    IFS=',' read -r -a PYTHON_BUILDS <<< "${i#*=}"
    shift # past argument
    ;;
    --tf-builds=*)
    TF_BUILDS=""
    IFS=',' read -r -a TF_BUILDS <<< "${i#*=}"
    shift # past argument
    ;;
    --unit-test)
    UNITTEST_DOCKER=1
    shift # past argument
    ;;
    --docker-hub-repo=*)
    DOCKER_HUB_REPO="${i#*=}"
    shift # past argument
    ;;

    *)
	 # unknown option
    ;;
esac
done

CURR_DIR=$(dirname "$(realpath "$0")")
if [ -f "$CURR_DIR/../docker/data.tar.gz" ]; then
  rm "$CURR_DIR/../docker/data.tar.gz"
fi
tar -czvf "$(realpath "$CURR_DIR/../docker/data.tar.gz")" --exclude='build*' --exclude='.git' -C "$(realpath "$CURR_DIR/../../../")" .

echo "Settings:"
echo "  DAU_VERSION=${DAU_VERSION}"
echo "  DOCKER_IMG_NAME=${DOCKER_IMG_NAME}"
echo "  PYTHON_BUILDS=${PYTHON_BUILDS[*]}"
echo "  TF_BUILDS=${TF_BUILDS[*]}"
echo "  UNITTEST_DOCKER=${UNITTEST_DOCKER}"

echo "Building docker images for:"
for TF_VER_BUILD_STR in "${TF_BUILDS[@]}"
do
  IFS=";" read -r -a TF_VER_BUILD <<< "${TF_VER_BUILD_STR}"
  TF_VER=${TF_VER_BUILD[0]}
  TF_BASE_IMAGE=${TF_VER_BUILD[1]}
  for PY_VER in "${PYTHON_BUILDS[@]}"
  do
    DOCKER_IMG_TAG=${DAU_VERSION}-py${PY_VER}-tf${TF_VER}

    echo -n "  ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} ... "

    BUILD_LOG="build_dau_${DOCKER_IMG_TAG}.log"
    PY_VER_MAJOR=${PY_VER%.*}
    if [ ${PY_VER_MAJOR} -eq 2 ]; then
       PY_VER_MAJOR=""
    fi

    DAU_CMAKE_FLAGS="-DPACKAGE_VERSION=${DAU_VERSION}"
    
    DOCKERFILE_VERSION=""
    if [[ $TF_BASE_IMAGE == *"ubuntu18.04"* ]]; then
      DOCKERFILE_VERSION=".ubuntu18.04"
    fi

    ${DOCKER_EXEC} build -t ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} \
			--build-arg BASE_CUDA_VERSION=${TF_BASE_IMAGE} \
			--build-arg TF_VER=${TF_VER} \
			--build-arg PY_VER=${PY_VER} \
			--build-arg PY_VER_MAJOR="${PY_VER_MAJOR}" \
			--build-arg DAU_CMAKE_FLAGS=${DAU_CMAKE_FLAGS} -f docker/Dockerfile${DOCKERFILE_VERSION} docker/ >& ${BUILD_LOG}
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
      echo "ERROR: check ${BUILD_LOG} for logs."
    else
      echo "OK"
    fi
  done
done

# Run each docker for unit-test and extract whl file
for TF_VER_BUILD_STR in "${TF_BUILDS[@]}"
do
  IFS=";" read -r -a TF_VER_BUILD <<< "${TF_VER_BUILD_STR}"
  TF_VER=${TF_VER_BUILD[0]}
  TF_BASE_IMAGE=${TF_VER_BUILD[1]}
  for PY_VER in "${PYTHON_BUILDS[@]}"
  do
    PY_VER_MAJOR=${PY_VER%.*}
    PY_VER_STR=${PY_VER//.}
    PYTHON_EXEC=/usr/bin/python${PY_VER}

    DOCKER_IMG_TAG=${DAU_VERSION}-py${PY_VER}-tf${TF_VER}
    CONTAINER_NAME="integration-testing-dau-convnet-${DOCKER_IMG_TAG}"

    echo "Testing ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG}:"

    echo -n "  Verifying dau_conv package integrity ... "
    ${DOCKER_EXEC} run -i --rm --name ${CONTAINER_NAME} ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} /usr/bin/python${PY_VER} /opt/verify_dau_import.py
    STATUS=$?
    
    if [ ${STATUS} -ne 0 ]; then
      echo "ERROR: cannot run 'import dau_conv'"
    else
      echo "OK"

      if [ ${UNITTEST_DOCKER} -ne 0 ]; then
        UNITTEST_LOG="test_dau_${DOCKER_IMG_TAG}.log"
        echo -n "  Running UnitTest ... "
        ${DOCKER_EXEC} run -i --rm --name ${CONTAINER_NAME} ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} /bin/bash /opt/test_dau.sh ${PYTHON_EXEC} &> ${UNITTEST_LOG}
        STATUS=$?

        if [ ${STATUS} -ne 0 ]; then
          echo "ERROR: check ${UNITTEST_LOG} for logs."
        else
          echo "OK"
        fi
      fi

      echo -n "  Copying .whl package to build-ci ... "
      WHL_STR="py${PY_VER_MAJOR}-none-any"
      if [ ${PY_VER_MAJOR} -eq 2 ]; then
         WHL_REPLACEMENT_STR="cp${PY_VER_STR}-cp${PY_VER_STR}mu-manylinux1_x86_64"
      else
         WHL_REPLACEMENT_STR="cp${PY_VER_STR}-cp${PY_VER_STR}m-manylinux1_x86_64"
      fi

      WHL_TMP_DIR=/tmp/whl-${DOCKER_IMG_TAG}
      if [ ! -d "$WHL_TMP_DIR" ]; then
        mkdir $WHL_TMP_DIR
      fi
      ${DOCKER_EXEC} create --name dummy ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} /bin/bash
      ${DOCKER_EXEC} cp dummy:/opt/dau-convnet/build/plugins/tensorflow/wheelhouse/ ${WHL_TMP_DIR}/.
      ${DOCKER_EXEC} rm -f dummy
      WHL_TMP_DIR=$WHL_TMP_DIR/wheelhouse      

      for file in $WHL_TMP_DIR/*.whl; do
        echo mv "$file" "${file/$WHL_STR/$WHL_REPLACEMENT_STR}"
      done
      mv -f $WHL_TMP_DIR/*.whl `dirname "$0"`/.
      rm -rf $WHL_TMP_DIR
      echo "done"

     if [ ! -z "${DOCKER_HUB_REPO}"  ]; then
       echo -n "  Tagging and pushing docker to DockerHub ... "

       DOCKERPUSH_LOG="docker_push_dau_${DOCKER_IMG_TAG}.log"

       ${DOCKER_EXEC} tag ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} ${DOCKER_HUB_REPO}/${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} >& /dev/null
       ${DOCKER_EXEC} push ${DOCKER_HUB_REPO}/${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} &> ${DOCKERPUSH_LOG}
       STATUS=$?

       if [ ${STATUS} -ne 0 ]; then
	 echo "ERROR: check ${DOCKERPUSH_LOG} for logs."
       else
         echo "OK"
       fi
     fi
    fi
  done
done

