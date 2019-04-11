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

DAU_VERSION=1.0
DOCKER_IMG_NAME=dau-convnet
UNITTEST_DOCKER=0

# python build numbers
PYTHON_BUILDS=(3.5 2.7)

# list of all TensorFlow version with corresponding nvidia/cuda image version 
# version and base image str are seperated by semicolumn (;)
TF_BUILDS=("1.13.1;nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04" \
           "1.12;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.11;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.10.1;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.10;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.9;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.8;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.7.1;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.7;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
           "1.6;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" )
# Not supported yet - needs code change
#          "1.5.1;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
#          "1.5;nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04" \
#          "1.4.1;nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04" \
#          "1.4;nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04")


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
    IFS=',' read -r -a PYTHON_BUILDS <<< "${i#*=}"
    shift # past argument
    ;;
    --tf-builds)
    IFS=',' read -r -a TF_BUILDS <<< "$2"
    shift # past argument
    ;;
    --unit-test)
    UNITTEST_DOCKER=1
    shift # past argument
    ;;
    *)
	 # unknown option
    ;;
esac
done

echo "Building docker images for:"
for TF_VER_BUILD_STR in "${TF_BUILDS[@]}"
do
  IFS=";" read -r -a TF_VER_BUILD <<< "${TF_VER_BUILD_STR}"
  TF_VER=${TF_VER_BUILD[0]}
  TF_BASE_IMAGE=${TF_VER_BUILD[1]}
  for PY_VER in "${PYTHON_BUILDS[@]}"
  do
    echo -n "  dau-convnet:py${PY_VER}-r${TF_VER} ... "
    BUILD_LOG="build_dau_py${PY_VER}_r${TF_VER}.log"
    PY_VER_MAJOR=${PY_VER%.*}
    if [ ${PY_VER_MAJOR} -eq 2 ]; then
       PY_VER_MAJOR=""
    fi

    DAU_CMAKE_FLAGS="-DPACKAGE_VERSION=${DAU_VERSION}"
    DOCKER_IMG_TAG=${DAU_VERSION}-py${PY_VER}-tf${TF_VER}

    nvidia-docker build -t ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} \
			--build-arg BASE_CUDA_VERSION=${TF_BASE_IMAGE} \
			--build-arg TF_VER=${TF_VER} \
			--build-arg PY_VER=${PY_VER} \
			--build-arg PY_VER_MAJOR="${PY_VER_MAJOR}" \
			--build-arg DAU_CMAKE_FLAGS=${DAU_CMAKE_FLAGS} docker/ >& ${BUILD_LOG}
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

    echo "Testing dau-convnet:py${PY_VER}-r${TF_VER}:"

    echo -n "  Verifying dau_conv package integrity ... "
    nvidia-docker run -i --rm --name ${CONTAINER_NAME} ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} /usr/bin/python${PY_VER} /opt/verify_dau_import.py
    STATUS=$?

    if [ ${STATUS} -ne 0 ]; then
      echo "ERROR: cannot run 'import dau_conv'"
    else
      echo "OK"

      if [ ${UNITTEST_DOCKER} -ne 0]; then
        UNITTEST_LOG="test_dau_py${PY_VER}_r${TF_VER}.log"
        echo -n "  Running UnitTest ... "
        nvidia-docker run -i --rm --name ${CONTAINER_NAME} dau-convnet:py${PY_VER}-r${TF_VER} /bin/bash /opt/test_dau.sh ${PYTHON_EXEC} &> ${UNITTEST_LOG}
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
      mkdir $WHL_TMP_DIR
      nvidia-docker run -i --rm --name ${CONTAINER_NAME} -v $WHL_TMP_DIR:/opt/output ${DOCKER_IMG_NAME}:${DOCKER_IMG_TAG} \
	 /bin/bash -c "cp build/plugins/tensorflow/wheelhouse/*.whl /opt/output/ "
      rename "s/${WHL_STR}/${WHL_REPLACEMENT_STR}/g" $WHL_TMP_DIR/*.whl
      mv -f $WHL_TMP_DIR/*.whl `dirname "$0"`/.
      rm -rf $WHL_TMP_DIR
      echo "done"
    fi
  done
done

