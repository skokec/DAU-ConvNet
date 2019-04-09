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


# list of all TensorFlow version with corresponding nvidia/cuda image version 
# version and base image str are seperated by semicolumn (;)
TF_BUILDS=("1.13.1;nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04")

# python build numbers
#PYTHON_BUILDS=(2.7 3.3 3.4 3.5 3.6 3.7)
PYTHON_BUILDS=(3.5)

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
    nvidia-docker build -t dau-convnet:py${PY_VER}-r${TF_VER} \
			--build-arg BASE_CUDA_VERSION=${TF_BASE_IMAGE} \
			--build-arg TF_VER=${TF_VER} \
			--build-arg PY_VER=${PY_VER} \
			--build-arg PY_VER_MAJOR=${PY_VER%.*} docker/ >& ${BUILD_LOG}
    echo "done"

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
    CONTAINER_NAME="integration-testing-dau-convnet-py${PY_VER}-r${TF_VER}"
    echo "Testing dau-convnet:py${PY_VER}-r${TF_VER}:"

    echo -n "  Verifying dau_conv package integrity ... "
    STATUS=`nvidia-docker run -i --rm --name ${CONTAINER_NAME} dau-convnet:py${PY_VER}-r${TF_VER} /usr/bin/python${PY_VER} /opt/verify_dau_import.py`

    if [ ${STATUS} -ne 0 ]; then
      echo "ERROR: cannot run 'import dau_conv'"
    else
      echo "OK"

      UNITTEST_LOG="test_dau_py${PY_VER}_r${TF_VER}.log"
      echo -n "  Running UnitTest ... "
      nvidia-docker run -i --rm --name ${CONTAINER_NAME} dau-convnet:py${PY_VER}-r${TF_VER} /bin/bash /opt/test_dau.sh ${PYTHON_EXEC} >& ${UNITTEST_LOG}
      STATUS=$?
      if [ ${STATUS} -ne 0 ]; then
        echo "ERROR: check ${UNITTEST_LOG} for logs."
      else
        echo "OK"
      fi
      echo -n "  Copying .whl package to build-ci ... "
      WHL_STR="py${PY_VER_MAJOR}-none-any"
      WHL_REPLACEMENT_STR="cp${PY_VER_STR}-cp${PY_VER_STR}m_linux_x86_64"

      WHL_TMP_DIR=/tmp/whl-${PY_VER}-r${TF_VER}
      mkdir $WHL_TMP_DIR
      nvidia-docker run -i --rm --name ${CONTAINER_NAME} -v $WHL_TMP_DIR:/opt/output dau-convnet:py${PY_VER}-r${TF_VER} \
	 /bin/bash -c "cp build/plugins/tensorflow/wheelhouse/*.whl /opt/output/ "
      rename "s/${WHL_STR}/${WHL_REPLACEMENT_STR}/g" $WHL_TMP_DIR/*.whl
      mv $WHL_TMP_DIR/*.whl `dirname "$0"`/.
      rm -rf $WHL_TMP_DIR
      echo "done"
    fi
  done
done

