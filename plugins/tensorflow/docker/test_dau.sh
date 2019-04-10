#!/bin/bash

PYTHON_EXEC=$1

apt install -y python-tk
${PYTHON_EXEC} -m pip install scipy matplotlib==2.2

${PYTHON_EXEC} /opt/dau-convnet/plugins/tensorflow/tests/dau_conv_test.py DAUConvTest.test_DAUConvQuick

STATUS=$?
exit $STATUS
