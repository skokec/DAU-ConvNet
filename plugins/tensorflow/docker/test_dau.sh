#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
PYTHON_EXEC=$1

apt update && apt install -y python-tk
${PYTHON_EXEC} -m pip install --no-cache-dir scipy matplotlib==2.2

${PYTHON_EXEC} -m dau_conv.test DAUConvTest.test_DAUConv

STATUS=$?
exit $STATUS
