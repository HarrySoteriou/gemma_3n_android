#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

tmp_dir=$(mktemp -d)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

QAIRT_URL='https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.34.0.250424/v2.34.0.250424.zip'
QAIRT_CONTENT_DIR='qairt/2.34.0.250424'

pushd "$tmp_dir"
wget "$QAIRT_URL" -O qairt_sdk.zip
unzip qairt_sdk.zip *.so > /dev/null
popd

SOURCE_DIR="${tmp_dir}/${QAIRT_CONTENT_DIR}"
QNN_VERSIONS=(69 73 75 79)
JNI_ARM64_DIR="src/main/jni/arm64-v8a"
DEST_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))

for version in "${QNN_VERSIONS[@]}"; do
  echo "Copying libraries to ${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtp.so
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnHtp.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnSystem.so
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnSystem.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtpV${version}Skel.so
  cp -rf "${SOURCE_DIR}/lib/hexagon-v${version}/unsigned/libQnnHtpV${version}Skel.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtpV${version}Stub.so
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnHtpV${version}Stub.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"
done
