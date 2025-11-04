#!/usr/bin/env bash
set -e  # stop if any command fails

mkdir -p streethazards_train streethazards_test

# ---------------- Train Set ----------------
if [ ! -f streethazards_train/streethazards_train.tar ]; then
    echo "Downloading train set..."
    curl -o streethazards_train/streethazards_train.tar \
        https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar
else
    echo "Train tar file already exists, skipping download."
fi

if [ ! -d streethazards_train/streethazards ]; then
    echo "Extracting train set..."
    tar -xf streethazards_train/streethazards_train.tar -C streethazards_train
else
    echo "Train set already extracted, skipping extraction."
fi

# ---------------- Test Set ----------------
if [ ! -f streethazards_test/streethazards_test.tar ]; then
    echo "Downloading test set..."
    curl -o streethazards_test/streethazards_test.tar \
        https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar
else
    echo "Test tar file already exists, skipping download."
fi

if [ ! -d streethazards_test/streethazards ]; then
    echo "Extracting test set..."
    tar -xf streethazards_test/streethazards_test.tar -C streethazards_test
else
    echo "Test set already extracted, skipping extraction."
fi

echo "✅ All done!"