#!/bin/bash

FILENAME=VoxelatedMandibles103
wget -O ./$FILENAME.tar.gz "https://drive.google.com/uc?export=download&id=1GCslF1eo6Bz2EK207CvBfRDUReb_dyXi" 
echo "Data downloaded"

mkdir data
tar xf ./$FILENAME.tar.gz -C ./data
echo "Data unzipped into directory: $PWD/data"

export MANDIBLE_DATA_PATH=$PWD/data
echo "Environment variable MANDIBLE_DATA_PATH set to $MANDIBLE_DATA_PATH"
