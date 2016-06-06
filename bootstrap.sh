#!/usr/bin/env bash

echo 'LC_ALL="en_US.UTF-8"' | sudo tee -a /etc/environment

sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools \
                        python3-numpy python3-scipy python3-pip libatlas-dev \
                        libatlas3gf-base python3-matplotlib

sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3

sudo pip3 install scikit-learn
