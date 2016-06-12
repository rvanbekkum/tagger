#!/usr/bin/env bash

echo 'LC_ALL="en_US.UTF-8"' | sudo tee -a /etc/environment

sudo apt-get update
sudo apt-get install -y libgtk2.0-0

cd /home/vagrant
wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh -b
rm Miniconda2-latest-Linux-x86_64.sh

echo 'PATH=/home/vagrant/miniconda2/bin:$PATH' | tee -a /home/vagrant/.bashrc

source /home/vagrant/.bashrc

./miniconda2/bin/conda install numpy scipy scikit-learn matplotlib opencv
