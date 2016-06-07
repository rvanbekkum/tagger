#!/usr/bin/env bash

echo 'LC_ALL="en_US.UTF-8"' | sudo tee -a /etc/environment

sudo apt-get update
sudo apt-get install -y libgtk2.0-0

cd /home/vagrant
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

echo 'PATH=/home/vagrant/miniconda3/bin:$PATH' | tee -a /home/vagrant/.bashrc

source /home/vagrant/.bashrc

./miniconda3/bin/conda install numpy scipy scikit-learn matplotlib
./miniconda3/bin/conda install -c menpo opencv3=3.1.0
