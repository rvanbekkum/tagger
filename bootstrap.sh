#!/usr/bin/env bash

echo 'LC_ALL="en_US.UTF-8"' | sudo tee -a /etc/environment

sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-numpy python3-scipy

sudo pip3 install virtualenv

cd /home/vagrant/tag-prediction
virtualenv .venv -p python3
source .venv/bin/activate
pip install -r requirements.txt
