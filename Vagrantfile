# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/trusty64"
  config.vm.provider "virtualbox" do |v|
    v.customize ["modifyvm", :id, "--nictype1", "virtio"]
  end
  config.vm.synced_folder ".", "/home/vagrant/tag-prediction"
  config.vm.provision "shell", path: "bootstrap.sh", privileged: false
end
