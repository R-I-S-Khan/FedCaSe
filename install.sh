#!/bin/bash
FEDSCALE_HOME=$(pwd)
FEDCASE_HOME=$(pwd)

echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc
echo export FEDCASE_HOME=$(pwd) >> ~/.bashrc

echo alias fedscale=\'bash ${FEDSCALE_HOME}/fedscale.sh\' >> ~/.bashrc
echo alias fedcase=\'bash ${FEDCASE_HOME}/fedscale.sh\' >> ~/.bashrc

isPackageNotInstalled() {
  $1 --version &> /dev/null
  if [ $? -eq 0 ]; then
    echo "$1: Already installed"
  else
    install_dir=$HOME/anaconda3
    wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH
  fi
}

# un-comment to install anaconda
isPackageNotInstalled conda


# create conda env
conda init bash
. ~/.bashrc
conda env create -f environment.yml # Install dependencies
conda activate fedscale


if [ "$1" == "--cuda" ]; then
  sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
  sudo apt-get --purge remove "*nvidia*"
  sudo apt-get autoremove
  sudo apt-get autoclean
  sudo /usr/local/cuda-11.2/bin/cuda-uninstaller
  sudo rm -rf /usr/local/cuda*
  sudo /usr/bin/nvidia-uninstall
  sudo apt-get update
  #wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
  sudo apt-get purge nvidia-* -y
  sudo sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
  sudo update-initramfs -u
  #sudo sh cuda_10.2.89_440.33.01_linux.run --override --driver --toolkit --samples --silent
  sudo sh cuda_11.3.0_465.19.01_linux.run --override --driver --toolkit --samples --silent
  export PATH=$PATH:/usr/local/cuda-11.3/
  conda install cudatoolkit=11.3 -y
fi

