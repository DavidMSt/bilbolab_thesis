#! /bin/bash
echo $'\n'" ****************** Starting Install Process ******************"$'\n'

echo $'\n'" ****************** Installing debian packages ******************"$'\n'
sudo apt-get install  libgl1 libcap-dev python3 python3-pip python3-picamera2  python3-dev python3-libcamera python3-kms++

echo $'\n'" ****************** Installing required pip packages ******************"$'\n'
pip3 install -r requirements.txt

echo $'\n'" ****************** Installing adafruit-blinka ******************"$'\n'
wget https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/master/raspi-blinka.py
sudo pip install adafruit-python-shell
echo n | sudo python3 raspi-blinka.py

TMP=$(grep -i 'dtoverlay=imx708,cam1' /boot/config.txt)
if [[ -z $TMP ]]
then

        echo $'\n'" ****************** Activating Camera ******************"$'\n'
        sudo bash -c 'echo "dtoverlay=imx708,cam1" >> /boot/config.txt'
fi


echo $'\n'" ****************** Finished installation, REBOOT necessary! ******************"$'\n'