#! /bin/bash
scp requirements.txt install_robot_control.sh test_aruco.py $1:.
ssh $1 ./install_robot_control.sh