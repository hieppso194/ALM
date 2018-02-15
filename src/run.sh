#!/bin/bash
case $1 in
	repo)
		echo "Source code is available in: https://github.com/hieppso194/ALM.git"
		;;
	help)
		echo "To use the first approach: ./run.sh 1"
		echo "To use the second approach/; ./run.sh 2"
		echo "Hint: the first use Python2, the second one use Python3"
		;;
	1)
		python lr.py
		;;
	2)
		python3 hmm.py
		;;
	*)
esac

