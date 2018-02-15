#!/bin/bash
case $1 in
	repo)
		echo "Source code and data are available in: https://github.com/hieppso194/ALM.git"
		;;
	help)
		echo "To use the first approach: ./run.sh 1"
		echo "To use the second approach/; ./run.sh 2"
		echo "Hint: The first approach uses Python2, the second one uses Python3"
		;;
	1)
		python lr.py
		;;
	2)
		python3 hmm.py
		;;
	*)
esac

