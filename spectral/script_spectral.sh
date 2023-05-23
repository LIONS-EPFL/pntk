#!/bin/bash
resultdir=result
for degree in 3 6 9
do
	echo "degree $degree"
	python main.py --degree $degree --resultdir $resultdir
done