#!/bin/bash

pausetime=5
for i in {1..10}
do
	echo "Launching sim $i pausing for $pausetime s"
	python hub.py --brain $1 --model="gb" & 
	sleep $pausetime
done 	