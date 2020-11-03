#!/bin/bash

<<COMMENT
this script will build a randomized subset of the dataset
place this script into where the "enemy" and "none" directories are
i.e dataset/rand.sh
where dataset contains "enemy" and "none" like so:
dataset/enemy
dataset/none
then run the script passing it a number from 1 to 100
depending on what percent of the dataset you would like to use
it will create 2 directories named "ranenemy" and "rannone"
where your randomized dataset resides
COMMENT


mkdir ranenemy
mkdir rannone

echo "Building randomized file list"

#ls -rt -d -1 dataset2/enemy/*
ENEMY=`ls -rt -d -1 enemy/*`
NONE=`ls -rt -d -1 none/*`


RANENEMY=`echo "$ENEMY" | shuf -n $(expr $(echo "$ENEMY" | wc -l) / 100 \* $1)`
RANNONE=`echo "$NONE" | shuf -n $(expr $(echo "$NONE" | wc -l) / 100 \* $1)`

i=1

for LINE in $RANENEMY
do
	echo "Copying random ENEMY image $i"
	cp "${LINE}" ranenemy/
	((i++))
done

i=1

for LINE2 in $RANNONE
do
	echo "Copying random NONE image $i"
	cp "${LINE2}" rannone/
	((i++))
done

echo "Done"
