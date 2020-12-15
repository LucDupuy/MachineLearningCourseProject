#!/bin/bash
mkdir testenemy
mkdir testnone

echo "Building split file list"

#ls -rt -d -1 dataset2/enemy/*
ENEMY=`ls -rt -d -1 enemy/*`
NONE=`ls -rt -d -1 none/*`


RANENEMY=`echo "$ENEMY" | shuf -n $(expr $(echo "$ENEMY" | wc -l) / 100 \* $1)`
RANNONE=`echo "$NONE" | shuf -n $(expr $(echo "$NONE" | wc -l) / 100 \* $1)`

i=1

for LINE in $RANENEMY
do
	echo "Moving random ENEMY image $i"
	mv "${LINE}" testenemy/
	((i++))
done

i=1

for LINE2 in $RANNONE
do
	echo "Moving random NONE image $i"
	mv "${LINE2}" testnone/
	((i++))
done

echo "Done"
