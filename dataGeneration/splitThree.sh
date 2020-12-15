#!/bin/bash
mkdir testenemy
mkdir testnone
mkdir validenemy
mkdir validnone

echo "Building split file list"

#ls -rt -d -1 dataset2/enemy/*
ENEMY=`ls -rt -d -1 enemy/*`
NONE=`ls -rt -d -1 none/*`
TOTALNONE=`echo "$NONE" | wc -l`
TOTALENEMY=`echo "$ENEMY" | wc -l`

TEST="test"
FOLDERS=( "test" "valid" )

for TYPES in "${FOLDERS[@]}"
do
	if [ "$TYPES" = "test" ]
	then
		SEGMENT=$1
	else
		SEGMENT=$2
	fi
	RANENEMY=`echo "$ENEMY" | shuf -n $(expr $TOTALENEMY / 100 \* $SEGMENT)`
	RANNONE=`echo "$NONE" | shuf -n $(expr $TOTALNONE / 100 \* $SEGMENT)`

	i=1

	for LINE in $RANENEMY
	do
		echo "Moving random ENEMY image $i to ${TYPES}"
		mv "${LINE}" ${TYPES}enemy/
		((i++))
	done

	i=1

	for LINE2 in $RANNONE
	do
		echo "Moving random NONE image $i to ${TYPES}"
		mv "${LINE2}" ${TYPES}none/
		((i++))
	done
	ENEMY=`ls -rt -d -1 enemy/*`
	NONE=`ls -rt -d -1 none/*`
done

echo "Done"
