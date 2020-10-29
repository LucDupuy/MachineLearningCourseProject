#!/bin/bash


<<COMMENT
Requirements: You must have ffmpeg installed

to run this code pass the following parameters: 

location of the video in the first parameter
path of the timestamp text file in the second parameter
directory where you would like the images to be saved into in the third parameter
If the directory doesnt exist it will create it.

e.g ./generateDataset.sh video.mp4 timestamp.txt result

This code assumes the first time stamp is a "non-enemy" set
and that the rest of the timestamps alternate.
The format used for timestamps on each line is something like the following:

00:12:34.800|00:43:21.700

the format being in hour:minute:second.millisecond

the video can be in any format and ffmpeg should handle it properly

The format of the output images is 3enemy0874.png

Where the first number is the "group" of images. All pictures with the
same first number and word will be continious.
The second number is the frame number in that group of images.

To check if the timestamps are correct all you need to do is check the
first and last frames of each group. If they have/dont have an enemy in
them then the rest of the frames should be correct as well.
COMMENT

i=0  #incrementor, used to number the images
mkdir $3
mkdir $3/none
mkdir $3/enemy

OUT=$(cat "$2")

for LINE in $OUT
do
	echo
	echo
	ARR=(${LINE//|/ })
	if [ $(($i % 2)) = 0 ]
	then #not enemy
		echo "Generating set "$(($i / 2))" of non-enemy images"
		echo "From: ${ARR[0]}"
		echo "To:   ${ARR[1]}"
		ffmpeg -loglevel warning -i $1 -ss "${ARR[0]}" -to "${ARR[1]}" "$3/none/$(($i / 2))none%04d.png"
	else #is enemy
		echo "Generating set "$(($i / 2))" of enemy images"
		echo "From: ${ARR[0]}"
		echo "To:   ${ARR[1]}"
		ffmpeg -loglevel warning -i $1 -ss "${ARR[0]}" -to "${ARR[1]}" "$3/enemy/$(($i / 2))enemy%04d.png"
	fi
	((i++))
done
echo "Completed generation, your images can be found in the '$3' directory"
