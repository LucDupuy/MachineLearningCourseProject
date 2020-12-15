# ML-Term-Assignment

Image Classification of CS:GO characters

## What is the directory structure?

### Machine Learning Algorithm

The code for our machine learning training and testing can be found in `/src/pytorch`

The code in `/src/Tensorflow Example` was just leftover from trying to get it to work before the TA advised us to use pytorch instead.

### Dataset Generation Scripts

These reside in `/dataGeneration/`

## How to setup and run the code

First you must change the following line numbers as they use hardcoded directories. You need to change them to match your own directories. The line number and files that need to be changed are as follows(within the pytorch directory as specified above):

- MakeSpreadsheet.py
	- 92
	- 104
	- 112
	- 125-130
- main.py
	- 254
	- 259
	- 270-275

Once you have these lines changes to match your own dataset, spreadsheet, and result directories and you have the datasets in the appropriate directory, run the MakeSpreadsheet.py file with python. Make sure to have the appropriate packages installed as specified at the top of the MakeSpreadsheet.py and CustomDataset.py files. Once this code finishes running your dataset will be labeled accordingly. Then you run main.py. Make sure you also have the appropriate packages as specified in the top of the main.py file. 

### The first time you run main.py

The first time you run this code it will most likely fail. You will see an error similar to the following:

![error](https://cdn.discordapp.com/attachments/753778167335878701/774795309234651176/unknown.png)

depending on the number you get in the error(in the image example above this number would be "2457000") you will need to place that number into the code. See line number 69 in main.py and change the line to the following:

`input_size = batch_sizes[i] * int(yourNumber / batch_sizes[i] / batch_sizes[i])`

where `yourNumber` is the specified number as explained above.

Now you can rerun main.py and it will train and test on your datasets accordingly. It will then save the results into a spreadsheed in your specified results folder.

