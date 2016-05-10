#!/bin/bash

# Parse options
while getopts ":i:h:p:" OPT; do
	case $OPT in
		i ) INPUT_PATH=($OPTARG) ;;

		h ) DATA_HOME=($OPTARG) ;;

		p ) PATTERN=($OPTARG) ;;
	esac
done

if [ -z $DATA_HOME ]; then
	DATA_HOME="~/.wort_data"
fi

find $INPUT_PATH -name "*$PATTERN*" | while read line; do
	INPUT_FILE=$(basename $line)
	echo "Running intrinsic evaluation with model=$INPUT_FILE..."
	python -m wort.evaluation -ip $INPUT_PATH -i $INPUT_FILE -h $DATA_HOME
	echo "==============================================================================="
done