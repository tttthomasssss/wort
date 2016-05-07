#!/bin/bash

# Parse options
while getopts ":p:v:" OPT; do
	case $OPT in
		p ) CACHE_PATH=($OPTARG) ;;

		v ) VERBOSE=($OPTARG) ;;
	esac
done

if [ -z $CACHE_PATH ]; then
	CACHE_PATH=$HOME/.wort_data/model_cache
fi

if [ -n $VERBOSE ]; then
	echo "Deleting cache from $CACHE_PATH..."
fi

rm -rf $CACHE_PATH/*

if [ -n $VERBOSE ]; then
	echo "Cache deleted!"
fi