#!/bin/bash
INLIST=inlist_muHer
change() {
	param=$1
	newval=$2
	filename=$3
	escapedParam=$(sed 's#[^^]#[&]#g; s#\^#\\^#g' <<< "$param")
	search="^\s*\!*\s*$escapedParam\s*=.+$"
	replace="	$param = $newval"
	if [ ! "$filename" == "" ]; then
		sed -r -i.bak -e "s#$search#$replace#g" $filename
	fi
	if [ ! "$filename" == "$INLIST" ]; then
		change $param $newval "$INLIST"
	fi
}

change new_y $1
change new_z $2
change Zbase $3
change mixing_length_alpha $4
change initial_mass $5
change log_directory "'$6'"
change special_rate_factor $7
#change overshoot_f $8
#change max_num_profile_models $8