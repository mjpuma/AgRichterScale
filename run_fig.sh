#!/bin/bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=$(pwd)/mpl_cache
mkdir -p $MPLCONFIGDIR
python3 "$@"

