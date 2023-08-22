#!/bin/bash

cd ${PBS_O_WORKDIR:-""}
mksquashfs tempdata/000000* tempdata.sqsh 2>/dev/null
