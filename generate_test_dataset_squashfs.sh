#!/bin/bash

cd ${PBS_O_WORKDIR:-""}
mksquashfs tempdata tempdata.sqsh 2>/dev/null
