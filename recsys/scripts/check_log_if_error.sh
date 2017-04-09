#!/bin/bash
# Captures the error log and display it through "less"
log_file="$(eval "$*" | grep -oP "\S+log")"
if [[ -n "${log_file}" ]]; then
	less "${log_file}"
fi