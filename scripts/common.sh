#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

NC='\033[0m' # No Color

function echo_info() {
  echo -e "${GREEN}$1${NC}"
}

function echo_warning (){
  echo -e "${YELLOW}$1${NC}"
}

function echo_error() {
  echo -e "${RED}$1"
}

function die() {
  echo_error "$1${NC}"
  exit 1
}

function file_exist() {
  [[ -f "$1" ]]
}

function directory_exist() {
  [[ -d "$1" ]]
}

function die_on_file_not_found() {
  die "file does not exist: $1"
}

function die_on_directory_not_found() {
  die "directory does not exist: $1"
}

function file_extension() {
  echo "${1##*.}"
}

function file_name() {
  local filename="$(basename $1)"
  echo ${filename%.*}
}

function suffix_filename() {
  echo "$(dirname $1)/$(file_name $1)$2.$(file_extension $1)"
}
