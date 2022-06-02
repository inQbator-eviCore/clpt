#!/bin/bash

CONDA_ENVIRONMENT_NAME="clpt"
CONDA_ENVIRONMENT_FILE="conda_requirements.yml"

function printhelp() {
    echo "Usage: ./app <command>"
    echo "Available commands:"
    echo "   (s)etup-environment:         setup the environment"
    echo "   (u)nit-(t)est:               run unit tests on project"
    echo "   (l)int-(t)est                run lint (flake8) tests"
    echo "   (e)xport:                    export dependencies"
    echo "   (g)enerate-(d)ocumentation:  generate sphinx documentation for the project"
    echo "   (s)tatic-(a)nalysis          run static analysis by calling Flake8"
    echo "   (c)overage                   review lines covered by backend unit tests"
    echo "   (d)elete-environment:        delete environment"
}

function createCondaEnv() {
  conda env create -q -n $CONDA_ENVIRONMENT_NAME -f $CONDA_ENVIRONMENT_FILE $FORCE
}

function activateCondaEnv() {
  if [[ $1 == '--force' || $1 == '-f' ]]; then
    shift
    FORCE='--force'
  fi

  eval "$(conda shell.bash hook)"

  # Find all existing environments (adding spaces around the name to ensure precise matching)
  ENVS=$(conda env list | awk '$1{print " "$1" "}' )

  # Search if CONDA_ENVIRONMENT_NAME exists and force is not set
  if [[ $ENVS == *" $CONDA_ENVIRONMENT_NAME "* && -z "$FORCE" ]]; then
    conda activate $CONDA_ENVIRONMENT_NAME
  else
    echo "Creating conda environment $CONDA_ENVIRONMENT_NAME"
    createCondaEnv
    conda activate $CONDA_ENVIRONMENT_NAME
  fi;

  echo "Activated conda environment $CONDA_ENVIRONMENT_NAME"
}

function deactivateCondaEnv() {
  eval "$(conda shell.bash hook)"

  # Deactivate conda env
  conda deactivate
  echo "Deactivated conda environment $CONDA_ENVIRONMENT_NAME"
}

COMMAND=$1

if [ -z $COMMAND ]; then
  printhelp
fi

shift

if [[ $COMMAND == 's' || $COMMAND == 'setup-environment' ]]; then
    activateCondaEnv --force
    python setup.py develop

elif [[ $COMMAND == 'ut' || $COMMAND == 'unit-test' ]]; then
    activateCondaEnv $@
    echo 'Running unit tests'
    python -m pytest .

elif [[ $COMMAND == 'lt' || $COMMAND == 'lint-test' ]]; then
    activateCondaEnv $@
    echo 'Running lint tests'
    flake8 --tee --output-file=lint-testresults.xml

elif [[ $COMMAND == 'e' || $COMMAND == 'export' ]]; then
  activateCondaEnv $@
  conda env export -n $CONDA_ENVIRONMENT_NAME -f $CONDA_ENVIRONMENT_FILE
  echo -e "Dependencies Exported"

elif [[ $COMMAND == 'gd' || $COMMAND == 'generate-documentation' ]]; then
  activateCondaEnv $@
  export RST_BUILD_DIR=src/docs
  export VERSION=$(grep '__version__' src/__init__.py | awk -F ' ' '{print $3}')
  export PROJECT_NAME=clpt
  export HTTP_HOSTING_PATH=/var/www/projects/html/docs/${PROJECT_NAME}
  export DOC_BUILD_DIR=src/latest
  export WEB_DOC_PATH=docs
  export TMP_DOC_DIR=transient
  export CODE_DIR=src
  sphinx-apidoc -o ${RST_BUILD_DIR} -f -d 2 -F -e --implicit-namespaces -M -V ${VERSION} -H ${PROJECT_NAME} -a -t ${RST_BUILD_DIR}/_templates ${CODE_DIR}
  sphinx-build -b html ${RST_BUILD_DIR} ${DOC_BUILD_DIR}

elif [[ $COMMAND == 'd' || $COMMAND == 'delete-environment' ]]; then
  deactivateCondaEnv
  conda env remove -y -n $CONDA_ENVIRONMENT_NAME

elif [[ $COMMAND == 'c' || $COMMAND == 'coverage' ]]; then
  activateCondaEnv $@
  coverage run -m nose -v
  coverage report -m
  coverage xml -o coverage.xml

elif [[ $COMMAND == 'sa' || $COMMAND == 'static-analysis' ]]; then
  activateCondaEnv $@
  flake8

else
  echo "Unknown command: $COMMAND"
  printhelp
fi