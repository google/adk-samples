#!/bin/bash

set -x

source agents/code-reviewer/.env

name_vars(){
    export AGENT_PATH="./code_reviewer"
    export SERVICE_NAME="code-reviewer"
    export APP_NAME="code-reviewer-app"
}

deploy_service(){
    adk deploy cloud_run \
    --project=$GOOGLE_CLOUD_PROJECT \
    --region=$GOOGLE_CLOUD_LOCATION \
    --service_name=$SERVICE_NAME \
    --app_name=$APP_NAME \
    --with_ui \
    $AGENT_PATH
}

main(){
    name_vars
    deploy_service
}

main

exit 0