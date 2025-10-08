serve_model() {
    local MODEL=$1
    local SERVED_MODEL_NAME=$2
    local TP=$3
    local LOG_DIR=$4
    mkdir -p ${LOG_DIR}

    VLLM_PATH=/cpfs03/data/shared/Group-m6/zhangjiajun/envs/vllm_infer
    export LD_LIBRARY_PATH=${VLLM_PATH}/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    echo "Starting serving ${MODEL} as ${SERVED_MODEL_NAME}..."
    ${VLLM_PATH}/bin/vllm serve ${MODEL} \
        --host 0.0.0.0 \
        --port 8000 \
        --task embed \
        --served-model-name ${SERVED_MODEL_NAME} \
        --tensor-parallel-size ${TP} \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --distributed-executor-backend ray \
        --trust-remote-code \
        --dtype auto \
        --api-key token-abc123 \

    VLLM_SERVER_PID=$!
    echo "VLLM Server PID: ${VLLM_SERVER_PID}"
    echo ${VLLM_SERVER_PID} > ${LOG_DIR}/pid.txt
    sleep 5

    # echo "Waiting for the model to be served..."
    # while true; do
    #     if grep -q 'Uvicorn running on' "${LOG_DIR}/vllm-server.log"; then
    #         echo "Model ${SERVED_MODEL_NAME} is being served..."
    #         break
    #     else
    #         echo "Waiting for model to start..."
    #         sleep 5
    #     fi
    # done
    echo "Waiting for the model to be served..."
    while true; do
        if grep -q 'Uvicorn running on' "${LOG_DIR}/vllm-server.log"; then
            echo "Model ${SERVED_MODEL_NAME} is now being served..."
            echo "Monitoring service status..."
            break
        else
            echo "Waiting for model to start..."
            sleep 5
        fi
    done

    # 持续监控服务状态
    while true; do
        if kill -0 $VLLM_SERVER_PID 2>/dev/null; then
            echo "Service is running... (PID: $VLLM_SERVER_PID)" 
            sleep 60  # 每60秒检查一次
        else
            echo "Service has stopped unexpectedly!"
            exit 1
        fi
    done
}



kill_vllm_server() {
    local LOG_DIR=$1
    VLLM_SERVER_PID=$(cat ${LOG_DIR}/pid.txt)
    echo "Killing VLLM Server with PID: ${VLLM_SERVER_PID}"
    kill ${VLLM_SERVER_PID}
}