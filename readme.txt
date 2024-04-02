
step 0: download SDXL Turbo model from huggingface
        mkdir models
        pip install -U huggingface_hub hf_transfer 
        HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --resume-download stabilityai/sdxl-turbo  --local-dir sdxl-turbo

step 1: build docker image
       docker build --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy    -t sdxl:v1 .

step 2: docker run
       docker run --name sdxl --privileged=true  -v /models:/models -it sdxl:v1  bash

step 3: run test 
      cd sdxlturbo
      sh run_local.sh
      #control different core numbers
      numactl -C 48-59 python sd_pipe_sdxl_turbo.py --height 1024 --width 1024  --repeat 11 --step 4 --bf16       

