
step 0: download SDXL/ SDXL lightning model from huggingface
        mkdir models
        pip install -U huggingface_hub hf_transfer 
        HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download  stabilityai/stable-diffusion-xl-base-1.0 --local-dir stable-diffusion-xl-base-1.0 --local-dir-use-symlinks False
        #or to reduce size
        #HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download  stabilityai/stable-diffusion-xl-base-1.0 --local-dir stable-diffusion-xl-base-1.0 --local-dir-use-symlinks False --include "*.json" "*.safetensors" "*.txt" 
        HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download  ByteDance/SDXL-Lightning --local-dir SDXL-Lightning --local-dir-use-symlinks False
        #or to reduce size
        #HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download  ByteDance/SDXL-Lightning --local-dir SDXL-Lightning --local-dir-use-symlinks False --include "sdxl_lightning_4step_unet.safetensors"

step 1: build docker image
       docker build --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy -t sdxl:lightning -f Dockerfile_sdxllightning .

step 2: docker run
       docker run --name sdxl_lightning --privileged=true  -v /home/models:/home/models -it sdxl:lightning  bash
       #if proxy needed
       #docker run --name sdxl_lightning --privileged=true -e https_proxy=http://proxy-dmz.intel.com:912/ -e http_proxy=http://proxy-dmz.intel.com:911/ --network host  -v /home/models:/home/models -it sdxl:lightning  bash

step 3: run test 
      cd sdxllightning
      sh run_local_docker.sh

