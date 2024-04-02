ENV_PATH=/root/miniconda3/envs/ipex_dev
#
export MKL_VERBOSE=0
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,1,0
#
export LD_PRELOAD=$ENV_PATH/lib/libiomp5.so
export LD_PRELOAD=$ENV_PATH/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

#export PYTORCH_TENSOREXPR=0

PROMPT="A panda listening to music with headphones. highly detailed, 8k"
PROMPT="A blue robot is fighting with a red robot. highly detailed, 8k"

# SDXL-
#numactl --localalloc --physcpubind=48-95 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 100 --bf16
#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 4 --step 4 --bf16 --prompt "$PROMPT"
#numactl --localalloc --physcpubind=48-95 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --step 4 --bf16 --prompt "$PROMPT"
#numactl --localalloc --physcpubind=48-95 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --step 4 --bf16 --prompt "$PROMPT"
#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 4 --bf16 --step 4 --prompt "$PROMPT"
#numactl --localalloc --physcpubind=48-95 python sd_pipe_sdxl_lightning.py --height 896 --width 896  --repeat 101 --bf16 --step 4
#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 2 --step 4
#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 2 --bf16 --step 4
#numactl --localalloc --physcpubind=48-95 python sd_pipe_sdxl_lightning.py --height 1400 --width 800  --repeat 101 --bf16 --step 4
#numactl --localalloc --physcpubind=48-95 python sd_pipe_sdxl_lightning.py --height 1920 --width 992  --repeat 101 --bf16 --step 4


#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT"
#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --bf16 --step 4 --prompt "$PROMPT"

#numactl --localalloc --physcpubind=48-71 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT"
#numactl --localalloc --physcpubind=48-71 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --bf16 --step 4 --prompt "$PROMPT"


#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" &
#numactl --localalloc --physcpubind=64-79 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" &
#numactl --localalloc --physcpubind=80-95 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT"

#numactl --localalloc --physcpubind=48-63 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" &
#numactl --localalloc --physcpubind=64-79 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" &
#numactl --localalloc --physcpubind=80-95 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --bf16 --step 4 --prompt "$PROMPT"

#numactl --localalloc --physcpubind=48-71 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" &
#numactl --localalloc --physcpubind=72-95 python sd_pipe_sdxl_lightning.py --height 512 --width 512  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" 

#numactl --localalloc --physcpubind=48-71 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 11 --bf16 --step 4 --prompt "$PROMPT" &
#numactl --localalloc --physcpubind=72-95 python sd_pipe_sdxl_lightning.py --height 1024 --width 1024  --repeat 40 --bf16 --step 4 --prompt "$PROMPT"

numactl -C 48-95 python sd_pipe_sdxl_lightning.py --height 832 --width 1152 --repeat 4 --bf16 --step 4 --prompt "$PROMPT"
