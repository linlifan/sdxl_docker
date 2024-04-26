PROMPT="A panda listening to music with headphones. highly detailed, 8k"
PROMPT="A blue robot is fighting with a red robot. highly detailed, 8k"

numactl -C 60-89 python sd_pipe_sdxl_lightning.py --height 832 --width 1152 --repeat 8 --bf16 --step 4 --prompt "$PROMPT"
