export HF_ENDPOINT=https://hf-mirror.com
#export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --resume-download codellama/CodeLlama-13b-hf --local-dir your_path/Codellama-13B --local-dir-use-symlinks False --exclude *.bin
huggingface-cli download --resume-download codellama/CodeLlama-34b-hf --local-dir your_path/Codellama-34B --local-dir-use-symlinks False --exclude *.bin

