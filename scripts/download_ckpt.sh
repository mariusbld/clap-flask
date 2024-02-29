#!/bin/bash

mkdir -p ckpt
ckpt_filename=music_audioset_epoch_15_esc_90.14.pt
ckpt_path=ckpt/$ckpt_filename
ckpt_url=https://huggingface.co/lukewys/laion_clap/resolve/main/$ckpt_filename?download=true
curl -L $ckpt_url > $ckpt_path
