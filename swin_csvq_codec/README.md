# cross-swin-spec-codec

### Training

Use the following script to train a swin transformer-based multiscale audio codec.

```{python}
python train_audio_codec.py \
        --scalable \
        --data_name "DNS_CHALLENGE" \
        --num_streams 6 \
        --patch_size (3,2) \
        --lr 1.0e-4 \
        --train_bs 72 \
        --test_bs 36 \
        --epochs 50 \
        --plot_interval .66 \
        --num_workers 4
```

If you want to train a non-scalable version codec, delete \scalable and specify \numstreams

### Inference

Use the following script to make a compression inference.

```{python}
python eval/test_codec.py \
        --model_path output/model \
        --device "cuda" \
        --source test/instance1.flac \
        --multiscale \
        --kbps 18 \
        --output_dir test
```

You can specify a fixed bitrate by removing \multiscale and set \kbps

### Results 
