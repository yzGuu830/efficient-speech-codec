# Efficient Speech Coding with Cross-Scale Residual Vector Quantized Transformers

[arXiv] This is the code repository for the ESC codec presented in the [ESC: Efficient Speech Coding with Cross-Scale Residual Vector Quantized Transformers](https://arxiv.org/abs/2404.19441) paper. 
- Our neural speech codec, within only 30MB, can compress 16kHz speech to 1.5, 3, 4.5, 6, 7.5 and 9kbps more efficiently while maintaining comparative reconstruction quality to Descript's audio codec. 
- We provide [Model Checkpoints](#model-checkpoints) for different ESC variants and DAC models, along with a [Demo Page](https://western-spatula-93a.notion.site/Efficient-Speech-Codec-0e513f33cf104f799e16bcad015b03ef?pvs=4) for multilingual speech audios. 

![An illustration of ESC Architecture](assets/architecture.png)
## Usage

### Model Checkpoints

| Codec  | Checkpoint                                      | #Param. |
|--------|-------------------------------------------------|----------|
| ESC-Base           | [Download](https://drive.google.com/file/d/1OF1ab3az6nKOY8owSUhUH0ksYHFmR1bc/view?usp=sharing) | 8.39M    |
| ESC-Base(adv)      | [Download](https://drive.google.com/file/d/1_g1dFYhY7qXKWkcq8_Q6I-kv8tQW_SF7/view?usp=sharing) | 8.39M    |
| ESC-Large          | [Download](https://drive.google.com/file/d/180Q4zctqeNnDmRvoMsVQ-3iCB5FriJbN/view?usp=sharing) | 15.58M   |
| DAC-Tiny(adv)      | [Download](https://drive.google.com/file/d/1ED-B_S7ftsb8CqoFGTNkWUIrMKrk-iiu/view?usp=sharing) | 8.17M    |
| DAC-Tiny           | [Download](https://drive.google.com/file/d/1jk8zPYBYmxgsiSzrgoQynF6hnzoiIuX8/view?usp=sharing) | 8.17M    |
| DAC-Base(adv)      | [Download](https://drive.google.com/file/d/1moy0FX-aPlx54MajBRuE-zjYeNlJUjI6/view?usp=sharing) | 74.31M   |


### Install Dev Dependencies
```bash
pip install -r requirements.txt
```

### To compress and decompress audio
```ruby
python -m scripts.compress  --input /path/to/input.wav --save_path /path/to/output --model_path /path/to/model --num_streams 6 --device cpu 
```
This will create `.pth` and `.wav` files (code and reconstructed audio) under `save_path`. Our codec supports `num_streams` from 1 to 6, corresponding to 1.5 ~ 9.0kbps bitrates. 

```python
import torchaudio, torch
from esc import ESC
model = ESC(**config)
model.load_state_dict(torch.load("model.pth", map_location="cpu"),)
x, _ = torchaudio.load("input.wav")
# Encoding. (@ num_streams*1.5 kbps)
codes, f_shape = model.encode(x, num_streams=6)
# Decoding.
recon_x = model.decode(codes, f_shape)
```
This is the programmatic usage of `esc` to compress audio tensors using `torchaudio`. For more details see the `example.ipynb` notebook.

### Training

We provide our developmental training and evaluation [dataset](https://huggingface.co/datasets/Tracygu/dnscustom/tree/main) on huggingface.
```ruby
accelerate launch main.py --exp_name esc9kbps --config_path ./configs/9kbps_esc_base.yaml --wandb_project efficient-speech-codec --lr 1.0e-4 --num_epochs 80 --num_pretraining_epochs 15 --num_devices 4 --dropout_rate 0.75 --save_path /path/to/output --seed 53
```
We use `accelerate` library to handle distributed training. Logging is processed by `wandb` library. With 4 NVIDIA RTX4090 GPUs, training an ESC codec requires ~12h for 250k training steps on 180k 3-second speech clips with a batch size of 36. For detailed configurations, refer to `./configs/` folder. 

### Evaluation

```ruby
python -m scripts.test --eval_folder_path path/to/data --batch_size 12 --model_path /path/to/model --device cuda
```
This will run codec evaluation at all bandwidth on a test set folder. We provide four metrics for reporting: `PESQ`, `Mel Distance`, `SI-SDR` and `Bitrate Utilization Rate`. The evaluation statistics will be saved into `model_path` by default.  

## Results

![Performance Evaluation](assets/results.png)
We provide a comprehensive performance comparison of ESC with Descript's audio codec (DAC) at different scales of model sizes (w/ and w/o adversarial trainings).