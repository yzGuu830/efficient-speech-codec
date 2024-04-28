# ESC: High-Fidelity Speech Coding with Efficient Cross-Scale Vector Quantized Transformers

This is the code repository for the Efficient Speech Codec presented in the paper [ESC: High-Fidelity Speech Coding with Efficient Cross-Scale Vector Quantized Transformers](https://drive.google.com/file/d/1QqqgoAb5qB8GJcD_IWiUepMsfkoLEdYS/view?usp=sharing). Our neural speech codec, within only 30MB, can compress 16kHz speech to 1.5, 3, 4.5, 6, 7.5 and 9kbps efficiently while maintaining high quality. We provide [model checkpoint](https://drive.google.com/file/d/157L22yu-bt_ARrsXYYGnEd6w-8saeUdV/view?usp=sharing) and [Demo Page]()

![An illustration of ESC Architecture](assets/architecture.png)

## Usage

### Install Dependencies
```{python}
pip install -r requirements.txt
```

### To compress and decompress audio
```{python}
python -m scripts.compress \
    --input /path/to/input.wav \
    --save_path /path/to/output \
    --model_path /path/to/model \
    --num_streams 6 \
    --device cpu 
```
This will create `.pth` and `.wav` files (codes and reconstructed audios) under `save_path`. Our codec supports `num_streams` from 1 to 6. 

```python
import torchaudio
from models import ESC

model = ESC(**config)
model.load_state_dict(
        torch.load("model.pth", map_location="cpu")["model_state_dict"],
    )
model = model.to("cuda")

x, _ = torchaudio.load("input.wav")
x.to("cuda")

# encode to codes
codes, pshape = model.encode(x, num_streams=6)
# decode to audios
recon_x = model.decode(codes, pshape)
```
This is the programmatic usage of esc to compress tensor audios. 

### Training

```{python}
accelerate launch main.py \
    --exp_name esc9kbps \
    --config_path ./configs/9kbps_final.yaml
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 80 \
    --num_pretraining_epochs 15 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path /path/to/output \
    --seed 53
```
We use `accelerate` library to handle distributed training. Logging is processed by `wandb` library. With 4 NVIDIA RTX4090 GPUs, training an ESC codec requires ~12h for 250k training steps on 180k 3-second audio clips with a batch size of 36.

### Evaluation

```{python}
python -m scripts.test \
    --eval_folder_path path/to/data \
    --batch_size 12 \
    --model_path /path/to/model \
    --device cuda
```
This will run codec evaluation at all bandwidth on a test set folder. We provide four metrics for reporting: `PESQ`, `Mel Distance`, `SI-SDR` and `Bitrate Utilization Rate`. The evaluation statistics will be saved into `model_path` by default.  


## Results

![Performance Evaluation](assets/results.png)
