from esc.models import make_model
from .utils import read_yaml
import torch, os, torchaudio, argparse, warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input 16kHz mono audio file to encode")
    parser.add_argument("--save_path", type=str, default="./output", help="folder to save codes and reconstructed audio")

    parser.add_argument("--model_path", type=str, required=True, help="folder contains model configuration and checkpoint")
    parser.add_argument("--num_streams", type=int, default=6, help="number of transmitted streams in encoding")

    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

def main(args):
    
    x, sr = torchaudio.load(f"{args.input}")
    x = x.to(args.device)

    model = make_model(read_yaml(f"{args.model_path}/config.yaml")['model'])
    model.load_state_dict(
        torch.load(f"{args.model_path}/model.pth", map_location="cpu")["model_state_dict"],
    )
    model = model.to(args.device)

    codes, size = model.encode(x, num_streams=args.num_streams)
    recon_x = model.decode(codes, size)

    fname = args.input.split("/")[-1]
    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)
    torchaudio.save(f"{args.save_path}/decoded_{args.num_streams*1.5}kbps_{fname}", recon_x, sr)
    torch.save(codes, f"{args.save_path}/encoded_{args.num_streams*1.5}kbps_{fname.split('.')[0]}.pth")
    print(f"compression outputs saved into {args.save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
python -m scripts.compress \
    --input ./audio.wav \
    --save_path ./output \
    --model_path ./esc9kbps \
    --num_streams 6 \
    --device cpu 

"""