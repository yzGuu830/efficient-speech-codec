from models.autoencoder import Swin_Audio_Codec
from utils import show_and_save, show_and_save_multiscale
import config as cfg

import torchaudio
import argparse
import torch
import json

class AudioCodec:
    def __init__(self, device) -> None:
        self.device = device
    
    def from_pretrain(self, path=f'{cfg.save_path}/DNS_CHALLENGE_Audio_Codec_18.0kbps_scalable_PatchSize3_2'):

        with open(f"{path}/config.json", 'r') as cfg:
            config = json.load(cfg)
            cfg.close()

        self.hparams = argparse.Namespace(**config)
        self.codec = Swin_Audio_Codec(self.hparams).to(self.device)

        weights = torch.load(f"{path}/best.pt",map_location=self.device)['model_state_dict']
        self.codec.load_state_dict(weights)
        print(f"Pretrained Model {path.split('/')[-1]} Loaded")
        self.codec.eval()

    def make_inference(self, source="test/instance1.flac", output_dir=f"test", num_stream=6):
        raw_audio, _ = torchaudio.load(source)
        kbps = num_stream * 3.0

        with torch.no_grad():
            raw_feat = torch.view_as_real(self.codec.ft(raw_audio))
            code, _ = self.codec.compress(raw_audio, num_stream, raw_feat)
            recon_audio, recon_feat = self.codec.decompress(code, num_stream, audio_len=10)

        show_and_save(raw_audio[0], raw_feat, recon_audio[0], recon_feat, path=f"{output_dir}/instance1_{kbps}kbps.jpg")
        return code

    def make_multiscale_inference(self, source="test/instance1.flac", output_dir=f"test"):

        raw_audio, _ = torchaudio.load(source)
        raw_feat = torch.view_as_real(self.codec.ft(raw_audio))

        codes, recon_audios, recon_feats = [], [], []
        for b in range(1, self.hparams.model_depth+2):
            code, _ = self.codec.compress(raw_audio, b, raw_feat)
            codes.append(code)
            recon_audio, recon_feat = self.codec.decompress(code, b, audio_len=10)
            recon_audios.append(recon_audio[0])
            recon_feats.append(recon_feat[0])

        show_and_save_multiscale(raw_audio[0], raw_feat[0], recon_audios, recon_feats, path=f"{output_dir}/instance1_multiscale.jpg")
        return codes

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="/scratch/eys9/output/DNS_CHALLENGE_Audio_Codec_18.0kbps_scalable_PatchSize3_2")
    parser.add_argument("--source", type=str, default="test/instance1.flac")
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--kbps", type=int, default=18)
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    Model = AudioCodec(args.device)
    Model.from_pretrain(args.model_path)

    if args.multiscale:
        print("Compress Audio at Multiple Scales")
        code = Model.make_multiscale_inference(args.source, args.output_dir)
        print(f"Result is stored at {args.output_dir}/**_multiscale.jpg")
    
    else:
        print(f"Compress Audio at {args.kbps}kbps")
        codes = Model.make_inference(args.source, args.output_dir, num_stream=args.kbps//3)
        print(f"Result is stored at {args.output_dir}/**_{args.kbps}kbps.jpg")

    return

if __name__ == "__main__":
    main()
