from models.codec import SwinAudioCodec
import dac, torch, os, yaml, torchaudio, glob, argparse, shutil
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, default_collate
from utils import manage_checkpoint, dict2namespace

dacs_configs = {
    "dac": {"enc_dim": 64, "dec_dim": 1536},
    "dac_tiny": {"enc_dim": 32, "dec_dim": 288},
    "dac_tiny_no_adv": {"enc_dim": 32, "dec_dim": 288}
}
ours_configs = {
    "swin9k": "residual_9k.yml",
    "swin9kgan": "residual_9k_gan.yml",
    "ours_final": "residual_9k_kmeans.yml",
    "ours_final_gan": "residual_9k_kmeans_gan.yml"
}

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pth", type=str, default="../data/DNS_CHALLENGE/processed_wav/test/1-1158")
    parser.add_argument("--save_pth", type=str, default="../compare")

    parser.add_argument("--ours_model_pth", type=str, default="/root/autodl-fs/swin-9k-residual-gan-ADAP/refine/best.pt")
    parser.add_argument("--dacs_model_pth", type=str, default="../dac_output/dac16khz9k/best/dac/weights.pth")

    parser.add_argument("--ours_model_tag", type=str, default=None) # swin9kgan swin9k
    parser.add_argument("--dacs_model_tag", type=str, default=None) # dac dac_tiny

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

def load_dac(dacs_model_pth, dacs_model_tag):
    dac_model = dac.DAC(
        encoder_dim=dacs_configs[dacs_model_tag]["enc_dim"],
        encoder_rates=[2,4,5,8],
        decoder_dim=dacs_configs[dacs_model_tag]["dec_dim"],
        decoder_rates=[8,5,4,2],
        sample_rate=16000,
        n_codebooks=18,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,)
    ckp = torch.load(dacs_model_pth, map_location="cpu")["state_dict"]
    dac_model.load_state_dict(ckp, strict=False)
    print(f"{dacs_model_tag} reproduced pretrained ckp loaded.")
    return dac_model

def load_ours(ours_model_pth, ours_model_tag):
    with open(os.path.join('./configs', ours_configs[ours_model_tag]), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    ours = SwinAudioCodec(**vars(config.model))
    ckp = torch.load(ours_model_pth, map_location="cpu")
    ours.load_state_dict(manage_checkpoint(ckp))
    print(f"ours {ours_model_tag} pretrained ckp loaded.")
    return ours

class EvalSet(Dataset):
    def __init__(self, test_file_path) -> None:
        super().__init__()
        self.testset_files = glob.glob(f"{test_file_path}/*.wav")
        
    def __len__(self):
        return len(self.testset_files)

    def __getitem__(self, i):
        x, _ = torchaudio.load(self.testset_files[i])
        clip_id = self.testset_files[i].split("/")[-1] #clip_N.wav#
    
        return {"audio": x[:, :-80], "clip": clip_id}
    
def collate_fn(batch):
    out = {"audio": [], "clip":[]}
    for b in batch:
        for k, v in b.items():
            if k == "audio":
                out[k].append(v)
            elif k == "clip":
                out[k].append(v)
    out["audio"] = torch.cat(out["audio"], dim=0) # [bs, T]
    return out

def dac_infer(model, dl, save_pth, device, dac_model_tag="dac"):
    model.eval()
    with torch.inference_mode():
        for n_q in [18, 15, 12, 9, 6, 3]: # 9k -> 1.5k
            os.makedirs(f"{save_pth}/{dac_model_tag}/{n_q*.5:.2f}kbps/", exist_ok=True)
            for i, input in tqdm(enumerate(dl), total=len(dl), desc=f"Saving DAC-16kHz {n_q*.5:.2f}kbps outputs into {save_pth}/{dac_model_tag}/{n_q*.5:.2f}kbps/"):
                x = input["audio"].unsqueeze(1).to(device)
                x_process = model.preprocess(x, sample_rate=16000)
                z, codes, latents, _, _ = model.encode(x_process, n_quantizers=n_q)
                x_recon = model.decode(z)
                x_recon = x_recon[:, :, :-72] 

                for j in range(x.size(0)):
                    clip_id = input["clip"][j]
                    torchaudio.save(
                        f"{save_pth}/{dac_model_tag}/{n_q*.5:.2f}kbps/recon_{clip_id}", x_recon[j].cpu(), 16000
                    ) 

def ours_infer(model, dl, save_pth, device, ours_model_tag="swin9kgan"):
    model.eval()
    with torch.inference_mode():
        for s in range(1, 7):
            os.makedirs(f"{save_pth}/{ours_model_tag}/{s*1.5:.2f}kbps/", exist_ok=True)
            for i, input in tqdm(enumerate(dl), total=len(dl), desc=f"Saving SwinCodec {s*1.5:.2f}kbps outputs into {save_pth}/{ours_model_tag}/{s*1.5:.2f}kbps/"):
                x = input["audio"].to(device)
                outputs = model(**dict(x=x, x_feat=None, streams=s, train=False))
                
                for j in range(x.size(0)):
                    clip_id = input["clip"][j]
                    torchaudio.save(
                        f"{save_pth}/{ours_model_tag}/{s*1.5:.2f}kbps/recon_{clip_id}", 
                        outputs['recon_audio'][j].unsqueeze(0).cpu(), 16000
                    ) 

def zip_files(root_folder, zip_folder):

    archived = shutil.make_archive(f"{root_folder}/{zip_folder}", 'zip', f"{root_folder}/{zip_folder}")

    if os.path.exists(f"{root_folder}/{zip_folder}.zip"):
        print(archived) 
    else: 
        print("zip file not created")

if __name__ == "__main__":

    args = make_args()

    dns_eval_set = EvalSet(test_file_path=args.data_pth)
    eval_loader = DataLoader(dns_eval_set, batch_size=12, shuffle=False, collate_fn=collate_fn)


    if args.ours_model_tag is not None:
    
        ours = load_ours(args.ours_model_pth, args.ours_model_tag).to(args.device)
        ours_infer(ours, eval_loader, args.save_pth, args.device, args.ours_model_tag)

        del ours
        print("Finish Inference, Zipping Folder...")
        zip_files(args.save_pth, args.ours_model_tag)

    if args.dacs_model_tag is not None:

        dac_model = load_dac(args.dacs_model_pth, args.dacs_model_tag).to(args.device)
        dac_infer(dac_model, eval_loader, args.save_pth, args.device, args.dacs_model_tag)

        del dac
        print("Finish Inference, Zipping Folder...")
        zip_files(args.save_pth, args.dacs_model_tag)


""" 
DACs
python make_instances.py \
    --data_pth ../data/DNS_CHALLENGE/processed_wav/test/1-1158 \
    --save_pth ../compare \
    --dacs_model_pth /root/autodl-fs/dac_output/dac16khz9k_tiny/best/dac/weights.pth \
    --dacs_model_tag dac_tiny 

python make_instances.py \
    --data_pth ../data/DNS_CHALLENGE/processed_wav/test/1-1158 \
    --save_pth ../compare \
    --dacs_model_pth /root/autodl-fs/dac_output/dac16khz9k_tiny_no_adv/best/dac/weights.pth \
    --dacs_model_tag dac_tiny_no_adv


Ours 
python make_instances.py \
    --data_pth ../data/DNS_CHALLENGE/processed_wav/test/1-1158 \
    --save_pth ../compare \
    --ours_model_pth ../output/ours-final-warmup-kmeans/finetune/best.pt \
    --ours_model_tag ours_final

python make_instances.py \
    --data_pth ../data/DNS_CHALLENGE/processed_wav/test/1-1158 \
    --save_pth ../compare \
    --ours_model_pth ../output/ours-final-warmup-kmeans-gan/finetune/best.pt \
    --ours_model_tag ours_final_gan
"""
