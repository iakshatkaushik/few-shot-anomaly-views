"""Download all backbone weights into the Hugging Face cache (retries on rate limits)."""
import time
from huggingface_hub import hf_hub_download
REPOS = [("timm/vit_small_patch14_dinov2.lvd142m", "model.safetensors"),
         ("timm/vit_base_patch14_dinov2.lvd142m", "model.safetensors"),
         ("timm/vit_giant_patch14_reg4_dinov2.lvd142m", "model.safetensors"),
         ("timm/vit_base_patch16_dinov3.lvd1689m", "model.safetensors"),
         ("timm/vit_base_patch16_plus_clip_240.laion400m_e31", "open_clip_model.safetensors"),
         ("timm/wide_resnet50_2.tv_in1k", "model.safetensors")]
for repo, f in REPOS:
    for a in range(200):
        try:
            for name in (f, "config.json"):
                try:
                    hf_hub_download(repo, name)
                except Exception as e:
                    if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                        continue
                    raise
            print("ok", repo, flush=True); break
        except Exception as e:
            print("retry", repo, str(e)[:80], flush=True); time.sleep(45)
