"""Persistent OpenVLA inference inside the TY1100 vendor VLA container."""
import argparse
import io
import os
from pathlib import Path

from .contract import ACTION_SPEC, action7, image_bytes, instruction, vector
from .rpc import serve


class OpenVLA:
    def __init__(self, model_dir):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        directory = str(Path(model_dir).resolve(strict=True))
        if not torch.cuda.is_available():
            raise RuntimeError("COREX device unavailable. Check ixsmi inside the vendor container.")
        opts = dict(local_files_only=True, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(directory, **opts)
        self.dtype = torch.bfloat16
        self.device = "cuda:0"
        # Place individual tensors on COREX while reading the shards. A complete
        # CPU copy would compete with the resident VLM for board system memory.
        self.policy = AutoModelForVision2Seq.from_pretrained(
            directory, attn_implementation="flash_attention_2", torch_dtype=self.dtype,
            low_cpu_mem_usage=True, device_map={"": self.device}, **opts
        ).eval()
        stats = self.policy.norm_stats.get("bridge_orig", {}).get("action", {})
        if len(stats.get("q01", [])) != 7 or len(stats.get("q99", [])) != 7:
            raise ValueError("Expected seven-dimensional bridge_orig action statistics")
        self.torch = torch

    def health(self):
        return {"model": "openvla-7b", "device": self.device, "action_spec": ACTION_SPEC}

    def predict(self, obs, options=None):
        from PIL import Image

        subtask = instruction(obs["task_descriptions"])
        with Image.open(io.BytesIO(image_bytes(obs["image"]))) as image:
            if image.width * image.height > 8_000_000:
                raise ValueError("Camera resolution exceeds input limit")
            rgb = image.convert("RGB")
        # The checkpoint expects token 29871 after Out:. Include its space in
        # tokenization so input_ids and attention_mask always have equal length.
        prompt = "In: What action should the robot take to {}?\nOut: ".format(subtask.rstrip("?.").lower())
        with self.torch.inference_mode():
            batch = self.processor(prompt, rgb).to(self.device, dtype=self.dtype)
            step = self.policy.predict_action(**batch, unnorm_key="bridge_orig", do_sample=False)
        action = vector(step.tolist(), 7, "OpenVLA prediction")
        # bridge_orig masks gripper out of quantile unnormalization. Its token
        # centers can be negative; the Bridge physical open fraction starts at 0.
        if not -1 <= action[6] <= 1:
            raise ValueError("Gripper prediction outside the tokenizer range")
        action[6] = max(0.0, action[6])
        return {"actions": [action7(action)], "action_spec": ACTION_SPEC, "obs_id": obs["id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()
    policy = OpenVLA(args.model)
    serve(args.port, {"healthz": policy.health, "vla.predict": policy.predict})


if __name__ == "__main__":
    main()
