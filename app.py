from typing import Any, Dict, List, Optional
import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# exactly as in the original Omni example:
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

class InferlessPythonModel:
    def initialize(self):
        # just stash our model identity and processor
        self.model_name    = "Qwen/Qwen2.5-Omni-7B"
        self.max_model_len = 5632
        self.max_num_seqs  = 5
        self.processor     = AutoProcessor.from_pretrained(self.model_name)

    def infer(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        inputs may contain:
          - prompt: str
          - system_prompt: str (overrides default)
          - seed: int
          - temperature: float
          - max_tokens: int
          - use_audio_in_video: bool
          - contents: List[{
                "type": "text" | "image" | "video" | "audio",
                # text → "text": "<your text>"
                # image → "url": "<path_or_url_to_image>"
                # video → "url": "<path_or_url_to_video>", optional "num_frames", "fps", "sampling_rate"
                # audio → "name": "<asset_name_or_path>" 
            }]
        """
        # 1) System & user prompt
        system_prompt = inputs.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        user_text     = inputs.get("prompt", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": []},  # build below
        ]

        # 2) Sampling‐seed & params
        seed        = inputs.get("seed", None)
        temperature = float(inputs.get("temperature", 0.2))
        max_tokens  = int(inputs.get("max_tokens", 64))

        # 3) Gather all multimodal data
        mm_data: Dict[str, Any] = {}
        mm_processor_kwargs: Dict[str, Any] = {}
        use_audio_in_video = bool(inputs.get("use_audio_in_video", False))

        for item in inputs.get("contents", []):
            t = item["type"]
            if t == "text":
                messages[1]["content"].append({"type": "text", "text": item["text"]})

            elif t == "image":
                # load & queue a PIL image
                img = ImageAsset(item["url"]).pil_image.convert("RGB")
                mm_data.setdefault("image", []).append(img)

            elif t == "audio":
                # load via AudioAsset exactly like the demo
                asset = AudioAsset(item["name"])
                mm_data.setdefault("audio", []).append(asset.audio_and_sample_rate)

            elif t == "video":
                # possibly extract audio‐in‐video if requested
                num_frames     = int(item.get("num_frames", 16))
                fps            = int(item.get("fps", 1))
                sampling_rate  = int(item.get("sampling_rate", 16000))

                if use_audio_in_video:
                    # same check as original
                    assert not envs.VLLM_USE_V1, (
                        "V1 does not support use_audio_in_video. "
                        "Please set VLLM_USE_V1=0."
                    )
                    asset = VideoAsset(name=item["url"], num_frames=num_frames)
                    mm_data["video"] = asset.np_ndarrays
                    mm_data["audio"] = asset.get_audio(sampling_rate=sampling_rate)
                    mm_processor_kwargs["use_audio_in_video"] = True
                else:
                    asset = VideoAsset(name=item["url"], num_frames=num_frames)
                    mm_data.setdefault("video", []).append(asset.np_ndarrays)

            else:
                raise ValueError(f"Unsupported content type: {t}")

        # 4) Flatten singletons (so AudioAsset vs list-of-audio both work)
        for k in ("audio", "image", "video"):
            if k in mm_data and isinstance(mm_data[k], list) and len(mm_data[k]) == 1:
                mm_data[k] = mm_data[k][0]

        # 5) Append the free‐form user text as the last element
        messages[1]["content"].append({"type": "text", "text": user_text})

        # 6) Let the processor inject all <|audio_bos|>, <|IMAGE|>, etc.
        llm_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 7) For vision, get any extra kwargs (fps, etc.) from qwen_vl_utils
        image_in, video_in, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        if image_in  is not None: mm_data["image"] = image_in
        if video_in  is not None: mm_data["video"] = video_in
        # merge with our explicit audio_in_video flag if set
        mm_processor_kwargs = {**video_kwargs, **mm_processor_kwargs}

        # 8) Compute the exact same limit_mm_per_prompt as the demo
        limit_mm_per_prompt: Dict[str, int] = {}
        for k, v in mm_data.items():
            if k in ("audio","image","video"):
                limit_mm_per_prompt[k] = (len(v) if isinstance(v, list) else 1)

        # 9) Instantiate vLLM exactly as in the original main()
        llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt=limit_mm_per_prompt,
            seed=seed,
        )

        # 10) Sampling params exactly like the demo
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 11) Generate
        outputs = llm.generate(
            {"prompt": llm_prompt,
             "multi_modal_data": mm_data,
             "mm_processor_kwargs": mm_processor_kwargs},
            sampling_params=sampling_params,
        )
        text = outputs[0].outputs[0].text

        return {"generated_result": text}

    def finalize(self):
        pass
        # mirror the demo’s cleanup


# 1) Build your inputs dict however you like. For example:
inputs = {
    "system_prompt": "You are a helpful assistant.",
    "prompt": "Describe what’s happening here and transcribe any speech.",
    "seed": 12345,
    "temperature": 0.3,
    "max_tokens": 128,
    "use_audio_in_video": True,
    "contents": [
        # an image
        {"type": "image", "url": "https://github.com/rbgo404/Files/raw/main/dog.jpg"},
    ]
}

# 2) Instantiate and run
model = InferlessPythonModel()
model.initialize()

try:
    result = model.infer(inputs)
    print("Generated result:", result["generated_result"])
finally:
    model.finalize()
