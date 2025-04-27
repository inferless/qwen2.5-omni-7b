import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import io
import base64
import inferless
from pydantic import BaseModel, Field
from typing import Optional


@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Who are you?")
    system_prompt: str = Field(default="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.")
    video_url: Optional[str] = None
    image_url: Optional[str] = None
    audio_url: Optional[str] = None

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")
    generated_audio: Optional[str] = None


class InferlessPythonModel:
    def initialize(self):
        model_id = "Qwen/Qwen2.5-Omni-7B"
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cuda")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        

    def infer(self, request: RequestObjects) -> ResponseObjects:
        conversation = self.get_query(request.system_prompt,  request.prompt, request.video_url, request.image_url, request.audio_url)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=True)
        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        buffer = io.BytesIO()
        sf.write(
            buffer,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
            format='WAV'
        )
        
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        generateObject = ResponseObjects(generated_text=text[0],generated_audio=audio_base64)
        return generateObject
    
    def get_query(self, system_prompt,  prompt, video_url=None, image_url=None, audio_url=None):
        content_list = []
        if image_url:
            content_list.append({"type": "image", "image": image_url})
        if video_url:
            content_list.append({"type": "video", "video": video_url})
        if audio_url:
            content_list.append({"type": "audio", "audio": audio_url})
        if prompt:
            content_list.append({"type": "text", "text": prompt})

        return [
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": content_list,
            }
        ]

    def finalize(self):
        self.model = None
