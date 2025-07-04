from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
import torch, gc
from PIL import Image

MODEL_ID = "nanonets/Nanonets-OCR-s"

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)     # comment out to run full-precision
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,                     # or torch.float16
    device_map="auto",        # remove if flash-attn not installed
)
model.eval()

tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID)
processor  = AutoProcessor.from_pretrained(MODEL_ID)


PROMPT = """Extract the text from the above document as if you were reading it naturally.
Return tables as HTML and equations as LaTeX."""

def ocr_image(path, max_new_tokens=4096):
    img = Image.open(path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{path}"},
            {"type": "text",  "text": PROMPT},
        ]}
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text], images=[img], return_tensors="pt", padding=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = out[:, inputs.input_ids.shape[-1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
