# app/guardrails/provider/llama_guard.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaGuardProvider:
    def __init__(self, model_id: str = "meta-llama/Llama-Guard-3-8B", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    async def check_safety(self, text: str) -> dict:
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30)
            
        output_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        is_safe = output_text.splitlines()[0].lower() == "safe" if output_text else False
        return {"passed": is_safe, "raw": output_text}

    async def check_output_safety(self, query: str, response: str) -> dict:
        # Formats the chat interaction context (User prompt + Model output) for Llama Guard
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30)
            
        output_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        is_safe = output_text.splitlines()[0].lower() == "safe" if output_text else False
        return {"passed": is_safe, "raw": output_text}