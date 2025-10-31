import re
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, 
    BitsAndBytesConfig
)
from typing import Dict, Any, Tuple
import unicodedata
import logging
import time

logger = logging.getLogger(__name__)
from src.config import (
    SEED, SYSTEM_PROMPT, MODEL_CONFIG
)


def set_deterministic_generation(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_config = MODEL_CONFIG.get(model_name, {})
    is_prm_model = model_config.get("is_prm", False)

    # The PRM doesn't require full precision, loading it in 4-bit is a fair memory reduction and speed up
    if is_prm_model:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            num_labels=2,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.warning(f"Tokenizer for '{model_name}' has no `pad_token`. Setting it to `eos_token` to avoid resizing.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning(f"Tokenizer for '{model_name}' has no `pad_token` or `eos_token`. Adding a new pad token and resizing embeddings.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

    device = model.device
    
    if not is_prm_model:
        model = torch.compile(model)

    return model, tokenizer, device


def extract_final_answer(text: str) -> str | None:
    # Extracts the content from the last \\boxed{...} block in a string using brace matching and sanitization
    
    # 1- Isolate the assistant's response
    assistant_markers = ["<|assistant|>", "[/INST]", "Assistant:"]
    last_marker_pos = -1
    marker_found = None
    for marker in assistant_markers:
        pos = text.rfind(marker)
        if pos > last_marker_pos:
            last_marker_pos = pos
            marker_found = marker
            
    if marker_found:
        response_content = text[last_marker_pos + len(marker_found):]
    else:
        # Fallback for models that might not echo the full chat structure, find the last question/answer separator.
        last_marker_pos = text.rfind("\nA:")
        if last_marker_pos != -1:
            response_content = text[last_marker_pos + len("\nA:"):]
        else:
            response_content = text # Assuming the whole text is the response (assuming this isn't an issue based on the strict PRM claim identification)

    # 2- Find the last box and perform brace matching
    last_box_start = response_content.rfind("\\boxed{")
    if last_box_start == -1:
        return None

    search_start = last_box_start + len("\\boxed{")
    
    brace_level = 1
    for i in range(search_start, len(response_content)):
        char = response_content[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
            if brace_level == 0:
                # 3- Sanitize the extracted content
                extracted_content = response_content[search_start:i]
                
                sanitized_content = extracted_content.strip()
                
                if sanitized_content.endswith(('.', ',', ';', '$')):
                    sanitized_content = sanitized_content[:-1].strip()
                    
                return sanitized_content

    return None

def prepare_batch_for_generation(
    prompts: list[str],
    model_name: str,
    tokenizer: AutoTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:

    prep_start_time = time.time()
    config = MODEL_CONFIG.get(model_name, {})
    logger.debug(f"Preparing batch for model '{model_name}' using config: {config}")
    
    chat_template = config.get("chat_template")
    if not chat_template:
        raise ValueError(f"Could not find a chat_template in config for model {model_name}.")

    batched_messages = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {prompt}\\nA:"}
        ]
        batched_messages.append(messages)

    if tokenizer.padding_side != "left":
        logger.warning(f"Tokenizer padding_side is '{tokenizer.padding_side}', changing to 'left' for generation.")
        tokenizer.padding_side = "left"
    
    # Let the tokenizer handle the chat template logic internally if it can, if it fails, we supply our own from the config.
    if tokenizer.chat_template:
         inputs = tokenizer.apply_chat_template(
            batched_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        )
    else:
        # Supply here
        logger.info("Tokenizer has no default template. Applying the hardcoded template from config.")

    inputs = tokenizer.apply_chat_template(
        batched_messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    )

    input_ids = inputs
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask


def generate_responses_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int = 1024,
    stop_tokens: list[str] | None = None,
) -> list[str]:
    generation_params = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if stop_tokens:
        stop_token_ids = [
            tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens
        ]
        if stop_token_ids:
            generation_params["eos_token_id"] = stop_token_ids
    
    inference_start_time = time.time()
    with torch.no_grad():
        generation_output = model.generate(**generation_params)

    # Decode the tokens to text, skipping special tokens
    decode_start_time = time.time()
    batch_responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

    # Remove the prompt from the responses
    return [text.strip() for text in batch_responses]
