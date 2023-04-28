import gradio as gr
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoModel
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LLaMATokenizer, AutoModel
from fastchat.conversation import get_default_conv_template
import abc
import re

@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""

class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        #print(f"{role}: ")
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream, skip_echo_len: int):
        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)

import argparse
parser = argparse.ArgumentParser(description='Ko Vicuna')
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()

#model_path = "/home/joo/PycharmProjects/FastChat_new/checkpoints/checkpoint-31000/"
model_path = args.model_path
kwargs = {"torch_dtype": torch.float16}
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             low_cpu_mem_usage=True, **kwargs)
device = "cuda:0" if torch.cuda.is_available() else -1
model.to(device)
conv = get_default_conv_template(model_path).copy()

def translate(text):
    temperature = float(0.7)
    max_new_tokens = int(512)
    chatio = SimpleChatIO()
    print(text)
    #inp = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", text)
    inp = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s?]", "", text)
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    generate_stream_func = generate_stream
    prompt = conv.get_prompt()
    skip_echo_len = len(prompt.replace("</s>", " ")) + 1

    params = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": conv.sep,
    }

    chatio.prompt_for_output(conv.roles[1])
    output_stream = generate_stream_func(model, tokenizer, params, device)
    outputs = chatio.stream_output(output_stream, skip_echo_len)
    conv.messages[-1][-1] = outputs.strip()
    return conv.messages[-1][-1]


title = """<h1 align="center">Demo of Korean GPT4 with Vicuna</h1>"""
description = """<h2> Korean Vicuna 데모 페이지입니다. </h2>"""
#article = """<h4><a href='https://hotco87.github.io'><img src='https://img.shields.io/badge/Github-Code'></a></h4>"""
#article = """<h4><a href='http://cilab.gist.ac.kr/hp/gallery/?pageid=1&mod=document&uid=16'><img src='http://cilab.gist.ac.kr/hp/wp-content/uploads/kboard_attached/1/202105/609fd3f4369675439724.jpeg'></a></h4>"""
#http://cilab.gist.ac.kr/hp/wp-content/uploads/kboard_attached/1/202105/609fd3f4369675439724.jpeg
demo = gr.Blocks()

def bot(history):
    response = translate(history[-1][0])
    history[-1][1] = response
    return history

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

with demo:
    gr.Markdown(title)
    gr.Markdown(description)
    #gr.Markdown(article)
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)
    #chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="무엇이든 물어보세요. 텍스트를 입력해주세요.",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            clear = gr.Button("Clear")
    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="172.27.186.15", share=True)