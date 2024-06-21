from typing import Iterator
from threading import Thread
import gradio as gr
import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, LlamaTokenizer, TextIteratorStreamer
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--load-in-8bit", action="store_true")
parser.add_argument("--server-name", type=str, default="0.0.0.0")
args = parser.parse_args()

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4096

DESCRIPTION = f"""
# Chat with {args.model_name}
"""

if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    print(texts)
    return ''.join(texts)


def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"].to("cpu")[0]))

    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)


def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message


def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return '', x


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')

model_id = args.model_path

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map='auto'
    )
else:
    model = None

if args.model_name.startswith("llama"):
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Group():
        chatbot = gr.Chatbot(label='Chatbot')
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='Type a message...',
                scale=10,
            )
            submit_button = gr.Button('Submit',
                                    variant='primary',
                                    scale=1,
                                    min_width=0)
    with gr.Row():
        retry_button = gr.Button('🔄  Retry', variant='secondary')
        undo_button = gr.Button('↩️ Undo', variant='secondary')
        clear_button = gr.Button('🗑️  Clear', variant='secondary')

    saved_input = gr.State()

    with gr.Accordion(label='Advanced options', open=False):
        system_prompt = gr.Textbox(label='System prompt',
                                value=DEFAULT_SYSTEM_PROMPT,
                                lines=6)
        max_new_tokens = gr.Slider(
            label='Max new tokens',
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        temperature = gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=1.0,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )

    gr.Examples(
        examples=[
            'Hello there! How are you doing?',
            'Can you explain briefly to me what is the Python programming language?',
            'Explain the plot of Cinderella in a sentence.',
            'How many hours does it take a man to eat a Helicopter?',
            "Write a 100-word article on 'Benefits of Open-Source in AI research'",
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=True,
    )

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

demo.queue(max_size=20).launch()