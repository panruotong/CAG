import gradio as gr
import gradio as gr
import logging
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from callbacks import Iteratorize, Stream
import torch


def generate_stream(model, tokenizer, input, max_length=128, do_sample=True):
    input_ids = tokenizer(input, return_tensors="pt", padding=True).to(model.device)

    def generate_with_callback(callback=None, **kwargs):
        kwargs["stopping_criteria"].append(Stream(callback_func=callback))
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    eos_token_ids = [tokenizer.eos_token_id]

    # 从data和配置文件中读取generate配置
    generate_params = {
        "max_new_tokens": max_length,
        "stopping_criteria": transformers.StoppingCriteriaList(),
        "do_sample": do_sample,
        "temperature": 0.5,
        "top_p": 1,
        "typical_p": 1.0,
        "repetition_penalty": 1.2,
        "top_k": 40,
        "min_length": 1,
    }

    with generate_with_streaming(**input_ids, **generate_params) as generator:
        reply = ""
        for output in generator:
            output = output.tolist()
            if output[-1] in eos_token_ids:
                break
            reply = tokenizer.decode(output, skip_special_tokens=True)
            # print("reply:"+reply)
            yield reply.replace(input, "")
        logging.info(f"\nprompt:{input}\nreply:{reply}\n")

def infer(question, high_credibility, medium_credibility, low_credibility):
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.The assistant answers questions based on given passages. Each article has a credibility rating that is categorized as: high, medium, or low. Credibility reflects the relevance and accuracy of the article to the question. The assistant's answer will need to synthesize the content and credibility of multiple articles. USER: Question:{question}\n"
    if high_credibility!="":
        prompt += f"High credibility of text: {high_credibility}\n\n"
    if medium_credibility!="":
        prompt += f"Middle credibility of text: {medium_credibility}\n\n"
    if low_credibility!="":
        prompt += f"Low credibility of text: {low_credibility}\n\n"
    prompt += " ASSISTANT:"
    print(prompt)
    for output in generate_stream(model, tokenizer, prompt, max_length=512):
        yield output

if __name__ == "__main__":
    #load model
    model_path = "/mnt/panruotong2021/Code/output_models/confidence_v4"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    with gr.Blocks() as demo:
        gr.Markdown("# Customize credibility with CAG!")
        with gr.Row():
            question = gr.Textbox(label="Question")
        with gr.Column():
            with gr.Row():
                high_credibility = gr.Textbox(label="High credibility")
            with gr.Row():
                medium_credibility = gr.Textbox(label="Medium credibility")
            with gr.Row():
                low_credibility = gr.Textbox(label="Low credibility")
            examples = [["What position does David Cameron serve?", "David Cameron, former PM and now Britain's new foreign minister", "", "David Cameron was the youngest Prime Minister since Lord Liverpool in 1812.\nDavid Cameron lands teaching job at Abu Dhabi university"]]
            
        with gr.Row():
            submit_button = gr.Button("Generate")

        gr.Examples(examples=examples, inputs=[question, high_credibility, medium_credibility, low_credibility])
        submit_button.click(
            fn=infer,
            inputs=[question, high_credibility, medium_credibility, low_credibility],
            outputs=[gr.Textbox(label="Output")],
        )
        
        #demo.add_examples(examples, inputs=[question, high_credibility, medium_credibility, low_credibility], outputs=[submit_button])
    server_name = "0.0.0.0"
    server_port = 7789

    demo.queue()
    demo.launch(share=False, server_name=server_name, server_port=server_port)
    #demo.launch(share=True)
