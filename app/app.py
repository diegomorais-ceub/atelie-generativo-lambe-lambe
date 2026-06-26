"""Aplicação Gradio — pipeline multimodal do Ateliê Generativo (estilo lambe-lambe).

Encadeia:  LLM (expande o tema)  →  Stable Diffusion v1-5 + LoRA (gera imagem)  →  TTS (narra).

Publicação: Hugging Face Spaces (SDK Gradio).
IMPORTANTE: credenciais SOMENTE em Settings → Variables and secrets do Space. Nunca no código.
"""

import os

import gradio as gr

# --- Configuração (ajustar aos repos da equipe) -----------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_REPO = os.environ.get("LORA_REPO", "equipe/lora-estilo-lambelambe")
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STYLE_TOKEN = "estilo_lambelambe,"


def expandir_prompt(tema: str) -> str:
    """Usa o LLM para expandir um tema curto em um prompt descritivo."""
    # TODO: carregar LLM_MODEL e gerar a expansão.
    return f"{STYLE_TOKEN} {tema}"


def gerar_imagem(prompt: str, guidance_scale: float = 7.5, seed: int = 0):
    """Gera a imagem com Stable Diffusion v1-5 + pesos LoRA."""
    # TODO: carregar BASE_MODEL, aplicar LORA_REPO via peft e gerar a imagem.
    raise NotImplementedError("Implementar carregamento do difusor + LoRA")


def narrar(texto: str):
    """Gera narração em áudio (TTS) da descrição da imagem."""
    # TODO: carregar o modelo de TTS (ex.: Bark) e sintetizar o áudio.
    raise NotImplementedError("Implementar TTS")


def pipeline(tema: str, guidance_scale: float, seed: int):
    prompt = expandir_prompt(tema)
    imagem = gerar_imagem(prompt, guidance_scale, seed)
    audio = narrar(prompt)
    return imagem, prompt, audio


with gr.Blocks(title="Ateliê Generativo — Lambe-Lambe") as demo:
    gr.Markdown("# Ateliê Generativo — Estilo Lambe-Lambe")
    with gr.Row():
        tema = gr.Textbox(label="Tema", placeholder="ex.: feira de domingo")
    with gr.Row():
        guidance = gr.Slider(1.0, 15.0, value=7.5, label="Guidance scale")
        seed = gr.Number(value=0, label="Seed", precision=0)
    btn = gr.Button("Gerar", variant="primary")
    with gr.Row():
        out_img = gr.Image(label="Imagem gerada")
        with gr.Column():
            out_prompt = gr.Textbox(label="Prompt expandido")
            out_audio = gr.Audio(label="Narração")

    btn.click(pipeline, inputs=[tema, guidance, seed], outputs=[out_img, out_prompt, out_audio])


if __name__ == "__main__":
    demo.launch()
