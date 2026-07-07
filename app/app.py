"""Aplicação Gradio — pipeline multimodal do Ateliê Generativo (estilo lambe-lambe).

Encadeia:  LLM (expande o tema)  ->  Stable Diffusion v1-5 + LoRA (gera imagem)  ->  TTS (narra).

Publicação: Hugging Face Spaces (SDK Gradio).
IMPORTANTE: credenciais SOMENTE em Settings -> Variables and secrets do Space. Nunca no código.

Modelos são carregados de forma preguiçosa (lazy): só na primeira geração, para não
estourar a memória na inicialização do Space e não pagar o custo de carga se ninguém usar.
"""

import os

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel

# --- Configuração (sobrescrevível por variável de ambiente no Space) ---------
BASE_MODEL = os.environ.get("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
LORA_REPO = os.environ.get("LORA_REPO", "lamble-lambe/atelie")  # pesos oficiais da equipe
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TTS_MODEL = os.environ.get("TTS_MODEL", "facebook/mms-tts-por")  # TTS neural em português
STYLE_TOKEN = "estilo_lambelambe,"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Estado dos modelos (preenchido sob demanda) ----------------------------
_diffusion = None
_llm = None
_llm_tok = None
_tts = None
_tts_tok = None


def _get_diffusion():
    global _diffusion
    if _diffusion is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=DTYPE, safety_checker=None
        )
        try:
            pipe.load_lora_weights(LORA_REPO)
            print(f"[ok] LoRA aplicado: {LORA_REPO}")
        except Exception as erro:  # sem LoRA publicado ainda -> cai no modelo base
            print(f"[aviso] LoRA não carregado ({erro}); usando somente o modelo base")
        _diffusion = pipe.to(DEVICE)
    return _diffusion


def _get_llm():
    global _llm, _llm_tok
    if _llm is None:
        _llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        _llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=DTYPE).to(DEVICE)
    return _llm, _llm_tok


def _get_tts():
    global _tts, _tts_tok
    if _tts is None:
        _tts_tok = AutoTokenizer.from_pretrained(TTS_MODEL)
        _tts = VitsModel.from_pretrained(TTS_MODEL).to(DEVICE)
    return _tts, _tts_tok


# --- Etapas do pipeline -----------------------------------------------------
def expandir_prompt(tema: str) -> str:
    """Expande um tema curto em uma descrição visual rica, em inglês.

    O prompt de difusão sai em inglês porque o Stable Diffusion v1-5 é nativo em
    inglês; o token do estilo (STYLE_TOKEN) é quem aciona o LoRA, independente do idioma.
    Se o LoRA for treinado com legendas em português (frente de dados), trocar a
    instrução abaixo para gerar em português e alinhar com a equipe.
    """
    model, tok = _get_llm()
    mensagens = [
        {
            "role": "system",
            "content": (
                "You expand a short theme into a single rich, concrete visual "
                "description for image generation. Reply with one English sentence, "
                "no preamble, no quotes."
            ),
        },
        {"role": "user", "content": f"Theme: {tema}"},
    ]
    entrada = tok.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    ids = tok(entrada, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        saida = model.generate(**ids, max_new_tokens=80, do_sample=True, temperature=0.7)
    texto = tok.decode(saida[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return f"{STYLE_TOKEN} {texto}"


def gerar_imagem(prompt: str, guidance_scale: float = 7.5, seed: int = 0):
    """Gera a imagem com Stable Diffusion v1-5 + pesos LoRA da equipe."""
    pipe = _get_diffusion()
    gerador = torch.Generator(device=DEVICE).manual_seed(int(seed))
    passos = 20 if DEVICE == "cpu" else 30  # menos passos em CPU (Space grátis) para não travar
    resultado = pipe(
        prompt,
        guidance_scale=float(guidance_scale),
        num_inference_steps=passos,
        generator=gerador,
    )
    return resultado.images[0]


def narrar(tema: str):
    """Sintetiza narração em português (VITS/MMS-TTS) a partir do tema.

    Devolve (sample_rate, waveform) — formato aceito por gr.Audio.
    """
    model, tok = _get_tts()
    frase = f"Uma imagem em estilo lambe-lambe inspirada em {tema}."
    ids = tok(frase, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        onda = model(**ids).waveform
    return model.config.sampling_rate, onda.squeeze().cpu().numpy()


def pipeline(tema: str, guidance_scale: float, seed: int):
    prompt = expandir_prompt(tema)
    imagem = gerar_imagem(prompt, guidance_scale, seed)
    audio = narrar(tema)
    return imagem, prompt, audio


# --- Interface Gradio -------------------------------------------------------
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
