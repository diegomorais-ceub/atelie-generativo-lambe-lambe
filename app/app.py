"""Aplicação Gradio — pipeline multimodal do Ateliê Generativo (estilo lambe-lambe).

Encadeia:  LLM (expande o tema)  ->  Stable Diffusion v1-5 + LoRA (gera imagem)  ->  TTS (narra).

Publicação: Hugging Face Spaces (SDK Gradio).
IMPORTANTE: credenciais SOMENTE em Settings -> Variables and secrets do Space. Nunca no código.

Modelos são carregados de forma preguiçosa (lazy): só na primeira geração, para não
estourar a memória na inicialização do Space e não pagar o custo de carga se ninguém usar.
"""

import base64
import mimetypes
import os

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel

# --- Configuração (sobrescrevível por variável de ambiente no Space) ---------
BASE_MODEL = os.environ.get("BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
LORA_REPO = os.environ.get("LORA_REPO", "lamble-lambe/atelie")  # pesos oficiais da equipe
# Versão do LoRA em produção. Vazio = último push no main (muda a cada treino).
# Recomendado: defina uma tag semver (ex.: "v1.0.0") nas Settings do Space para fixar a produção.
LORA_REVISION = os.environ.get("LORA_REVISION", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TTS_MODEL = os.environ.get("TTS_MODEL", "facebook/mms-tts-por")  # TTS neural em português
STYLE_TOKEN = "estilo_lambelambe,"

# --- Identificação do projeto (exibida na aba "Sobre") ----------------------
DISCIPLINA = "Inteligência Artificial Generativa e Modelos Multimodais — UniCEUB"
PROFESSOR = "Prof. Romes Heriberto"
PARTICIPANTES = [
    "Diego Nunes de Morais",
    "Eduardo Deodoro de Moraes Florindo",
    "Higo Soares do Lago",
    "Lucio Flavio Vilar de Azevedo",
    "Paulo Victor Torres Martins",
]
# Versão vigente do modelo em produção (vinda de LORA_REVISION; vazio = última do main)
VERSAO_VIGENTE = LORA_REVISION or "main (última publicada)"

SOBRE_MD = f"""
### Sobre o projeto
**Ateliê Generativo — Estilo Lambe-Lambe.** Especialização do **Stable Diffusion v1-5** no estilo de
**fotografia lambe-lambe / retrato vintage** (preto e branco, granulação, composição clássica) via **LoRA**,
integrada a um **pipeline multimodal**: um LLM expande o tema, o difusor com o LoRA gera a imagem no estilo
treinado e um modelo de voz narra a descrição.

- **Disciplina:** {DISCIPLINA}
- **Professor:** {PROFESSOR}
- **Participantes:** {", ".join(PARTICIPANTES)}
- **Modelo:** [`{LORA_REPO}`](https://huggingface.co/{LORA_REPO})
- **Versão vigente (produção):** `{VERSAO_VIGENTE}`
"""

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
            # revision=None -> main (última versão); revision="vX.Y.Z" -> versão fixada
            pipe.load_lora_weights(LORA_REPO, revision=LORA_REVISION or None)
            print(f"[ok] LoRA aplicado: {LORA_REPO}@{LORA_REVISION or 'main'}")
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


# --- Identidade visual (cores UniCEUB) --------------------------------------
CEUB_ROXO = "#56148a"           # roxo institucional UniCEUB (extraído do site oficial)
CEUB_ROXO_ESCURO = "#330066"
CEUB_DOURADO = "#f0cc25"        # dourado de destaque
LOGO_PATH = os.environ.get("LOGO_PATH", "ceub_logo.png")  # logo oficial (logoCEUB2021.png) na raiz do Space


def _logo_data_uri(path: str) -> str:
    """Embute a logo como data-URI (evita depender do file-serving do Space)."""
    if not os.path.exists(path):
        return ""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    dados = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{dados}"


LOGO_URI = _logo_data_uri(LOGO_PATH)

TEMA_CEUB = gr.themes.Default(
    primary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.gray,
)
CSS_CEUB = f"""
#cabecalho {{ display:flex; align-items:center; gap:16px;
             border-bottom:4px solid {CEUB_DOURADO}; padding-bottom:12px; margin-bottom:6px; }}
#cabecalho img {{ height:56px; width:auto; }}
#cabecalho h1 {{ margin:0; color:{CEUB_ROXO}; font-weight:800; line-height:1.1; }}
button.primary, .primary {{ background:{CEUB_ROXO} !important; border-color:{CEUB_ROXO} !important; }}
button.primary:hover, .primary:hover {{ background:{CEUB_ROXO_ESCURO} !important; }}
a {{ color:{CEUB_ROXO}; }}
"""

# --- Interface Gradio -------------------------------------------------------
with gr.Blocks(title="Ateliê Generativo — Lambe-Lambe") as demo:
    _logo_img = f'<img src="{LOGO_URI}" alt="UniCEUB">' if LOGO_URI else ""
    gr.HTML(f'<div id="cabecalho">{_logo_img}<h1>Ateliê Generativo — Estilo Lambe-Lambe</h1></div>')
    gr.Markdown(f"*{DISCIPLINA} · {PROFESSOR} · modelo `{VERSAO_VIGENTE}`*")
    with gr.Accordion("Sobre o projeto", open=True):
        gr.Markdown(SOBRE_MD)
    with gr.Row():
        tema = gr.Textbox(label="Tema", placeholder="e.g.: an old sunday market")
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
    # Gradio 6: theme e css são passados no launch() (não mais no Blocks()).
    demo.launch(theme=TEMA_CEUB, css=CSS_CEUB)
