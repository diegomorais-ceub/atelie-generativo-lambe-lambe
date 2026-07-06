# Ateliê Generativo — Estilo Lambe-Lambe

[![Open in Spaces](https://img.shields.io/badge/🤗%20Spaces-Testar%20a%20aplicação-blue)](https://huggingface.co/spaces/lamble-lambe/atelie)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-Pesos%20LoRA-yellow)](https://huggingface.co/lamble-lambe/atelie)

Projeto da disciplina **Inteligência Artificial Generativa e Modelos Multimodais** (UniCEUB).
Especialização do **Stable Diffusion v1-5** em um estilo visual próprio via **LoRA**, integrado a um
**pipeline multimodal** (LLM → Difusor+LoRA → TTS) publicado na web com Gradio.

## Fluxo da aplicação

```
Tema curto  →  LLM expande prompt  →  Difusor+LoRA gera imagem  →  TTS narra descrição  →  Gradio exibe tudo
("feira de       (Qwen2.5-0.5B)         (estilo lambe-lambe)          (Bark)             (imagem+texto+áudio)
 domingo")
```

**Token do estilo:** `estilo_lambelambe,`
**Stack:** Google Colab (GPU T4) · Hugging Face (Hub + Spaces) · GitHub · `diffusers` · `peft` · Gradio.

## Como executar

1. Abrir os notebooks em `notebooks/` no Google Colab (Ambiente de execução → T4 GPU).
2. `01_dataset.ipynb` — coleta de imagens, legendas BLIP e revisão manual.
3. `02_treino_lora.ipynb` — fine-tuning LoRA e publicação dos pesos no Hub.
4. `03_avaliacao.ipynb` — grade comparativa, CLIPScore e avaliação humana.
5. App Gradio: ver `app/app.py` (publicado no Spaces).

## Links

- **Aplicação — Space (Gradio):** https://huggingface.co/spaces/lamble-lambe/atelie
- **Modelo — Pesos LoRA (Hub):** https://huggingface.co/lamble-lambe/atelie
- **Relatório final:** [`relatorio/relatorio_final.pdf`](relatorio/relatorio_final.pdf)

### Usar o modelo

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("lamble-lambe/atelie")

img = pipe("estilo_lambelambe, retrato de uma feira de domingo").images[0]
img.save("saida.png")
```

## Estrutura do repositório

```
atelie-generativo-lambe-lambe/
├── README.md
├── dados/
│   ├── fontes.csv          # url, autor, licença, data — uma linha por imagem
│   └── legendas.txt        # caption revisado de cada imagem
├── notebooks/
│   ├── 01_dataset.ipynb
│   ├── 02_treino_lora.ipynb
│   └── 03_avaliacao.ipynb
├── app/
│   └── app.py              # aplicação Gradio (publicada no Spaces)
└── relatorio/
    └── relatorio_final.pdf
```

## Equipe

| Frente | Responsável | Entregável principal |
|--------|-------------|----------------------|
| Dados | _________ | `01_dataset.ipynb`, `fontes.csv`, legendas |
| Treinamento | _________ | `02_treino_lora.ipynb`, pesos no Hub |
| Pipeline | _________ | `app/app.py` |
| Interface | _________ | Space Gradio público |
| Documentação | _________ | `relatorio_final.pdf`, model card, README |

## Ética e licenças

- Apenas imagens em **domínio público / CC0 / CC-BY / CC-BY-SA / autorais da equipe**.
- Proveniência obrigatória em `dados/fontes.csv` (imagem sem proveniência é desconsiderada).
- Vedado imitar estilo de artistas vivos identificáveis ou reproduzir IP de terceiros.
- Credenciais **somente** em secrets do Space; nunca no código.
