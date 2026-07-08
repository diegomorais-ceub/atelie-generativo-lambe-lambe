# Ateliê Generativo — Estilo Lambe-Lambe

[![Open in Spaces](https://img.shields.io/badge/🤗%20Spaces-Testar%20a%20aplicação-blue)](https://huggingface.co/spaces/lamble-lambe/atelie)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-Pesos%20LoRA-yellow)](https://huggingface.co/lamble-lambe/atelie)

Projeto da disciplina **Inteligência Artificial Generativa e Modelos Multimodais** (UniCEUB).
Especialização do **Stable Diffusion v1-5** em um estilo visual próprio via **LoRA**, integrado a um
**pipeline multimodal** (LLM → Difusor+LoRA → TTS) publicado na web com Gradio.

O estilo escolhido é a **fotografia lambe-lambe / retrato vintage** — preto e branco, granulação e
composição clássica de estúdio (tintypes, ambrótipos, cartes de visite e cabinet cards do séc. XIX / início do XX).

## Fluxo da aplicação

```
Tema curto  →  LLM expande prompt  →  Difusor+LoRA gera imagem  →  TTS narra descrição  →  Gradio exibe tudo
("feira de       (Qwen2.5-0.5B)         (estilo lambe-lambe)          (Bark)             (imagem+texto+áudio)
 domingo")
```

**Token do estilo:** `estilo_lambelambe,`
**Stack:** Google Colab (GPU T4) · Hugging Face (Hub + Spaces) · GitHub · `diffusers` · `peft` · Gradio.

## Dataset

- **41 imagens** de retratos vintage, todas com licença reutilizável (**Domínio Público / CC0 / CC-BY / CC-BY-SA**).
- **Fontes:** Wikimedia Commons, The Met (Open Access) e Flickr.
- **Proveniência** completa em [`dados/fontes.csv`](dados/fontes.csv) — uma linha por imagem com página de
  origem, link direto, autor, licença e data de coleta.
- **Legendas em inglês** (melhor para o SD-1.5), prefixadas com o token do estilo, em
  [`dados/metadata.jsonl`](dados/metadata.jsonl) / [`dados/legendas.txt`](dados/legendas.txt) — rascunho via **BLIP** e revisão manual.

## Como executar (Google Colab)

Rode os notebooks em `notebooks/` com **Ambiente de execução → T4 GPU**. Pré-requisito: subir
`dados/fontes.csv` e `dados/metadata.jsonl` para a pasta do Drive `MyDrive/Colab Notebooks/multimodais/dados`
e criar o secret **`HF_TOKEN`** (token de escrita) no Colab.

1. **`01_dataset.ipynb`** — baixa as imagens do `fontes.csv`, gera legendas com **BLIP** (inglês) e
   regrava o `metadata.jsonl` após a revisão manual.
2. **`02_treino_lora.ipynb`** — fine-tuning **LoRA** (fp16, T4), `push_to_hub` dos pesos e **versionamento
   semântico** da release no Hub (célula interativa `#2`).
3. **`03_avaliacao.ipynb`** — grade comparativa base × LoRA, **CLIPScore**, verificação de memorização e avaliação humana.
4. **App Gradio** — [`app/app.py`](app/app.py), publicado no Spaces.

## Links

- **Aplicação — Space (Gradio):** https://huggingface.co/spaces/lamble-lambe/atelie
- **Modelo — Pesos LoRA (Hub):** https://huggingface.co/lamble-lambe/atelie
- **Relatório final:** [`relatorio/relatorio_final.pdf`](relatorio/relatorio_final.pdf)

### Usar o modelo

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
# opcional: revision="v1.0.0" carrega uma versão específica (ver "Versionamento" abaixo)
pipe.load_lora_weights("lamble-lambe/atelie")

img = pipe("estilo_lambelambe, a vintage black and white portrait of a woman at a sunday market").images[0]
img.save("saida.png")
```

### Versionamento dos modelos (Hub)

Cada treino é publicado no Hub e recebe uma **tag semântica** `MAJOR.MINOR.PATCH` (célula `#2` do notebook 02):
`BUG → PATCH`, `FEATURE → MINOR`, `BREAKING → MAJOR`. A nova versão é calculada a partir da maior tag em produção.
Para carregar uma versão específica, use `revision`:

```python
pipe.load_lora_weights("lamble-lambe/atelie", revision="v1.0.0")
```

## Estrutura do repositório

```
atelie-generativo-lambe-lambe/
├── README.md
├── .gitignore
├── dados/
│   ├── fontes.csv          # página de origem, link direto, autor, licença, data — uma linha por imagem
│   ├── metadata.jsonl      # file_name + caption (inglês, token do estilo) — usado no treino
│   └── legendas.txt        # mesmas legendas em texto, para revisão humana
├── notebooks/
│   ├── 01_dataset.ipynb    # download das imagens + legendas BLIP + revisão
│   ├── 02_treino_lora.ipynb# fine-tuning LoRA + push_to_hub + versionamento semântico
│   └── 03_avaliacao.ipynb  # grade comparativa, CLIPScore, memorização, avaliação humana
├── app/
│   ├── app.py              # aplicação Gradio (publicada no Spaces)
│   └── requirements.txt    # dependências do Space
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
- Estilo histórico/estética genérica — **vedado** imitar estilo de artistas vivos identificáveis ou reproduzir IP de terceiros.
- Credenciais **somente** em secrets do Colab/Space; nunca no código ou no repositório.
