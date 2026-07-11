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
("feira de       (Qwen2.5-0.5B)         (estilo lambe-lambe)          (MMS-TTS pt)        (imagem+texto+áudio)
 domingo")
```

**Token do estilo:** `estilo_lambelambe,`
**Stack:** Google Colab (GPU T4) · Hugging Face (Hub + Spaces) · GitHub · `diffusers` · `peft` · Gradio.

## Dataset

- **35 imagens** de retratos vintage, todas com licença reutilizável (**Domínio Público / CC0 / CC-BY / CC-BY-SA**).
- **Fontes:** Wikimedia Commons, The Met (Open Access) e Flickr.
- **Proveniência** completa em [`dados/fontes.csv`](dados/fontes.csv) — uma linha por imagem com página de
  origem, link direto, autor, licença e data de coleta.
- **Legendas revisadas** (inglês, prefixadas com o token do estilo) em [`dados/legendas.txt`](dados/legendas.txt)
  — rascunho gerado por **BLIP** no notebook 01 e **revisado manualmente** pela equipe.
- As imagens (`*.jpg`) e o `metadata.jsonl` **não são versionados** — são reconstruídos a partir do
  `fontes.csv` + legendas na execução do notebook 01 (armazenamento no Google Drive).

## Como executar (Google Colab)

Rode os notebooks em `notebooks/` com **Ambiente de execução → T4 GPU** e o secret **`HF_TOKEN`**
(token de escrita) criado no Colab. O notebook 01 faz o *bootstrap* do dataset a partir do repositório.

1. **`01_dataset.ipynb`** — clona o repo, baixa as imagens do `fontes.csv`, **recorta 512×512 centrado no
   rosto**, gera legendas com **BLIP** (inglês) e regrava o `metadata.jsonl` após a revisão.
2. **`02_treino_lora.ipynb`** — fine-tuning **LoRA** (fp16, T4) de **duas configurações** (rank 8 × rank 4),
   com **backup no Drive + upload dos pesos + versionamento semântico** no Hub (função `subir_e_versionar`).
3. **`03_avaliacao.ipynb`** — grade comparativa base × LoRA, **CLIPScore**, verificação de memorização e
   avaliação humana cega (troque `LORA_REVISION` para comparar as versões).
4. **App Gradio** — [`app/app.py`](app/app.py), publicado no Spaces.

## Links

- **Aplicação — Space (Gradio):** https://huggingface.co/spaces/lamble-lambe/atelie
- **Modelo — Pesos LoRA (Hub):** https://huggingface.co/lamble-lambe/atelie
- **Versão em produção:** `v1.4.0`
- **Relatório final:** [`relatorio/relatorio_final.pdf`](relatorio/relatorio_final.pdf)

### Usar o modelo

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
# revision fixa uma versão específica (ver "Versionamento" abaixo); v1.4.0 = produção
pipe.load_lora_weights("lamble-lambe/atelie", revision="v1.4.0")

img = pipe(
    "estilo_lambelambe, a vintage portrait of an elderly man with a hat, detailed face, black and white",
    cross_attention_kwargs={"scale": 0.7},
).images[0]
img.save("saida.png")
```

### Versionamento dos modelos (Hub)

Cada treino recebe uma **tag semântica** `MAJOR.MINOR.PATCH` (função `subir_e_versionar` no notebook 02):
`BUG → PATCH`, `FEATURE → MINOR`, `BREAKING → MAJOR`. A nova versão é calculada a partir da maior tag existente.
Para carregar uma versão específica, use `revision`:

```python
pipe.load_lora_weights("lamble-lambe/atelie", revision="v1.4.0")
```

### Configuração do Space (produção)

O `app/app.py` lê variáveis de ambiente (definidas em **Settings → Variables and secrets** do Space).
A mais importante é a **`LORA_REVISION`**, que fixa qual versão do modelo roda em produção — assim o
Space fica **estável e imune a pushes experimentais** no `main`:

| Variável | Padrão | Função |
|----------|--------|--------|
| `LORA_REVISION` | *(vazio = `main`)* | **Fixa a versão de produção** (em produção: `v1.4.0`). Promover = trocar a tag aqui. |
| `LORA_REPO` | `lamble-lambe/atelie` | Repositório dos pesos LoRA |
| `BASE_MODEL` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Modelo base |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | LLM que expande o tema (em inglês) |
| `TTS_MODEL` | `facebook/mms-tts-por` | Síntese de voz (português) |

> **Nunca** coloque tokens/segredos no código — apenas em *secrets* do Space.

## Estrutura do repositório

```
atelie-generativo-lambe-lambe/
├── README.md
├── .gitignore
├── dados/
│   ├── fontes.csv          # origem, link direto, autor, licença, data — uma linha por imagem (35)
│   └── legendas.txt        # legendas REVISADAS (inglês, token do estilo) — entregável
│                           # (metadata.jsonl e as imagens .jpg são gerados no notebook 01, não versionados)
├── notebooks/
│   ├── 01_dataset.ipynb    # bootstrap + download + recorte no rosto + legendas BLIP + revisão
│   ├── 02_treino_lora.ipynb# fine-tuning LoRA (2 configs) + backup/upload + versionamento semântico
│   └── 03_avaliacao.ipynb  # grade comparativa, CLIPScore, memorização, avaliação humana
├── app/
│   ├── app.py              # aplicação Gradio (publicada no Spaces)
│   ├── requirements.txt    # dependências do Space
│   └── ceub_logo.png       # logo institucional (cabeçalho do app)
└── relatorio/
    └── relatorio_final.pdf
```

## Equipe

**Disciplina:** Inteligência Artificial Generativa e Modelos Multimodais — UniCEUB
**Professor:** Romes Heriberto

**Participantes:**
- Diego Nunes de Morais
- Eduardo Deodoro de Moraes Florindo
- Higo Soares do Lago
- Lucio Flávio Vilar de Azevedo
- Paulo Victor Torres Martins

## Ética e licenças

- Apenas imagens em **domínio público / CC0 / CC-BY / CC-BY-SA / autorais da equipe**.
- Proveniência obrigatória em `dados/fontes.csv` (imagem sem proveniência é desconsiderada).
- Estilo histórico/estética genérica — **vedado** imitar estilo de artistas vivos identificáveis ou reproduzir IP de terceiros.
- Credenciais **somente** em secrets do Colab/Space; nunca no código ou no repositório.
