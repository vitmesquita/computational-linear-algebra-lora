# Decomposição em Valores Singulares na Adaptação de Baixo Posto de Modelos de Linguagem

Repositório de pesquisa para comparar `full fine-tuning`, `LoRA` e `AdaLoRA` em `GPT-2` com `WikiText-2`.

O projeto combina treino de modelos com uma etapa de análise espectral e algébrica dos updates aprendidos, usando artefatos salvos pelos scripts para estudar perda, custo computacional, posto efetivo, energia espectral e similaridade de subespaço entre métodos.

## Visão Geral

Este repositório foi organizado para apoiar um estudo acadêmico sobre adaptação eficiente de modelos de linguagem. O foco principal é observar como diferentes estratégias de ajuste alteram os pesos do modelo e como essas alterações podem ser comparadas por métricas clássicas de álgebra linear computacional.

No estado atual, os experimentos usam `GPT-2` da biblioteca `transformers` e o conjunto `Salesforce/wikitext`, configuração `wikitext-2-raw-v1`. Os scripts treinam o modelo, salvam artefatos intermediários e finais, e o notebook em [analysis/article_analysis.ipynb](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/analysis/article_analysis.ipynb) lê esses artefatos para gerar tabelas e gráficos.

## Objetivos do Estudo

- Comparar `full fine-tuning`, `LoRA` e `AdaLoRA` em um mesmo backbone (`GPT-2`).
- Medir diferenças de custo e eficiência, incluindo parâmetros treináveis e tempo total de execução.
- Investigar propriedades algébricas dos updates aprendidos por meio de espectros singulares, `stable_rank` e ranks associados à energia acumulada.
- Explorar a proximidade entre subespaços principais dos updates, especialmente na comparação entre `LoRA` e `AdaLoRA`.

## Estrutura do Repositório

| Caminho | Descrição |
| --- | --- |
| [src/model](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model) | Entrypoints de treino e submódulos auxiliares organizados por responsabilidade. |
| [src/model/data](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/data) | Camada de dados, atualmente com o dataset `WikiTextDataset`. |
| [src/model/metrics](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/metrics) | Construção das métricas de treino e de hardware, além dos schemas dos históricos salvos. |
| [src/model/runtime](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/runtime) | Infraestrutura comum dos experimentos: seed, device, dataloaders, config, checkpoints e persistência de artefatos. |
| [src/model/analysis](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/analysis) | Análise algébrica pós-treino, incluindo deltas, SVD e estatísticas espectrais por camada. |
| [analysis](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/analysis) | Notebook de análise para gerar tabelas e gráficos a partir dos artefatos salvos. |
| [cla_lora_runs](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/cla_lora_runs) | Artefatos de execuções anteriores versionados no repositório. |
| `outputs/cla_lora_runs` | Diretório criado localmente pelos scripts atuais para novos experimentos. |

### Organização de `src/model`

- [src/model/lora.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/lora.py), [src/model/adalora.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/adalora.py) e [src/model/full_finetune.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/full_finetune.py) ficaram como entrypoints principais, concentrando o loop de treino e a lógica específica de cada método.
- [src/model/data/load_dataset.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/data/load_dataset.py) define o `WikiTextDataset` usado pelos experimentos.
- [src/model/metrics/training_metrics.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/metrics/training_metrics.py) monta as linhas de `train_history.csv`.
- [src/model/metrics/hardware_metrics.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/metrics/hardware_metrics.py) encapsula coleta, resumo e schema das métricas de hardware.
- [src/model/runtime/experiment_utils.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/runtime/experiment_utils.py) concentra a infraestrutura compartilhada entre os métodos.
- [src/model/analysis/algebraic_analysis.py](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/src/model/analysis/algebraic_analysis.py) centraliza a análise espectral dos deltas salvos ao final do treino.

## Requisitos e Instalação

- Python compatível com [pyproject.toml](/Users/voymesquita/Documents/mestrado/algebra_linear_computacional/trabalho/computational-linear-algebra-lora/pyproject.toml), atualmente `>=3.11`.
- `Poetry` para instalar dependências e executar os scripts.
- `cuda`, `mps` ou `cpu`. Os scripts escolhem o dispositivo automaticamente nessa ordem.

Instalação:

```bash
poetry install
```

Observações:

- Os comandos de execução devem continuar sendo feitos a partir da raiz do repositório.
- Os entrypoints em `src/model` agora importam helpers organizados em subpastas como `data`, `metrics`, `runtime` e `analysis`, mas o modo de execução permanece o mesmo.
- O notebook de análise usa `pandas` e um ambiente Jupyter ou VS Code Notebook. Esses itens não aparecem explicitamente no `pyproject.toml` atual.

## Como Executar Cada Experimento

### Full Fine-Tuning

```bash
poetry run python src/model/full_finetune.py
```

Esse script treina todos os parâmetros do `GPT-2`, salva métricas por época e também registra artefatos algébricos dos deltas finais das camadas monitoradas.

### LoRA

```bash
poetry run python src/model/lora.py
```

Esse script congela o backbone, substitui camadas lineares alvo por módulos `LoRALinear` e salva deltas finais, espectros singulares e estatísticas por camada.

### AdaLoRA

```bash
poetry run python src/model/adalora.py
```

Esse script segue a lógica de adaptação de posto com orçamento variável, mantendo informações adicionais como `effective_rank` nas estatísticas finais por camada.

Observações:

- Os hiperparâmetros ficam definidos diretamente no corpo dos scripts.

## Onde os Artefatos São Salvos

### Execução Local

- `full_finetune.py` salva novos runs em `outputs/cla_lora_runs/full_finetune/<timestamp>/`.
- `lora.py` salva novos runs em `outputs/cla_lora_runs/lora/<timestamp>/`.
- `adalora.py` salva novos runs em `outputs/cla_lora_runs/adalora/<timestamp>/`.

### Execução em Colab

- `full_finetune.py` usa `/content/drive/MyDrive/cla_lora_runs/full_finetune/<timestamp>/` quando o Google Drive está montado. Se `/content` existir sem Drive montado, ele usa `/content/cla_lora_runs/full_finetune/<timestamp>/`.
- `lora.py` e `adalora.py` verificam apenas a existência de `/content` e, nesse caso, direcionam a saída para `/content/drive/MyDrive/cla_lora_runs/<metodo>/<timestamp>/`.

## Como Abrir e Usar `analysis/article_analysis.ipynb`

1. Garanta que existam artefatos completos para os métodos que você quer comparar.
2. Abra analysis/article_analysis.ipynb em Jupyter ou VS Code.
3. Revise as constantes `RUN_ROOT`, `FULL_FINETUNE_ROOT` e `RUN_NAMES` logo nas primeiras células.
4. Execute as células em ordem para carregar os runs mais recentes configurados nesses diretórios.

Observação importante:

- No estado atual, o notebook procura `LoRA` e `AdaLoRA` dentro de `cla_lora_runs/lora_session/lora/` e `cla_lora_runs/adalora_session/adalora/`, enquanto `full_finetune` é buscado em `outputs/cla_lora_runs/full_finetune/`.
- Se você gerar novos runs usando os scripts atuais, pode ser necessário ajustar `RUN_NAMES` e `FULL_FINETUNE_ROOT` para apontar para os diretórios corretos.

## Artefatos Gerados

| Arquivo | Finalidade |
| --- | --- |
| `config.json` | Registra método, dataset, hiperparâmetros, dispositivo e caminho de saída. |
| `train_history.csv` | Guarda métricas por época, incluindo loss e tempos acumulados. |
| `summary.json` | Resume os principais resultados finais do run. |
| `target_deltas.pt` | Salva os deltas finais das camadas monitoradas para análise algébrica. |
| `target_svdvals.pt` | Salva os valores singulares de cada delta monitorado. |
| `layer_stats.json` | Consolida estatísticas por camada, como `fro_norm`, `spectral_norm`, `stable_rank` e ranks por energia acumulada. |
| `latest_checkpoint.pt` | Checkpoint atualizado ao final de cada época. |
| `final_checkpoint.pt` | Checkpoint final do experimento. |

Detalhes adicionais:

- Em `AdaLoRA`, `layer_stats.json` também inclui `effective_rank`.
- Em `full_finetune`, `train_history.csv` e `summary.json` incluem métricas extras de comparação baseadas em loss `legacy`.

## O Que É Comparado na Análise

- `loss` por época.
- `loss` final comparável entre métodos.
- Quantidade absoluta e percentual de parâmetros treináveis.
- Tempo total de execução.
- Espectro singular dos deltas por camada.
- `stable_rank`.
- `energy_90_rank` e `energy_95_rank`.
- Similaridade de subespaço entre deltas, especialmente entre `LoRA` e `AdaLoRA`.

## Limitações e Observações Atuais

- Os scripts escolhem automaticamente entre `cuda`, `mps` e `cpu`, então o comportamento depende do hardware disponível.
- O notebook lê os runs mais recentes dentro dos diretórios configurados nas primeiras células, não necessariamente os diretórios padrão de saída dos scripts atuais.
- `full_finetune.py` usa uma loss mascarada para otimização e também registra métricas `legacy` para manter comparabilidade com os experimentos de `LoRA` e `AdaLoRA`.
- A análise depende da presença dos artefatos esperados em disco. Se faltar `target_deltas.pt`, `target_svdvals.pt`, `layer_stats.json` ou `summary.json`, algumas células do notebook não vão executar corretamente.
- Como os hiperparâmetros ficam definidos no código, mudanças nos scripts podem tornar os artefatos antigos diferentes dos runs novos, mesmo dentro do mesmo método.
