# Módulo 05 — Agente ReAct com ferramentas

Um agente que alterna raciocínio e ação, com sete ferramentas funcionais, memória hierárquica e término classificado.

> ⚠️ **Aviso de uso.** Este material é **educacional**, voltado ao estudo de arquiteturas de agentes. As implementações são reproduções didáticas e simplificadas dos artigos originais, não substituem os métodos publicados nem os resultados neles reportados.

> 📄 **Sobre os artigos.** Os artigos que fundamentam cada peça são **citados ao longo de todo o código**, nos docstrings dos módulos e nos comentários das decisões de projeto. Os PDFs **não** estão incluídos por questão de direitos autorais. Use as referências ao final para localizá-los nas fontes originais.

## A ideia do ReAct

O artigo de Yao et al. (2023) formaliza o agente aumentando o espaço de ações com a linguagem:

```
Â = A ∪ L
```

onde `A` são as ações que afetam o ambiente, isto é, as ferramentas, e `L` é o espaço da linguagem. Uma ação em `L` é um **pensamento**: ela não afeta o ambiente e não gera observação, apenas atualiza o contexto.

Isso resolve dois problemas opostos. O raciocínio em cadeia puro é um processo fechado, que não se ancora em nada externo e por isso alucina e propaga erro. Agir sem raciocinar produz sequências de ações sem plano. O ReAct alterna os dois.

## As sete ferramentas

Todas determinísticas, offline e sem custo:

| Ferramenta | O que faz |
|---|---|
| `calculadora` | avalia expressão aritmética pela AST |
| `buscar_corpus` | busca BM25 nos resumos dos artigos |
| `listar_corpus` | lista os documentos disponíveis |
| `ler_arquivo` | lê um arquivo dentro da sandbox |
| `listar_arquivos` | lista os arquivos operacionais |
| `data_hora` | data e hora atuais em UTC |
| `converter_unidades` | comprimento, massa, tempo e dados |

Duas decisões de segurança merecem destaque, porque quem escolhe a string de entrada é o modelo, não a pessoa.

A **calculadora avalia a árvore sintática, nunca `eval()`**. Cada nó da AST é verificado contra uma lista de operadores, funções e constantes permitidos, e qualquer construção fora dela é recusada. Expoentes grandes também são recusados, para que `9**9**9` não trave o processo.

A **leitura de arquivos é confinada a uma sandbox**. O caminho pedido é resolvido e comparado com a raiz permitida, de modo que `../../config.py` e `/etc/passwd` voltam como erro em vez de conteúdo.

Erro é dado, não exceção. Uma ferramenta que falha devolve o texto do erro como observação, e o agente pode corrigir o curso no passo seguinte. Um agente que trava porque uma ferramenta levantou exceção não tem como aprender nada com a falha.

## Memória hierárquica

A memória segue o MemGPT, com a analogia de sistema operacional:

```
CONTEXTO PRINCIPAL (a "RAM", dentro da janela)
  instruções do sistema   somente leitura
  working context         fatos que o agente escolheu manter à vista
  fila FIFO               histórico recente
  resumo recursivo        o que já saiu, condensado

CONTEXTO EXTERNO (o "disco", fora da janela)
  recall storage          todas as mensagens que já passaram
  archival storage        fatos guardados de propósito
```

Em 70% da janela, um alerta de pressão entra no contexto avisando o agente para salvar o que importa. Em 100%, metade da fila é despejada para o recall e vira resumo recursivo. O alerta é disparado na **subida** do limiar, uma vez só: repetir o aviso a cada mensagem gastaria justamente o contexto que ele existe para proteger.

O resumo é extrativo e determinístico, para que o comportamento da memória seja testável sem custo. A versão do artigo, que usa o próprio modelo para resumir, está em `resumir_com_llm` e cai de volta no extrativo se a chamada falhar.

Com `--memoria`, o catálogo do agente ganha as funções de memória e ele passa a decidir sozinho o que salvar e o que recuperar.

## Término classificado

O agente para por um motivo, e o motivo é a informação útil. A taxonomia vem do AgentBench:

| Código | Significado | O que consertar |
|---|---|---|
| `complete` | concluiu a tarefa | nada |
| `IF` | não seguiu o formato exigido | o prompt está mal especificado |
| `IA` | escolheu uma ação inexistente | o catálogo de ferramentas está confuso |
| `TLE` | estourou os passos ou entrou em laço | o agente não planeja em vários turnos |
| `CLE` | estourou o limite de contexto | a memória não consegue liberar espaço |
| `erro` | falha de infraestrutura | nada, é rede |

Nem `IF` nem `IA` matam a execução na primeira ocorrência. O agente recebe de volta uma observação explicando o erro, com a lista de ferramentas válidas quando é o caso, e costuma se corrigir no passo seguinte. Insistir no erro é que encerra a execução.

## Comandos

Sem chave de API e sem custo:

```bash
python main.py config          # configuração ativa
python main.py ferramentas     # exercita as sete ferramentas
python main.py memoria         # demonstra pressão, flush e resumo recursivo
python main.py tarefas         # lista o conjunto de avaliação
python main.py prompt          # imprime o prompt do sistema
python testes_offline.py       # 146 verificações
python testes_mock.py          # 71 verificações com LLM simulado
```

Com chave, chamando o modelo:

```bash
python main.py agente "Qual o valor total do pedido 4471?"
python main.py agente "Quanto pesam os monitores em libras?" --memoria
python main.py avaliar
python main.py avaliar --tarefas t01,t04,t10
```

## Avaliação

Dez tarefas com gabarito, em `data/tarefas.json`, sobre um mundo pequeno e verificável: uma planilha de pedidos, um inventário e um arquivo de notas com regras de negócio.

Seis delas exigem combinar duas ou mais ferramentas, por exemplo ler a planilha, aplicar um desconto descrito nas notas e converter o resultado para outra unidade. A décima não tem resposta em ferramenta nenhuma: ela mede **abstenção**, ou seja, se o agente admite que não sabe em vez de inventar um número plausível.

O relatório traz a taxa de sucesso junto com a distribuição de razões de término, porque a taxa sozinha não diz o que consertar.

Também traz uma medida de utilidade por ferramenta, comparando a taxa de acerto das tarefas em que a ferramenta foi usada com a das tarefas em que não foi. Ela é inspirada no critério do Toolformer, mas **não é o critério do artigo**, que mede redução de perda em tokens. É uma aproximação por comportamento, está rotulada como tal no código e só é interpretável com um número razoável de tarefas em cada grupo.

## Estrutura

```
05-agente-react/
├── config.py            Settings a partir do ambiente
├── ferramentas.py       as sete ferramentas e o registro
├── memoria.py           a hierarquia do MemGPT
├── react.py             o laço, o parser e a taxonomia de término
├── avaliacao.py         verificação, relatório e utilidade por ferramenta
├── main.py              CLI
├── testes_offline.py    146 verificações sem rede
├── testes_mock.py       71 verificações com LLM simulado
└── data/
    ├── corpus/          resumos autorais dos artigos
    ├── arquivos/        sandbox: pedidos, estoque e notas
    └── tarefas.json     conjunto de avaliação com gabarito
```

## Referências

1. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
2. Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv.
3. Sumers, T. R. et al. (2024). *Cognitive Architectures for Language Agents.* TMLR.
4. Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS.
5. Liu, X. et al. (2024). *AgentBench: Evaluating LLMs as Agents.* ICLR.
