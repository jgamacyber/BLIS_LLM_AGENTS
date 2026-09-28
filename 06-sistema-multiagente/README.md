# Módulo 06 — Sistema multiagente com supervisor

Um supervisor coordena três especialistas com ferramentas diferentes, usando handoff, estado compartilhado e o RAG como ferramenta.

> ⚠️ **Aviso de uso.** Este material é **educacional**, voltado ao estudo de arquiteturas de agentes. As implementações são reproduções didáticas e simplificadas dos artigos originais, não substituem os métodos publicados nem os resultados neles reportados.

> 📄 **Sobre os artigos.** Os artigos que fundamentam cada peça são **citados ao longo de todo o código**, nos docstrings dos módulos e nos comentários das decisões de projeto. Os PDFs **não** estão incluídos por questão de direitos autorais. Use as referências ao final para localizá-los nas fontes originais.

## Divisão de papéis com restrição real

| Agente | Alcança | Não alcança |
|---|---|---|
| **pesquisador** | `consultar_documentos`, `listar_documentos` | arquivos operacionais, cálculo |
| **analista** | `ler_arquivo`, `listar_arquivos`, `calculadora`, `converter_unidades`, `data_hora` | corpus de artigos |
| **redator** | nenhuma ferramenta de coleta | tudo que não estiver no quadro |
| **supervisor** | nenhuma ferramenta | tudo |

Todos os quatro leem e escrevem no quadro compartilhado, e é só isso que o redator tem.

Essa tabela vive em `ferramentas.py`, em `FERRAMENTAS_POR_PAPEL`, e mudar uma linha dela muda o comportamento do sistema inteiro. É isso que separa uma divisão de papéis de verdade de uma frase no prompt pedindo que cada um cuide da sua parte.

Duas consequências valem o desenho. Como o redator não tem como buscar nada, um fato que ninguém apurou aparece como buraco em vez de ser preenchido com invenção. Como pesquisador e analista não se sobrepõem, qualquer tarefa que cruze as duas fontes exige handoff de verdade, e não por escolha do supervisor.

## Agente conversável

O bloco básico segue o AutoGen: o agente envia, recebe e gera resposta. A parte que mais rende na prática é a **lista de funções de resposta**, tentadas em ordem até uma devolver algo. O artigo chama isso de controle por linguagem de programação, em contraste com escrever a regra dentro do prompt.

As três funções padrão, nesta ordem:

1. **terminação**, que reconhece o encerramento sem gastar uma chamada ao modelo;
2. **cache**, que devolve a resposta anterior quando a mesma instrução chega de novo;
3. **modelo**, o caminho normal, com um laço curto de ferramentas.

O cache não é só economia. Um supervisor confuso reencaminha a mesma instrução várias vezes, e sem ele cada repetição custa uma chamada e produz uma resposta ligeiramente diferente, o que confunde ainda mais a coordenação.

Registrar uma função nova é uma linha, e ela entra na frente da lista.

## Estado compartilhado

A tentação, em um sistema multiagente, é passar tudo pela conversa. Isso tem dois defeitos: o contexto cresce a cada turno, porque todo mundo carrega a fala de todo mundo, e a informação se degrada, porque cada repasse é uma paráfrase da paráfrase anterior.

Aqui os dois canais são separados. A conversa carrega decisões, do tipo "agora é com você, faça X". O quadro compartilhado carrega os fatos, cada um com **autor e fonte**. Quem precisa de um número lê o número, não a lembrança que outro agente tem dele.

O autor do fato vem do registro da ferramenta, não da entrada escrita pelo modelo. Um agente não pode assinar um fato em nome de outro, porque procedência que o próprio agente escolhe não é procedência.

## O RAG como ferramenta

O enunciado do módulo pede que o RAG apareça no papel de ferramenta de um especialista, e é isso que está aqui: chunking por seção com sobreposição, BM25 implementado do zero e citação obrigatória na saída. O rótulo `[fonte: documento#seção]` que a ferramenta devolve é o mesmo que o especialista copia ao registrar o fato, e que chega à resposta final.

O que esta versão **não** faz, e é honesto dizer: não usa embeddings, então não encontra um trecho que responde à pergunta com outras palavras. Para o corpus deste módulo, pequeno e com vocabulário próximo ao das perguntas, o BM25 basta. Em um corpus real, não bastaria. A versão completa, com embeddings, busca híbrida, reordenação e HyDE, está no repositório [blis-rag](https://github.com/jgamacyber/blis-rag).

## As salvaguardas do CAMEL

O artigo trata os quatro modos de falha com regras no prompt de concepção. Isso reduz a frequência, mas não elimina: o prompt é um pedido, não uma garantia. Um sistema que depende só dele descobre a falha quando a conversa já gastou dez rodadas trocando gentilezas.

Aqui os quatro viram detectores que olham a mensagem produzida:

| Modo de falha | Como é detectado |
|---|---|
| **inversão de papéis** | o especialista instrui, delega ou só faz perguntas |
| **repetição da instrução** | a mensagem cobre quase toda a instrução e acrescenta pouco |
| **resposta evasiva** | promete agir sem agir, ou não traz número, fonte nem limite declarado |
| **laço infinito** | só cortesia, ou repetição quase literal da própria fala anterior |

Todos exigem **evidência dupla**: o padrão textual e a ausência de trabalho feito, isto é, nenhuma ferramenta usada e nenhum fato registrado. Prometer e entregar não é evasiva; prometer e não entregar é. Uma resposta curta que traz um número e uma fonte não é evasiva; uma resposta curta e vazia é.

São heurísticas sobre texto, e por isso erram nos dois sentidos. O critério de projeto foi preferir o falso negativo, porque uma correção injetada sem motivo atrapalha um agente que estava indo bem.

Detectada a falha, a correção volta ao mesmo agente junto com a instrução original, e o número de insistências é limitado pelo teto de respostas automáticas consecutivas do AutoGen. Duas ideias de artigos diferentes se encaixando: o CAMEL diz o que detectar, o AutoGen diz quantas vezes insistir antes de parar.

Rode `python main.py sistema "..." --sem-salvaguardas` para comparar.

## Término classificado

A taxonomia do AgentBench, com um acréscimo que só existe em sistema multiagente:

| Código | Significado |
|---|---|
| `complete` | entregou a resposta final |
| `IF` | o supervisor não seguiu o formato de decisão |
| `IA` | insistiu em acionar um especialista inexistente |
| `TLE` | estourou as rodadas, ou repetiu o mesmo especialista sem fechar |
| `SG` | encerrada pelas salvaguardas do CAMEL |
| `erro` | falha de infraestrutura |

## Comandos

Sem chave de API e sem custo:

```bash
python main.py config          # configuração ativa
python main.py time            # papéis e ferramentas de cada um
python main.py rag             # o RAG usado como ferramenta
python main.py estado          # o quadro compartilhado
python main.py salvaguardas    # os quatro modos de falha, com casos negativos
python main.py tarefas         # conjunto de avaliação
python main.py prompt          # prompts do supervisor e dos especialistas
python testes_offline.py       # 190 verificações
python testes_mock.py          # 84 verificações com LLM simulado
```

Com chave, chamando o modelo:

```bash
python main.py sistema "Quantos pedidos estão em trânsito?"
python main.py sistema "O que o CAMEL diz sobre modos de falha?" --sem-salvaguardas
python main.py avaliar
python main.py avaliar --comparar
```

## Avaliação

Oito tarefas com gabarito, em `data/tarefas.json`. Três exigem dois ou mais especialistas, uma exige o time inteiro, e a última não tem resposta em ferramenta nenhuma: ela mede se o sistema admite que não sabe em vez de **inventar em grupo**, que é o pior resultado possível quando vários agentes concordam com algo que ninguém apurou.

O relatório mede quatro coisas separadas, e a separação é o ponto:

- **acerto**: a resposta final bate com o gabarito;
- **roteamento**: os especialistas acionados são os que a tarefa exigia;
- **procedência**: a resposta cita fonte e os fatos do quadro têm origem;
- **término**: a razão de parada, mais as detecções de salvaguarda.

Uma queda no acerto acompanhada de muitos `TLE` é um problema de coordenação. Acompanhada de muitas detecções de laço, é comportamento dos agentes. Com roteamento errado, é o prompt do supervisor. São três consertos diferentes, e a taxa de acerto sozinha não distingue nenhum deles.

O relatório também mostra o **custo de coordenação** por tarefa, em rodadas, handoffs e tokens. Um sistema multiagente é mais caro que um agente só, e vale medir o quanto se paga pelo que se ganha.

Sobre a comparação com e sem salvaguardas: com oito tarefas e uma execução de cada, a diferença entre as duas taxas tem intervalo de confiança largo o bastante para incluir zero em quase qualquer resultado. Serve para ver o mecanismo funcionando e levantar hipótese, não para concluir que as salvaguardas ajudam. O próprio código diz isso ao imprimir a comparação.

## Estrutura

```
06-sistema-multiagente/
├── config.py            Settings a partir do ambiente
├── estado.py            quadro compartilhado com procedência
├── rag.py               chunking, BM25 e citação
├── ferramentas.py       ferramentas e a tabela de papéis
├── agentes.py           agente conversável e funções de resposta
├── salvaguardas.py      os quatro detectores do CAMEL
├── supervisor.py        gerente da conversa, handoff e término
├── avaliacao.py         acerto, roteamento, procedência e custo
├── main.py              CLI
├── testes_offline.py    190 verificações sem rede
├── testes_mock.py       84 verificações com LLM simulado
└── data/
    ├── corpus/          resumos autorais dos artigos
    ├── arquivos/        pedidos, estoque e notas da equipe
    └── tarefas.json     conjunto de avaliação com gabarito
```

## Referências

1. Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* arXiv.
2. Li, G. et al. (2023). *CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society.* NeurIPS.
3. Liu, X. et al. (2024). *AgentBench: Evaluating LLMs as Agents.* ICLR.
4. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
5. Sumers, T. R. et al. (2024). *Cognitive Architectures for Language Agents.* TMLR.
