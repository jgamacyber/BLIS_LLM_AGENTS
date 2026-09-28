# BLIS Agents — Agentes com ferramentas e sistemas multiagente

Implementações práticas e documentadas de **agentes de linguagem**, do laço ReAct com ferramentas ao sistema com supervisor e especialistas, cada peça fundamentada em um artigo da área. Os dois módulos são autocontidos, rodam por CLI e usam a **API da OpenRouter**.

Todo o código foi escrito do zero, sem frameworks de agentes. Não há LangChain, LlamaIndex, AutoGen nem CrewAI: o laço ReAct, a memória hierárquica, o BM25, o gerente de conversa em grupo e os detectores de modo de falha estão implementados à mão, porque o objetivo é **entender o mecanismo**, não montar o agente mais rápido.

> ⚠️ **Aviso de uso.** Este material é **educacional**, voltado ao estudo de arquiteturas de agentes. As implementações são reproduções didáticas e simplificadas dos artigos originais, não substituem os métodos publicados nem os resultados neles reportados.

> 📄 **Sobre os artigos.** Os artigos que fundamentam cada peça são **citados ao longo de todo o trabalho**, nos docstrings dos módulos, nos comentários das decisões de projeto e na lista de referências ao final. Os PDFs **não** estão incluídos no repositório por questão de direitos autorais. Use as referências para localizá-los nas fontes originais.

## Módulos

| Módulo | Tema | Artigos de referência |
|--------|------|----------------------|
| [`05-agente-react`](./05-agente-react) | Laço ReAct, sete ferramentas funcionais, memória hierárquica e taxonomia de término | ReAct (Yao et al., 2023) · MemGPT (Packer et al., 2023) · CoALA (Sumers et al., 2024) · Toolformer (Schick et al., 2023) · AgentBench (Liu et al., 2024) |
| [`06-sistema-multiagente`](./06-sistema-multiagente) | Supervisor e especialistas, handoff, estado compartilhado, RAG como ferramenta e as salvaguardas do role-playing | AutoGen (Wu et al., 2023) · CAMEL (Li et al., 2023) · AgentBench (Liu et al., 2024) |

## O que cada módulo faz

### Módulo 05 — Agente ReAct

O laço completo, do pensamento à ação e de volta:

```
Thought  -> raciocínio sobre o que fazer agora
Action   -> chamada de ferramenta
Observation -> resultado real, devolvido pelo sistema
... até Final Answer
```

Sete ferramentas funcionais, todas determinísticas e offline: calculadora com avaliação de AST, busca BM25 no corpus, leitura de arquivos em sandbox, listagem de arquivos e de documentos, data e hora, e conversão de unidades.

A memória segue a hierarquia do MemGPT, com contexto principal e contexto externo, aviso de pressão em 70% da janela, flush em 100% e resumo recursivo do que sai. O agente pode operar a própria memória com `working_context_append`, `archival_insert`, `archival_search` e `recall_search`.

O término não é sucesso ou falha: é classificado na taxonomia do AgentBench, que separa formato inválido, ação inexistente, estouro de passos e estouro de contexto. Cada uma dessas razões aponta para um conserto diferente.

### Módulo 06 — Sistema multiagente

Um supervisor sem ferramenta nenhuma coordena três especialistas com ferramentas diferentes:

| Agente | O que alcança | O que não alcança |
|---|---|---|
| **pesquisador** | corpus de artigos, via RAG | arquivos operacionais, cálculo |
| **analista** | planilhas, notas, cálculo, conversão | corpus de artigos |
| **redator** | só o quadro compartilhado | qualquer coleta de dados |
| **supervisor** | decidir quem trabalha e fechar a resposta | qualquer ferramenta |

Essa tabela é a divisão de papéis de verdade, e não uma frase no prompt. Como o redator não tem como buscar nada, um fato que ninguém apurou aparece como buraco em vez de ser preenchido com invenção. Como pesquisador e analista não se sobrepõem, qualquer tarefa que cruze as duas fontes exige handoff.

O **RAG é usado como ferramenta**, no papel que o enunciado do módulo pede: chunking por seção com sobreposição, BM25 do zero e citação obrigatória, de forma que o rótulo da fonte acompanha o fato até a resposta final.

Os quatro modos de falha do CAMEL, inversão de papéis, repetição da instrução, resposta evasiva e laço infinito, são detectados **fora do prompt**, por heurísticas que exigem evidência dupla: o padrão textual e a ausência de trabalho feito. Detectada a falha, a correção volta ao mesmo agente, e o número de insistências é limitado pelo teto de respostas automáticas consecutivas do AutoGen.

## Início rápido

```bash
cd 05-agente-react                    # ou 06-sistema-multiagente

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # preencha OPENROUTER_API_KEY

python testes_offline.py              # valida a lógica sem gastar créditos
python testes_mock.py                 # valida as chamadas de API (LLM simulado)

python main.py ferramentas            # módulo 05, sem custo
python main.py agente "Qual o valor total do pedido 4471?"

python main.py time                   # módulo 06, sem custo
python main.py sistema "Quantos pedidos estão em trânsito?"
```

O passo a passo completo, com os resultados esperados de cada comando e a estimativa de custo, está em **[REPRODUTIBILIDADE.md](./REPRODUTIBILIDADE.md)**.

## Validação sem custo

As duas suítes de teste de cada módulo não fazem uma única chamada de rede:

| Módulo | Testes offline | Testes com LLM simulado |
|---|---|---|
| 05 | 146 | 71 |
| 06 | 190 | 84 |

Os testes offline cobrem ferramentas, memória, parsers, BM25 e verificação de respostas. Os testes com LLM simulado encenam de propósito cada razão de término, incluindo as de falha: formato inválido, ação inexistente, laço, estouro de limites e queda de rede. Um agente costuma ser testado só no caminho feliz, e é nos caminhos de falha que ele trava depois.

## Configuração da API

Os dois módulos usam a [OpenRouter](https://openrouter.ai/) como gateway, compatível com o SDK da OpenAI:

```
OPENROUTER_API_KEY=sua_chave_aqui
MODEL=openai/gpt-4o-mini
TEMPERATURE=0.0
```

A temperatura padrão é zero de propósito. Um agente que escolhe ações precisa ser reproduzível: com temperatura alta, o mesmo erro não se repete e não há como depurar.

Boa parte dos comandos roda **sem chave nenhuma**, porque as ferramentas, a memória, o RAG, o estado compartilhado e as salvaguardas são determinísticos. Só o que chama o modelo precisa de crédito.

## Corpus e dados de exemplo

Os documentos em `data/corpus/` são **resumos autorais em português**, escritos a partir da leitura dos artigos. Não são traduções nem reproduções dos textos originais.

Os dados operacionais em `data/arquivos/` são fictícios, criados para este estudo: uma planilha de pedidos, um inventário e um arquivo de notas com regras de negócio. Servem para dar ao agente um mundo pequeno, verificável e sem custo, em que cada resposta tem gabarito.

## Resultados de exemplo

O módulo 05 mede a taxa de sucesso junto com a distribuição de razões de término, porque a taxa sozinha não diz o que consertar: muitos IF apontam para o prompt, muitos IA para o catálogo de ferramentas e muitos TLE para a capacidade de planejar em vários turnos.

O módulo 06 mede quatro coisas separadas, e a separação é o ponto: acerto, roteamento, procedência e término. Uma resposta pode estar certa e sem fonte, e isso é um resultado diferente de estar certa e auditável.

Os números dependem do modelo e da chave de cada um. Reproduza com `python main.py avaliar` e compare.

## Licença

MIT, ver [`LICENSE`](./LICENSE).

A licença cobre o **código deste repositório**. Os artigos citados pertencem a seus respectivos autores e editoras.

## Referências

Todos os artigos abaixo são citados no código, nos pontos em que a ideia correspondente é implementada. Os PDFs não são distribuídos aqui.

1. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.
2. Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv.
3. Sumers, T. R. et al. (2024). *Cognitive Architectures for Language Agents.* TMLR.
4. Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS.
5. Liu, X. et al. (2024). *AgentBench: Evaluating LLMs as Agents.* ICLR.
6. Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* arXiv.
7. Li, G. et al. (2023). *CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society.* NeurIPS.

## Sobre

Parte da trilha prática do **BLIS, Brazilian Laboratory for Intelligent Systems**, 2026. Cada módulo reproduz artigos acadêmicos em código executável, com testes e documento de reprodutibilidade.

Outros repositórios da trilha: [BLIS_LLM_SEC](https://github.com/jgamacyber/BLIS_LLM_SEC) (segurança de LLM), [blis-rag](https://github.com/jgamacyber/blis-rag) (RAG básico e avançado), [blis-prompt-engineering](https://github.com/jgamacyber/blis-prompt-engineering) (engenharia de prompts) e [blis-evals](https://github.com/jgamacyber/blis-evals) (avaliação de sistemas com LLM).
