# Documento de Reprodutibilidade

Roteiro para reproduzir todos os resultados deste repositório na sua máquina, com sua chave da OpenRouter. Cada comando traz o que esperar de saída e o custo aproximado.

---

## 1. Ambiente

### Requisitos

- Python **3.10 ou superior** (o código usa `X | None`, sintaxe de união de tipos do 3.10)
- Conexão com a internet apenas para os comandos que chamam a API
- Uma chave da OpenRouter: https://openrouter.ai/keys

### Verificar a versão do Python

```bash
python --version      # deve mostrar 3.10+
```

Se a sua distribuição tiver `python3` em vez de `python`, use `python3` em todos os comandos.

### Instalação

Faça isso **dentro da pasta do módulo** que vai rodar. Cada módulo é autocontido e tem o seu próprio ambiente:

```bash
cd 05-agente-react                # ou 06-sistema-multiagente

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
```

Dependências instaladas: `openai` e `python-dotenv`. Nada além disso. Não há biblioteca de agentes, de RAG nem de avaliação: BM25, memória hierárquica, laço ReAct, gerente de conversa e detectores estão implementados no próprio repositório.

### Configuração

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Edite o `.env` e preencha a chave:

```
OPENROUTER_API_KEY=sk-or-v1-...sua_chave...
MODEL=openai/gpt-4o-mini
TEMPERATURE=0.0
```

> O `.env` está no `.gitignore` e nunca deve ser commitado. O repositório traz apenas o `.env.example`.

**Importante rodar de dentro da pasta do módulo.** Os caminhos de dados são relativos (`data/corpus`, `data/arquivos`), então `python 05-agente-react/main.py` a partir da raiz não encontra o corpus.

---

## 2. Validação sem custo

Antes de gastar qualquer crédito, rode as duas suítes de teste de cada módulo. Nenhuma delas faz uma única chamada de rede, e as duas levam menos de um segundo cada.

### Módulo 05

```bash
cd 05-agente-react
python testes_offline.py
python testes_mock.py
```

Esperado, ao final de cada uma:

```
====================================================================
  Todos os testes passaram.
====================================================================
```

São **146 verificações offline** e **71 com LLM simulado**. Se alguma falhar, a saída lista o nome de cada verificação que falhou antes do resumo.

O que os testes offline cobrem: precedência e segurança da calculadora (inclusive recusa de `__import__`, `open`, `lambda` e expoentes gigantes), fuga da sandbox de arquivos, conversão de unidades com ida e volta, índice BM25, memória hierárquica com pressão e flush, parser da saída do modelo em sete variações, taxonomia de término e verificação de respostas.

O que os testes com LLM simulado cobrem: o laço completo no caminho feliz, e depois cada razão de término provocada de propósito, isto é, formato inválido, ação inexistente, estouro de passos, laço, estouro de contexto e queda de rede.

### Módulo 06

```bash
cd 06-sistema-multiagente
python testes_offline.py
python testes_mock.py
```

São **190 verificações offline** e **84 com LLM simulado**, com o mesmo formato de saída.

Vale olhar dois grupos em particular. Nas salvaguardas, os casos **negativos** são os que importam: uma resposta que usa vocabulário parecido com a instrução mas fez trabalho não é repetição, e uma que começa com "vou calcular" mas entrega o número na mesma mensagem não é evasiva. Nos testes com LLM simulado, o teste do isolamento do redator verifica que `ler_arquivo` e `consultar_documentos` não aparecem no catálogo dele, porque é isso que sustenta a divisão de papéis.

---

## 3. Módulo 05, comandos sem custo

Todos funcionam sem `OPENROUTER_API_KEY`.

### 3.1 Configuração ativa

```bash
python main.py config
```

Esperado:

```
  modelo=openai/gpt-4o-mini | temp=0.0 | max_passos=8 | janela=4000 tok | api_key=AUSENTE

  limiar de pressão:  2800 tokens (70% da janela)
  fração do flush:    50% da fila
  ...
  ferramentas (7): calculadora, buscar_corpus, listar_corpus, ler_arquivo, listar_arquivos, data_hora, converter_unidades
```

### 3.2 As sete ferramentas

```bash
python main.py ferramentas
```

Faz onze chamadas reais, escolhidas para mostrar o contrato de cada ferramenta e não apenas o caso feliz. Duas delas falham de propósito:

```
  [ERR] calculadora[10 / 0]
        ERRO: ZeroDivisionError: divisão por zero

  [ERR] ler_arquivo[../../config.py]
        ERRO: ValueError: acesso negado: '../../config.py' está fora da pasta permitida (data/arquivos)
```

Ao final:

```
  11 chamadas, 2 com erro tratado.
```

O ponto da demonstração é que o erro volta como texto, e não como exceção que derruba o agente. Vale conferir também a chamada `calculadora[sqrt(144) + round(2.71828, 2)]`, que precisa devolver `14.72`: a vírgula que separa argumentos não pode ser confundida com a vírgula decimal brasileira de `98,50`.

### 3.3 Memória hierárquica

```bash
python main.py memoria
```

Usa uma janela de 150 tokens de propósito, para que a pressão e o flush aconteçam em poucas mensagens. Esperado:

```
  msg  8  ocupação   74%   <- aviso de pressão
  ...
  msg 12  ocupação   95%
  msg 13  ocupação   77%   <- FLUSH: a fila foi despejada para o recall

  Memória  [##########################....] 87% (130/150 tokens)
    main context:     7 msgs na fila, 0 fatos no working context
    external context: 14 msgs no recall, 0 fatos no archival
    eventos:          1 alertas de pressão, 1 flushes
```

Repare em **1 alerta** para cinco mensagens acima do limiar: o aviso sai na subida, não a cada mensagem. E em **14 mensagens no recall** contra 7 na fila: o que saiu do contexto continua recuperável por `recall_search`.

O comando termina imprimindo o contexto principal montado, que é exatamente o texto que iria ao modelo, e salva o estado completo em `saidas/memoria_demo.json`.

### 3.4 Conjunto de tarefas e prompt

```bash
python main.py tarefas
python main.py prompt --memoria
```

O primeiro lista as dez tarefas com gabarito. O segundo imprime o prompt do sistema com as funções de memória incluídas no catálogo.

---

## 4. Módulo 05, comandos com custo

Os custos abaixo usam `openai/gpt-4o-mini` como referência e são aproximados. Modelos maiores custam mais e tendem a fechar as tarefas em menos passos.

### 4.1 Uma tarefa

```bash
python main.py agente "Qual o valor total do pedido 4471, somando itens e frete?"
```

Custo aproximado: **US$ 0,002 a 0,01**, de duas a cinco chamadas conforme o modelo resolva em menos ou mais passos.

Saída esperada, em formato:

```
  ── passo 1 ──
  Thought: preciso ver a planilha de pedidos
  Action:  ler_arquivo[pedidos.csv]
  Obs:     pedido;cliente;data;itens;valor_unitario;frete;status ...

  ── passo 2 ──
  Thought: 3 itens de 98,50 mais 22,50 de frete
  Action:  calculadora[(3 * 98.50) + 22.50]
  Obs:     318

  ── passo 3 ──
  Thought: já tenho o valor
  Finish:  O total do pedido 4471 é R$ 318,00.

  [OK] complete — concluiu a tarefa
```

O rastro completo, com os tokens de cada passo, fica em `saidas/ultima_execucao.json`.

### 4.2 O conjunto inteiro

```bash
python main.py avaliar
```

Dez tarefas, com um agente novo para cada uma. Custo aproximado: **US$ 0,03 a 0,10**.

Para gastar menos enquanto confere o encanamento, rode um subconjunto:

```bash
python main.py avaliar --tarefas t01,t04,t10
```

Essas três cobrem os três tipos: cálculo sobre arquivo, busca no corpus e abstenção.

O relatório traz a taxa de sucesso, a distribuição de razões de término, o uso por ferramenta e a medida de utilidade. A tabela final tem uma linha por tarefa, e o JSON fica em `saidas/relatorio.json`.

**Como ler o resultado.** A taxa de sucesso sozinha não diz o que consertar. Muitos `IF` apontam para o prompt, muitos `IA` para o catálogo de ferramentas e muitos `TLE` para a capacidade do modelo de planejar em vários turnos. A tarefa `t10` merece atenção separada: ela só é acertada se o agente admitir que não sabe.

---

## 5. Módulo 06, comandos sem custo

```bash
cd 06-sistema-multiagente
```

### 5.1 O time

```bash
python main.py time
```

Esperado:

```
  PESQUISADOR (pesquisador)
    ferramentas: consultar_documentos, listar_documentos, registrar_fato, ler_fatos, escrever_artefato, ler_artefato

  ANALISTA (analista)
    ferramentas: ler_arquivo, listar_arquivos, calculadora, converter_unidades, data_hora, registrar_fato, ...

  REDATOR (redator)
    ferramentas: registrar_fato, ler_fatos, escrever_artefato, ler_artefato
```

O redator tem apenas as ferramentas do quadro compartilhado. É isso que impede que ele complete um buraco com invenção.

### 5.2 O RAG como ferramenta

```bash
python main.py rag
```

Indexa o corpus e roda cinco consultas. Esperado:

```
  6 documentos, 40 trechos indexados.

  consulta: quais são os quatro modos de falha do role-playing
    [ 5.03] camel#Os quatro modos de falha
```

A última consulta, sobre bolo de cenoura, volta vazia de propósito: é o caso em que a ferramenta precisa dizer que não achou, em vez de devolver o trecho menos ruim como se servisse.

### 5.3 O quadro compartilhado

```bash
python main.py estado
```

Mostra dois especialistas escrevendo fatos com fonte, a recusa de duplicata, a escrita e a leitura de artefato, e o resumo que o supervisor vê antes de decidir. Salva tudo em `saidas/estado_demo.json`.

### 5.4 As salvaguardas

```bash
python main.py salvaguardas
```

Seis casos: os quatro modos de falha do CAMEL, mais **dois casos negativos** que não devem disparar nada. Esperado:

```
  resposta evasiva
    mensagem: Vou calcular o total do pedido agora mesmo e retorno com o número.
    trabalho feito: 0 ferramentas, 0 fatos
    -> resposta_evasiva: prometeu sem executar: 'vou calcular'

  promessa seguida de entrega (não deve disparar)
    mensagem: Vou calcular: 3 x 98,50 = 295,50, mais 22,50 de frete, total R$ 318,00. [fonte: pedidos.csv]
    trabalho feito: 1 ferramentas, 1 fatos
    -> nada detectado
```

A mesma abertura, "vou calcular", em um caso dispara e no outro não. A diferença é o trabalho feito, e é por isso que os detectores exigem evidência dupla.

---

## 6. Módulo 06, comandos com custo

### 6.1 Uma tarefa

```bash
python main.py sistema "Quantos pedidos estão em trânsito?"
```

Custo aproximado: **US$ 0,005 a 0,02**. Uma tarefa de um especialista só costuma gastar duas decisões do supervisor mais duas ou três chamadas do especialista.

Saída esperada, em formato:

```
  ── rodada 1 ──
    handoff -> analista: conte os pedidos com status em_transito em pedidos.csv
    analista [modelo] (ler_arquivo; 1 fatos): são 2 pedidos em trânsito [fonte: pedidos.csv]

  ── rodada 2 ──
    supervisor: RESPOSTA FINAL

  [OK] complete — entregou a resposta final
  ...
  A resposta final cita fonte: sim
```

Uma tarefa que cruze corpus e dados operacionais, como a `m06`, gasta mais:

```bash
python main.py sistema "Monte um documento com quantos pedidos há em cada status e por que limitar as rodadas da conversa"
```

Custo aproximado: **US$ 0,02 a 0,05**, porque envolve três especialistas e mais rodadas de coordenação.

### 6.2 O conjunto inteiro

```bash
python main.py avaliar
```

Oito tarefas, com um sistema novo para cada uma, e portanto quadro compartilhado limpo. Custo aproximado: **US$ 0,08 a 0,20**.

Com a comparação com e sem salvaguardas, o custo dobra:

```bash
python main.py avaliar --comparar     # US$ 0,15 a 0,40
```

O relatório traz as quatro medidas separadas:

```
  Acerto:      xx.x% (n/8)
  Roteamento:  xx.x% das tarefas acionaram pelo menos os especialistas necessários
  Procedência: xx.x% das respostas que exigiam fonte citaram uma

  Razões de término (AgentBench):
    complete    n
    TLE         n
  ...
  Custo de coordenação, por tarefa:
    x.x rodadas, x.x handoffs, xxxx tokens, x.x fatos registrados
```

**Como ler o resultado.** As quatro medidas apontam para consertos diferentes. Acerto baixo com muitos `TLE` é coordenação. Acerto baixo com muitas detecções de laço é comportamento dos agentes. Roteamento baixo é o prompt do supervisor. Procedência baixa com acerto alto significa que o sistema está certo por motivos que ninguém pode auditar, o que é um problema próprio.

**Sobre a comparação com e sem salvaguardas.** Com oito tarefas e uma execução de cada, o intervalo de confiança da diferença inclui zero em quase qualquer resultado. A comparação serve para ver o mecanismo funcionando e levantar hipótese, não para concluir que as salvaguardas melhoram o sistema. Concluir isso exigiria repetições e um conjunto maior de tarefas. O próprio comando imprime esse aviso ao final.

---

## 7. Custo total para reproduzir tudo

| Etapa | Custo aproximado |
|---|---|
| Todos os comandos sem custo, nos dois módulos | US$ 0,00 |
| Todos os testes, nos dois módulos | US$ 0,00 |
| Módulo 05, uma tarefa | US$ 0,002 a 0,01 |
| Módulo 05, conjunto inteiro | US$ 0,03 a 0,10 |
| Módulo 06, uma tarefa | US$ 0,005 a 0,02 |
| Módulo 06, conjunto inteiro | US$ 0,08 a 0,20 |
| Módulo 06, conjunto com comparação | US$ 0,15 a 0,40 |
| **Total** | **US$ 0,30 a 0,80** |

Os valores usam `openai/gpt-4o-mini` como referência, em setembro de 2026. Confira o preço atual na OpenRouter, porque a tabela muda.

---

## 8. Solução de problemas

**`OPENROUTER_API_KEY não encontrada`**
O `.env` não existe ou está em outra pasta. Confira que você copiou o `.env.example` dentro da pasta do módulo e que está rodando o `main.py` de lá.

**`FileNotFoundError: data/corpus`**
Você rodou o comando da raiz do repositório. Entre na pasta do módulo primeiro.

**O agente termina com muitos `IF`**
O modelo escolhido tem dificuldade em seguir o formato. Tente um modelo maior, ou aumente `MAX_ERROS_FORMATO` para dar mais chances de correção. `IF` é diagnóstico, não crash.

**O agente termina com `TLE` por repetição**
Ele ficou chamando a mesma ferramenta com a mesma entrada. Isso costuma indicar que a observação devolvida não responde ao que ele precisava. Rode com `--verboso` no módulo 06, ou olhe o rastro em `saidas/ultima_execucao.json` no 05, para ver qual observação travou o raciocínio.

**No módulo 06, o supervisor aciona sempre o mesmo especialista**
O limite `MAX_REPETICAO_ESPECIALISTA` corta isso e encerra com `TLE`. Se acontecer sempre, olhe a instrução do handoff: costuma ser um pedido que aquele especialista não tem ferramenta para cumprir.

**Os resultados não batem com os meus**
Eles não vão bater exatamente. O modelo muda, o provedor muda a versão, e mesmo com temperatura zero a saída varia entre execuções. O que deve se manter é o comportamento estrutural: as ferramentas recusarem entradas perigosas, o flush acontecer sob pressão, o redator não alcançar as ferramentas de coleta e as salvaguardas dispararem nos casos positivos e ficarem quietas nos negativos. Isso tudo é verificado pelos testes offline, que são determinísticos.
