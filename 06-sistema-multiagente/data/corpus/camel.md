# CAMEL: agentes que cooperam interpretando papéis

O artigo "CAMEL: Communicative Agents for 'Mind' Exploration of Large Scale Language Model Society", de Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin e Bernard Ghanem, foi publicado no NeurIPS 2023 pela KAUST.

## O problema

Conversas entre agentes de linguagem costumam precisar de uma pessoa no meio, corrigindo o rumo a cada turno. Os autores perguntam o que seria preciso para dois agentes cooperarem de forma autônoma até terminar uma tarefa, sem perder o alinhamento com a intenção original de quem pediu.

## Interpretação de papéis

A resposta proposta é a interpretação de papéis. Três agentes participam.

O especificador de tarefa recebe uma ideia vaga e a transforma em uma tarefa concreta e específica. Sem esse passo, a conversa entre os outros dois tende a ficar genérica.

O usuário de IA dá as instruções. Ele fala uma instrução por vez e acompanha o progresso.

O assistente de IA cumpre as instruções e devolve soluções. Ele não pergunta, não instrui e não decide o que fazer em seguida: isso é papel do usuário.

## Prompt de concepção

O mecanismo que faz os dois se manterem nos papéis é o que os autores chamam de prompt de concepção: os prompts de sistema entregues no início da conversa, que já contêm as regras de comportamento, o formato das mensagens e as condições de término. Depois disso, a conversa corre sozinha.

O assistente recebe regras explícitas. Nunca inverter papéis e nunca instruir o usuário. Começar toda resposta com a palavra "Solution:", a menos que a tarefa tenha terminado. Terminar toda resposta com "Next request.". E dar soluções específicas, em vez de prometer que vai fazer algo.

O usuário recebe o formato complementar: dar uma instrução por vez, no formato "Instruction:" seguido de "Input:", e sinalizar o fim com um marcador de tarefa concluída.

## Os quatro modos de falha

A parte mais útil do artigo para quem constrói sistemas é a lista de falhas observadas na prática. Elas não são bugs de implementação, são comportamentos recorrentes desse tipo de conversa.

O primeiro é a inversão de papéis. O assistente começa a dar instruções e a fazer perguntas, virando usuário. A conversa deixa de progredir porque ninguém está mais executando.

O segundo é o assistente repetir a instrução. Sem inverter papéis, ele devolve a instrução recebida parafraseada, sem nenhuma solução nova.

O terceiro são as respostas evasivas. O assistente responde no formato de promessa, dizendo que vai fazer algo, sem de fato fazer. A mensagem parece cooperativa e não entrega nada.

O quarto é o laço infinito de mensagens. Os dois agentes entram em uma troca educada e vazia, agradecendo um ao outro ou se despedindo repetidamente, e a tarefa nunca termina.

## O que isso implica para o projeto

As salvaguardas do artigo são regras dentro do prompt. Um sistema em produção precisa também detectar esses padrões do lado de fora, porque o prompt sozinho não garante obediência: é preciso olhar a mensagem produzida, decidir se ela é uma dessas quatro falhas e reagir, seja corrigindo o agente, seja encerrando a conversa com um motivo registrado.
