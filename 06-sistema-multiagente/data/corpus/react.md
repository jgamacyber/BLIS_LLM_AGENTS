# ReAct: raciocinar e agir de forma entrelaçada

O artigo "ReAct: Synergizing Reasoning and Acting in Language Models", de Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan e Yuan Cao, foi publicado no ICLR 2023 pela Princeton University e pelo Google Research, Brain team.

## A ideia

O espaço de ações do agente é aumentado com a linguagem. Formalmente, o novo espaço é a união do espaço de ações original com o espaço da linguagem. Uma ação no espaço da linguagem é chamada de pensamento, ou traço de raciocínio. Ela não afeta o ambiente externo e portanto não gera nenhuma observação de retorno. Seu efeito é apenas atualizar o contexto do agente, compondo informação útil para apoiar o raciocínio ou a ação seguinte.

## Por que alternar

O raciocínio em cadeia puro é um processo fechado: o modelo usa apenas suas representações internas e não se ancora no mundo externo. Isso limita sua capacidade de reagir ou atualizar conhecimento, e leva a alucinação de fatos e propagação de erro ao longo da cadeia.

Agir sem raciocinar, por outro lado, produz sequências de ações sem plano. O agente não decompõe metas nem rastreia progresso.

O ReAct alterna os dois. O raciocínio ajuda a induzir, rastrear e atualizar planos de ação, além de tratar exceções. As ações permitem interagir com fontes externas, como bases de conhecimento, e trazer informação nova para dentro do raciocínio.

## Tipos de pensamento

As trajetórias boas analisadas pelos autores contêm pensamentos que decompõem metas, injetam conhecimento de senso comum relevante para a tarefa, extraem as partes importantes das observações, rastreiam o progresso e transitam entre planos de ação, e tratam exceções ajustando o plano.

## Espaço de ações do experimento

Para perguntas e respostas sobre a Wikipédia, os autores projetaram três ações: search de uma entidade, que devolve as cinco primeiras frases da página correspondente; lookup de uma string, que devolve a próxima frase da página contendo aquela string, simulando o Ctrl+F do navegador; e finish com a resposta, que encerra a tarefa.

## Resultados

Em ALFWorld e WebShop, o ReAct supera métodos de imitação e de aprendizado por reforço treinados com milhares de instâncias, com melhora absoluta de 34% e 10% respectivamente, usando apenas um ou dois exemplos no prompt.

Os autores destacam quatro propriedades. É intuitivo de projetar, porque anotadores humanos simplesmente escrevem seus pensamentos sobre as ações que tomaram. É geral e flexível, funcionando para espaços de ação e necessidades de raciocínio distintos. É performático e robusto a variações do prompt. E é alinhado ao humano e controlável, porque o traço de raciocínio é inspecionável e um humano pode corrigir o comportamento do agente editando um pensamento no meio da execução.

A melhor abordagem geral combina ReAct e chain-of-thought, permitindo usar tanto o conhecimento interno quanto a informação obtida externamente durante o raciocínio.
