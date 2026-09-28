# CoALA: arquiteturas cognitivas para agentes de linguagem

O artigo "Cognitive Architectures for Language Agents", de Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan e Thomas L. Griffiths, da Princeton University, foi publicado na Transactions on Machine Learning Research em 2024.

## A proposta

Os autores recorrem à história das arquiteturas cognitivas e da inteligência artificial simbólica para propor um arcabouço conceitual que organiza os agentes de linguagem existentes e orienta o projeto de novos. O CoALA descreve um agente de linguagem com componentes modulares de memória, um espaço de ações estruturado para interagir com memórias internas e ambientes externos, e um procedimento generalizado de tomada de decisão para escolher ações.

## Os módulos de memória

Modelos de linguagem são sem estado: não persistem informação entre chamadas. Agentes de linguagem, ao contrário, armazenam e mantêm informação internamente para interação em múltiplos passos.

A memória de trabalho mantém a informação ativa e prontamente disponível para o ciclo de decisão atual, como variáveis simbólicas. Inclui entradas perceptuais, conhecimento ativo gerado por raciocínio ou recuperado da memória de longo prazo, e outra informação central carregada do ciclo anterior, como as metas ativas do agente. É o hub central que conecta os diferentes componentes do agente.

A memória episódica guarda experiências de ciclos de decisão anteriores. Pode conter pares de entrada e saída de treino, fluxos de eventos históricos, trajetórias de jogo de episódios anteriores, ou outras representações das experiências do agente. Durante o planejamento, esses episódios podem ser recuperados para a memória de trabalho para apoiar o raciocínio. O agente também pode escrever novas experiências da memória de trabalho para a episódica, como forma de aprendizado.

A memória semântica guarda o conhecimento do agente sobre o mundo e sobre si mesmo. Métodos de geração aumentada por recuperação podem ser vistos como recuperação de uma memória semântica.

A memória procedural contém tanto os pesos do próprio modelo quanto o código do agente.

## O espaço de ações

As ações se dividem em externas e internas.

As ações externas interagem com ambientes externos, por exemplo controlar um robô, comunicar-se com um humano ou navegar em um site. São chamadas de grounding, ou aterrissagem.

As ações internas interagem com as memórias internas. Dependendo de qual memória é acessada e de a operação ser de leitura ou escrita, dividem-se em três tipos. Recuperação é leitura da memória de longo prazo. Raciocínio é atualização da memória de trabalho de curto prazo com o modelo de linguagem. Aprendizado é escrita na memória de longo prazo.

As ações de raciocínio e recuperação são usadas para apoiar o planejamento.

## O ciclo de decisão

O procedimento de decisão executa um ciclo em laço com o ambiente externo. Em cada ciclo, o agente usa recuperação e raciocínio para planejar, propondo e avaliando ações candidatas de aprendizado ou de grounding. A melhor ação é então selecionada e executada. Uma observação pode ser feita, e o ciclo recomeça.

O subprocesso de planejamento tem três etapas: proposta, avaliação e seleção.
