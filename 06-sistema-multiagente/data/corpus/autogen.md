# AutoGen: aplicações com múltiplos agentes que conversam entre si

O artigo "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", de Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger e Chi Wang, foi publicado em 2023 pela Microsoft Research em parceria com a Penn State University e a University of Washington.

## A tese

Uma aplicação complexa fica mais fácil de construir quando é decomposta em vários agentes que conversam, em vez de um único agente gigante com um prompt enorme. A conversa vira o mecanismo de composição: cada agente resolve a parte que sabe fazer e passa o resto adiante.

## Agente conversável

O bloco básico é o agente conversável. Ele tem três características. É conversável, porque sabe enviar e receber mensagens de outros agentes. É customizável, porque pode ser movido por um modelo de linguagem, por uma pessoa, por ferramentas ou por qualquer combinação das três. E tem estado, porque guarda o histórico das mensagens que trocou.

A interface unificada tem três métodos. O método de envio manda uma mensagem para outro agente. O método de recebimento aceita uma mensagem que chegou. E o método de geração de resposta produz a resposta a partir das mensagens recebidas, de acordo com a configuração daquele agente.

## Resposta automática

Quando um agente recebe uma mensagem, ele invoca automaticamente a geração de resposta e devolve o resultado ao remetente, a menos que uma condição de término seja satisfeita. É isso que faz a conversa andar sozinha, sem um laço externo orquestrando cada turno.

Dois parâmetros controlam o mecanismo. O limite de respostas automáticas consecutivas impede que dois agentes fiquem conversando para sempre. O modo de entrada humana define se uma pessoa é consultada a cada rodada, nunca, ou apenas quando a conversa está prestes a terminar.

## Programação por conversa

O artigo chama de programação por conversa a forma de escrever essas aplicações. Ela tem duas partes.

A primeira é a computação: o que um agente faz para produzir sua resposta. A segunda é o fluxo de controle: em que ordem essas computações acontecem. No AutoGen, o fluxo de controle é dirigido pela conversa, ou seja, a próxima computação depende do conteúdo das mensagens trocadas, não de um roteiro fixo.

Essas duas partes podem ser programadas de dois jeitos, e os dois podem ser misturados. Em linguagem natural, escrevendo instruções no prompt do agente. E em linguagem de programação, com código Python que define condições de término, lógica de execução e funções de resposta.

## Funções de resposta registradas

A customização acontece registrando funções de resposta no agente. Cada função recebe as mensagens e devolve uma resposta ou nada. As funções são tentadas em ordem, e a primeira que devolver algo define a resposta daquele turno.

O framework já traz funções prontas baseadas em inferência do modelo, em execução de código ou de função, e em consulta a um humano. Quando nenhuma função é registrada, é essa lista padrão que vale.

## Conversa em grupo dinâmica

Para conversas com mais de dois participantes, o artigo descreve um gerente de conversa em grupo. Ele repete três passos. Primeiro escolhe dinamicamente quem fala em seguida, usando um prompt em estilo de interpretação de papéis que leva em conta o contexto da conversa e o papel de cada agente. Depois coleta a resposta de quem foi escolhido. Por fim transmite essa mensagem para todos os participantes.

A diferença em relação a uma ordem predefinida é que todos compartilham o mesmo contexto e a sequência de falas emerge da conversa, em vez de seguir uma hierarquia rígida.

## Aplicações avaliadas

Os autores demonstram o framework em resolução de problemas de matemática, conversa aumentada por recuperação, decisão em ambiente textual, programação com vários agentes, conversa em grupo dinâmica e xadrez conversacional.

Na conversa aumentada por recuperação, quando o agente que responde não encontra base suficiente no contexto recuperado, ele responde com um sinal de atualização de contexto, e o agente que faz a recuperação busca novos trechos. É o mesmo princípio de um agente admitir que não tem material em vez de inventar.
