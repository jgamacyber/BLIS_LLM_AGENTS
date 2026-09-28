# MemGPT: memória hierárquica inspirada em sistemas operacionais

O artigo "MemGPT: Towards LLMs as Operating Systems", de Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica e Joseph E. Gonzalez, foi publicado em 2024 pela University of California, Berkeley.

## O problema

A janela de contexto dos modelos é fixa e limitada. Estender o comprimento de contexto dos transformers custa caro, porque a autoatenção tem custo quadrático. E mesmo quando o contexto é maior, pesquisas recentes mostram que os modelos têm dificuldade de usar efetivamente esse contexto adicional.

## A solução

Os autores propõem gerenciamento de contexto virtual, uma técnica inspirada nos sistemas de memória hierárquica dos sistemas operacionais tradicionais, que criam a ilusão de memória estendida por meio de paginação entre a memória física e o disco.

A hierarquia tem dois níveis. O contexto principal, análogo à memória RAM, são os tokens do prompt. O contexto externo, análogo ao disco, é tudo que está fora da janela fixa do modelo. Dados fora do contexto precisam ser explicitamente movidos para o contexto principal para serem processados.

## As três seções do contexto principal

As instruções do sistema são somente leitura e estáticas. Contêm informação sobre o fluxo de controle, o uso pretendido dos diferentes níveis de memória, e instruções sobre como recuperar dados fora do contexto.

O working context, ou contexto de trabalho, é um bloco de tamanho fixo de texto não estruturado, com escrita apenas por chamadas de função. Guarda fatos-chave, preferências e informação importante sobre o usuário e sobre a persona que o agente adota.

A fila FIFO guarda o histórico rolante de mensagens, incluindo mensagens entre agente e usuário, mensagens de sistema e entradas e saídas de chamadas de função. O primeiro índice da fila guarda um resumo recursivo das mensagens que já foram despejadas.

## O contexto externo

O recall storage é o banco de dados de mensagens: guarda todas as mensagens que já passaram pela fila. A escrita é feita pelo gerenciador de fila e a leitura por chamadas de função.

O archival storage é um banco de leitura e escrita que guarda objetos de texto de comprimento arbitrário. Tanto leitura quanto escrita são feitas por chamadas de função.

## O gerenciador de fila

Quando os tokens do prompt excedem um limiar de aviso, tipicamente 70% da janela de contexto do modelo, o gerenciador insere uma mensagem de sistema avisando sobre a iminente remoção de dados. Esse aviso de pressão de memória permite que o modelo use as funções de memória para salvar informação importante da fila no working context ou no archival storage.

Quando os tokens do prompt excedem um limiar de descarga, tipicamente 100% da janela, o gerenciador despeja uma fração específica das mensagens, por exemplo 50% da janela, e gera um novo resumo recursivo usando o resumo existente e as mensagens despejadas. As mensagens despejadas saem do contexto mas continuam armazenadas indefinidamente no recall storage, acessíveis por chamadas de função.

## O executor de funções

O MemGPT orquestra a movimentação de dados entre contexto principal e externo por chamadas de função geradas pelo próprio modelo. As edições e recuperações de memória são inteiramente autodirigidas: o modelo decide quando mover itens entre contextos com base no contexto atual.

Erros de runtime, como tentar acrescentar ao contexto principal quando ele já está na capacidade máxima, são devolvidos ao processador. Esse laço de realimentação permite que o sistema aprenda com suas ações e ajuste o comportamento.

As funções podem ser chamadas com uma flag especial que pede que o controle retorne imediatamente ao processador após a conclusão. Isso permite encadear chamadas de função, executando várias em sequência antes de devolver controle ao usuário, o que viabiliza recuperação em múltiplos passos.

Os mecanismos de recuperação implementam paginação, para evitar que as chamadas de recuperação estourem a janela de contexto.
