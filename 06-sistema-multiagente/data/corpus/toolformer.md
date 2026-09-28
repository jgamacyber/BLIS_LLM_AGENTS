# Toolformer: modelos que aprendem a usar ferramentas sozinhos

O artigo "Toolformer: Language Models Can Teach Themselves to Use Tools", de Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda e Thomas Scialom, foi publicado em 2023 pela Meta AI Research e pela Universitat Pompeu Fabra.

## O problema

Modelos de linguagem exibem habilidades notáveis a partir de poucos exemplos, mas paradoxalmente têm dificuldade com funcionalidades básicas como aritmética ou consulta factual, onde modelos muito menores e mais simples se saem melhor. Eles também não acessam informação atualizada sobre eventos recentes e tendem a alucinar fatos.

## A representação de uma chamada

Cada chamada de API é representada como uma tupla composta pelo nome da API e a entrada correspondente. Dada uma chamada e seu resultado, os autores definem sequências linearizadas com tokens especiais que marcam o início e o fim da chamada, e um separador antes do resultado. Isso permite inserir chamadas de API em qualquer ponto de um texto.

## O critério de utilidade

O ponto mais interessante do método é como decidir quais chamadas valem a pena, sem anotação humana.

Os autores comparam duas perdas. A primeira é a perda de entropia cruzada ponderada quando o modelo recebe como prefixo tanto a chamada quanto o resultado dela. A segunda é o mínimo entre a perda de não fazer chamada nenhuma e a perda de fazer a chamada sem receber o resultado.

Intuitivamente, uma chamada de API é útil ao modelo se fornecer tanto a entrada quanto a saída dela torna mais fácil predizer os tokens seguintes, em comparação com não receber a chamada de forma alguma ou receber apenas sua entrada. Só são mantidas as chamadas em que a diferença entre as duas perdas supera um limiar de filtragem.

Esse critério é auto-supervisionado: não depende do que humanos acham útil, mas do que efetivamente reduz a perda do modelo.

## As ferramentas

O artigo incorpora um sistema de perguntas e respostas, uma calculadora, um mecanismo de busca na Wikipédia, um sistema de tradução e um calendário.

## O resultado

O Toolformer, baseado em um GPT-J pré-treinado com 6,7 bilhões de parâmetros, alcança resultados zero-shot substancialmente melhores em diversas tarefas, superando claramente um GPT-3 com 175 bilhões de parâmetros, sem sacrificar suas capacidades centrais de modelagem de linguagem.
