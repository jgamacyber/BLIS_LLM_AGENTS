"""
Testes offline — validam a lógica sem chamar a API nem gastar créditos.

Cobrem o RAG (chunking, BM25, citação), o estado compartilhado, a divisão de
ferramentas por papel, os quatro detectores do CAMEL, os parsers do especialista
e do supervisor, e a verificação de respostas.

A bateria mais importante é a das salvaguardas, e não pelos casos positivos: os
dois casos negativos, em que uma resposta boa poderia ser confundida com falha,
são o que impede o sistema de ficar corrigindo agentes que estavam trabalhando.

    python testes_offline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import salvaguardas
from agentes import analisar_saida
from avaliacao import (
    Relatorio,
    ResultadoTarefa,
    Tarefa,
    _extrair_numeros,
    carregar_tarefas,
    citou_fonte,
    eh_abstencao,
    verificar,
)
from estado import EstadoCompartilhado
from ferramentas import (
    FERRAMENTAS_POR_PAPEL,
    RegistroFerramentas,
    registro_para,
)
from rag import RAG, dividir_por_secao, obter_rag, tokenizar
from supervisor import PERFIS, analisar_decisao, montar_time

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 68)


# --------------------------------------------------------------------------- #
# 1. Chunking
# --------------------------------------------------------------------------- #

DOC_EXEMPLO = """# Título do documento

Parágrafo de abertura, antes de qualquer seção.

## Primeira seção

Conteúdo da primeira seção, curto.

## Segunda seção

Conteúdo da segunda seção, também curto.
"""


def testar_chunking() -> None:
    secao("1. Chunking por seção")

    chunks = dividir_por_secao(DOC_EXEMPLO, "exemplo")
    checar(len(chunks) == 3, f"abertura mais duas seções dão 3 chunks ({len(chunks)})")
    checar(all(c.titulo == "Título do documento" for c in chunks),
           "todo chunk carrega o título do documento")

    secoes = [c.secao for c in chunks]
    checar(secoes == ["", "Primeira seção", "Segunda seção"],
           f"as seções são reconhecidas na ordem: {secoes}")

    checar(chunks[1].fonte == "exemplo#Primeira seção",
           "a fonte junta documento e seção, e é o que vira citação")
    checar(chunks[0].fonte == "exemplo",
           "trecho sem seção cita só o documento")

    # Seção longa vira janelas com sobreposição.
    longo = "# T\n\n## Longa\n\n" + ("palavra " * 400)
    janelas = dividir_por_secao(longo, "longo", tamanho=300, sobreposicao=60)
    checar(len(janelas) > 1, f"seção longa é dividida em janelas ({len(janelas)})")
    checar(all(len(j.texto) <= 300 for j in janelas),
           "nenhuma janela passa do tamanho pedido")

    # A sobreposição precisa existir de fato, senão uma regra na emenda se perde.
    fim_da_primeira = janelas[0].texto[-40:]
    checar(fim_da_primeira.strip()[:20] in janelas[1].texto or
           janelas[1].texto[:40].strip()[:20] in janelas[0].texto,
           "janelas vizinhas se sobrepõem")

    checar(dividir_por_secao("", "vazio") == [], "documento vazio não gera chunk")

    checar(tokenizar("A memória do MemGPT, com acentuação!") == ["memoria", "memgpt",
                                                                "acentuacao"],
           "tokenizador remove acentos, pontuação e stopwords")


# --------------------------------------------------------------------------- #
# 2. RAG
# --------------------------------------------------------------------------- #


def testar_rag() -> None:
    secao("2. RAG como ferramenta")

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        (pasta / "a.md").write_text(
            "# Papéis\n\n## Inversão\n\nO assistente vira usuário e passa a "
            "instruir em vez de executar.\n", encoding="utf-8")
        (pasta / "b.md").write_text(
            "# Memória\n\n## Paginação\n\nO sistema move dados entre o contexto "
            "e o disco conforme a pressão.\n", encoding="utf-8")

        rag = RAG(pasta)
        checar(len(rag) == 2, "indexou os dois trechos")
        checar(rag.documentos() == ["a", "b"], "lista os documentos em ordem")
        checar(rag.titulo_de("b") == "Memória", "recupera o título do documento")

        top = rag.consultar("paginação de memória entre contexto e disco")
        checar(bool(top) and top[0][0].doc == "b", "a consulta acha o documento certo")
        checar(top[0][1] > 0, "o score é positivo")

        checar(rag.consultar("zzzz qqqq inexistente") == [],
               "consulta sem casamento devolve lista vazia")
        checar(rag.consultar("") == [], "consulta vazia devolve lista vazia")

        texto = rag.formatar(top)
        checar("[fonte: b#Paginação]" in texto,
               "a formatação traz a fonte, que é o que o agente copia no fato")

        vazio = rag.formatar([])
        checar("não invente" in vazio.lower(),
               "sem resultado, a ferramenta manda o agente admitir que não sabe")

    # O corpus real do módulo.
    real = obter_rag()
    checar(len(real.documentos()) == 6, "o corpus do módulo tem 6 documentos")
    for doc in ("camel", "autogen", "react", "memgpt", "coala", "toolformer"):
        checar(doc in real.documentos(), f"o corpus inclui {doc}")

    top = real.consultar("quais são os modos de falha do role-playing", top_k=1)
    checar(bool(top) and top[0][0].doc == "camel",
           "pergunta sobre modos de falha cai no documento do CAMEL")

    top = real.consultar("como o gerente escolhe quem fala em seguida", top_k=1)
    checar(bool(top) and top[0][0].doc == "autogen",
           "pergunta sobre seleção de falante cai no documento do AutoGen")


# --------------------------------------------------------------------------- #
# 3. Estado compartilhado
# --------------------------------------------------------------------------- #


def testar_estado() -> None:
    secao("3. Estado compartilhado")

    e = EstadoCompartilhado("tarefa de teste")
    e.rodada = 1

    e.registrar_fato("o pedido 4471 soma R$ 318,00", "analista", "pedidos.csv")
    checar(len(e.fatos) == 1, "o fato foi registrado")
    checar(e.fatos[0].autor == "analista", "o fato guarda o autor")
    checar(e.fatos[0].fonte == "pedidos.csv", "o fato guarda a fonte")
    checar(e.fatos[0].rodada == 1, "o fato guarda a rodada")

    resposta = e.registrar_fato("O PEDIDO 4471 SOMA R$ 318,00", "redator", "outra")
    checar(len(e.fatos) == 1, "fato repetido, mesmo com outra caixa, não duplica")
    checar("já registrado" in resposta, "a resposta explica que já existia")

    checar("erro" in e.registrar_fato("   ", "analista").lower(),
           "fato vazio é recusado")

    e.registrar_fato("o corpus tem 6 documentos", "pesquisador", "camel#Resumo")
    achados = e.buscar_fatos("pedido")
    checar(len(achados) == 1, "a busca por fatos filtra por termo")
    checar(len(e.buscar_fatos()) == 2, "busca sem termo devolve tudo")

    e.escrever_artefato("resumo", "texto final", "redator")
    checar(e.ler_artefato("resumo") == "texto final", "artefato é lido de volta")
    checar("não existe" in e.ler_artefato("fantasma"),
           "artefato inexistente devolve erro explicativo")
    checar("Disponíveis" in e.ler_artefato("fantasma"),
           "o erro lista os artefatos que existem")

    e.registrar_handoff("supervisor", "analista", "levante os pedidos")
    e.registrar_handoff("supervisor", "redator", "consolide")
    e.registrar_handoff("supervisor", "analista", "confira o total")
    checar(e.especialistas_acionados() == ["analista", "redator"],
           "especialistas acionados vêm sem repetição e na ordem")

    st = e.estatisticas()
    checar(st["fatos"] == 2 and st["fatos_com_fonte"] == 2,
           "as estatísticas contam fatos e procedência")
    checar(st["handoffs"] == 3, "as estatísticas contam os handoffs")

    resumo = e.resumo()
    checar("pedidos.csv" in resumo, "o resumo mostra a fonte dos fatos")
    checar("TAREFA" in resumo, "o resumo mostra a tarefa")

    # O resumo é o que vai ao supervisor a cada rodada: não pode carregar tudo.
    for i in range(12):
        e.registrar_mensagem("analista", "supervisor", f"mensagem {i}")
    resumo = e.resumo(ultimas=4)
    checar(resumo.count("mensagem ") == 4,
           "o resumo leva só as últimas mensagens, não o histórico inteiro")
    checar(len(e.historico) == 12, "o histórico completo continua no estado")

    d = e.como_dict()
    checar(d["estatisticas"]["fatos"] == 2 and len(d["historico"]) == 12,
           "a serialização preserva fatos e histórico")


# --------------------------------------------------------------------------- #
# 4. Divisão de ferramentas por papel
# --------------------------------------------------------------------------- #


def testar_divisao_de_papeis() -> None:
    secao("4. Divisão de papéis")

    e = EstadoCompartilhado("tarefa")

    pesquisador = registro_para("pesquisador", e)
    analista = registro_para("analista", e)
    redator = registro_para("redator", e)

    checar("consultar_documentos" in pesquisador, "o pesquisador consulta o corpus")
    checar("calculadora" not in pesquisador, "o pesquisador não calcula")
    checar("ler_arquivo" not in pesquisador, "o pesquisador não lê arquivos")

    checar("ler_arquivo" in analista and "calculadora" in analista,
           "o analista lê arquivos e calcula")
    checar("consultar_documentos" not in analista,
           "o analista não alcança o corpus")

    checar(not any(n in redator for n in
                   ("consultar_documentos", "ler_arquivo", "calculadora")),
           "o redator não tem nenhuma ferramenta de coleta")
    checar("ler_fatos" in redator,
           "o redator lê o quadro compartilhado, que é a sua única fonte")

    for papel in FERRAMENTAS_POR_PAPEL:
        registro = registro_para(papel, e)
        for nome in ("registrar_fato", "ler_fatos", "escrever_artefato",
                     "ler_artefato"):
            checar(nome in registro, f"{papel} tem a ferramenta de estado {nome}")

    try:
        registro_para("inexistente", e)
        checar(False, "papel desconhecido levanta erro")
    except ValueError as erro:
        checar("Conhecidos" in str(erro),
               "o erro de papel desconhecido lista os papéis válidos")


def testar_ferramentas_de_estado() -> None:
    secao("5. Ferramentas de estado")

    e = EstadoCompartilhado("tarefa")
    analista = registro_para("analista", e, autor="analista")

    analista.executar("registrar_fato",
                      "o pedido 4477 soma R$ 900,00 | fonte: pedidos.csv")
    checar(len(e.fatos) == 1, "registrar_fato escreve no quadro")
    checar(e.fatos[0].fonte == "pedidos.csv", "a sintaxe '| fonte:' é entendida")
    checar(e.fatos[0].autor == "analista",
           "o autor vem do registro, não da entrada do modelo")

    # Um agente não pode assinar em nome de outro.
    analista.executar("registrar_fato",
                      "fato forjado | fonte: nenhuma  (autor: pesquisador)")
    checar(all(f.autor == "analista" for f in e.fatos),
           "nenhum fato sai assinado por outro agente")

    analista.executar("registrar_fato", "fato sem fonte declarada")
    sem_fonte = [f for f in e.fatos if not f.fonte]
    checar(len(sem_fonte) == 1,
           "fato sem fonte é aceito, mas fica marcado como sem procedência")
    checar(e.estatisticas()["fatos_com_fonte"] == 2,
           "a contagem de procedência separa os dois casos")

    leitura = analista.executar("ler_fatos", "pedido").como_observacao()
    checar("4477" in leitura, "ler_fatos encontra o que foi registrado")

    erro = analista.executar("escrever_artefato", "sem separador")
    checar(not erro.sucesso and "nome | conte" in erro.erro,
           "escrever_artefato sem separador explica o formato")

    ok = analista.executar("escrever_artefato", "nota | conteúdo da nota")
    checar(ok.sucesso and e.ler_artefato("nota") == "conteúdo da nota",
           "escrever_artefato salva com o formato certo")

    # As ferramentas herdadas do módulo anterior continuam seguras.
    checar(not analista.executar("ler_arquivo", "../../config.py").sucesso,
           "leitura fora do sandbox é recusada")
    checar(not analista.executar("calculadora", '__import__("os")').sucesso,
           "a calculadora recusa construção perigosa")
    checar(analista.executar("calculadora", "(20 * 45,00) * 0,85").como_observacao()
           == "765", "a calculadora entende a vírgula decimal brasileira")

    vazio = RegistroFerramentas()
    checar(not vazio.executar("qualquer", "x").sucesso,
           "registro vazio recusa qualquer ferramenta")


# --------------------------------------------------------------------------- #
# 6. Salvaguardas
# --------------------------------------------------------------------------- #


def testar_salvaguardas() -> None:
    secao("6. Salvaguardas do CAMEL")

    ctx_sem_trabalho = salvaguardas.Contexto(
        agente="analista", instrucao_recebida="calcule o total do pedido 4471")

    # 1. Inversão de papéis
    for mensagem in ("Você precisa buscar o arquivo e me informe o total.",
                     "HANDOFF: pesquisador | veja no corpus",
                     "O que eu devo fazer agora?"):
        d = salvaguardas.detectar_inversao_papeis(mensagem, ctx_sem_trabalho)
        checar(d is not None and d.tipo == "inversao_papeis",
               f"detecta inversão de papéis: {mensagem[:42]}")

    d = salvaguardas.detectar_inversao_papeis(
        "Qual o prazo? Devo considerar o frete? E o desconto?", ctx_sem_trabalho)
    checar(d is not None, "só perguntas, sem trabalho, também é inversão")

    # 2. Repetição da instrução
    d = salvaguardas.detectar_repeticao_instrucao(
        "Devo calcular o total do pedido 4471, entendido.", ctx_sem_trabalho)
    checar(d is not None and d.tipo == "repeticao_instrucao",
           "detecta a instrução devolvida parafraseada")

    com_trabalho = salvaguardas.Contexto(
        agente="analista", instrucao_recebida="calcule o total do pedido 4471",
        ferramentas_usadas=["calculadora"], fatos_registrados=1)
    d = salvaguardas.detectar_repeticao_instrucao(
        "Calculei o total do pedido 4471: R$ 318,00.", com_trabalho)
    checar(d is None,
           "vocabulário parecido com a instrução não é repetição quando houve trabalho")

    # Sem trabalho feito, mas trazendo conteúdo que não estava na instrução:
    # não é repetição. É o falso positivo que mais atrapalharia na prática.
    for legitima in ("O total é R$ 318,00, conforme já apurado pelo analista no quadro.",
                     "Não consigo: minhas ferramentas não alcançam a planilha de pedidos."):
        d = salvaguardas.detectar_repeticao_instrucao(legitima, ctx_sem_trabalho)
        checar(d is None, f"conteúdo novo não é repetição: {legitima[:44]}")

    d = salvaguardas.detectar_repeticao_instrucao("qualquer coisa",
                                                  salvaguardas.Contexto(agente="x"))
    checar(d is None, "sem instrução recebida não há como detectar repetição")

    checar(salvaguardas.cobertura("o total do pedido", "o total do pedido 4471") > 0.7,
           "cobertura mede quanto da instrução a mensagem repete")
    checar(salvaguardas.novidade("o total é 318 reais agora", "o total") > 0.5,
           "novidade mede quanto a mensagem acrescenta")
    checar(salvaguardas.cobertura("", "x") == 0.0, "cobertura com texto vazio é zero")

    # 3. Resposta evasiva
    for mensagem in ("Vou calcular o total agora mesmo.",
                     "Pretendo consultar os documentos em seguida.",
                     "Em breve eu retorno com o número."):
        d = salvaguardas.detectar_resposta_evasiva(mensagem, ctx_sem_trabalho)
        checar(d is not None and d.tipo == "resposta_evasiva",
               f"detecta evasiva: {mensagem[:40]}")

    d = salvaguardas.detectar_resposta_evasiva(
        "Vou calcular: 3 x 98,50 mais 22,50 dá R$ 318,00. [fonte: pedidos.csv]",
        com_trabalho)
    checar(d is None, "prometer e entregar na mesma mensagem não é evasiva")

    d = salvaguardas.detectar_resposta_evasiva("ok", ctx_sem_trabalho)
    checar(d is not None, "resposta curta demais e sem trabalho é evasiva")

    # Curto não é sinônimo de vazio: o que caracteriza a evasiva é a falta de
    # conteúdo verificável. Sem esta distinção, o detector reprova respostas
    # certas e enxutas, que é o falso positivo mais caro dos quatro.
    for curta_mas_cheia in ("são 2 pedidos [fonte: pedidos.csv]",
                            "R$ 765,00",
                            "Não alcanço a planilha com minhas ferramentas."):
        d = salvaguardas.detectar_resposta_evasiva(curta_mas_cheia, ctx_sem_trabalho)
        checar(d is None,
               f"resposta curta com conteúdo não é evasiva: {curta_mas_cheia[:36]}")

    # 4. Laço infinito
    d = salvaguardas.detectar_laco("Muito obrigado! Fico à disposição.",
                                   ctx_sem_trabalho)
    checar(d is not None and d.tipo == "laco_infinito",
           "detecta a troca de cortesias sem avanço")

    ctx_repetido = salvaguardas.Contexto(
        agente="analista",
        mensagens_anteriores=["O total do pedido 4471 é R$ 318,00 conforme apurado."])
    d = salvaguardas.detectar_laco(
        "O total do pedido 4471 é R$ 318,00, conforme apurado.", ctx_repetido)
    checar(d is not None, "repetir quase literalmente a própria fala é laço")

    d = salvaguardas.detectar_laco("Obrigado pelos dados. O total é R$ 318,00, "
                                   "calculado a partir de 3 itens de R$ 98,50 mais "
                                   "R$ 22,50 de frete. [fonte: pedidos.csv]",
                                   com_trabalho)
    checar(d is None, "agradecer junto com a entrega do resultado não é laço")

    # Ordem dos detectores e caso negativo completo.
    d = salvaguardas.avaliar(
        "O total do pedido 4471 é R$ 318,00. [fonte: pedidos.csv]", com_trabalho)
    checar(d is None, "resposta legítima não dispara nenhuma salvaguarda")

    d = salvaguardas.avaliar("Você deve buscar isso e me avise depois",
                             ctx_sem_trabalho)
    checar(d is not None and d.tipo == "inversao_papeis",
           "a inversão de papéis é testada antes da evasiva")

    checar(len(salvaguardas.TIPOS) == 4, "são quatro tipos, como no artigo")
    checar(salvaguardas.jaccard("a b c", "a b c") == 1.0, "jaccard de textos iguais")
    checar(salvaguardas.jaccard("a b", "c d") == 0.0, "jaccard de textos disjuntos")
    checar(salvaguardas.jaccard("", "a") == 0.0, "jaccard com texto vazio não quebra")


# --------------------------------------------------------------------------- #
# 7. Parsers
# --------------------------------------------------------------------------- #


def testar_parser_especialista() -> None:
    secao("7. Parser do especialista")

    a = analisar_saida("PASSO: preciso da planilha\nFERRAMENTA: ler_arquivo\n"
                       "ENTRADA: pedidos.csv")
    checar(a.ferramenta == "ler_arquivo" and a.entrada == "pedidos.csv",
           "forma de três linhas")
    checar(not a.erro_formato, "forma de três linhas é válida")

    a = analisar_saida("PASSO: calcular\nFERRAMENTA: calculadora[12 * 45.00]")
    checar(a.ferramenta == "calculadora" and a.entrada == "12 * 45.00",
           "forma compacta nome[entrada]")

    a = analisar_saida("PASSO: calcular\nFERRAMENTA: calculadora | 12 * 45.00")
    checar(a.ferramenta == "calculadora" and a.entrada == "12 * 45.00",
           "forma com barra vertical, como nos exemplos do catálogo")

    a = analisar_saida("RESPOSTA: o total é R$ 540,00 [fonte: pedidos.csv]")
    checar(a.concluiu and "540" in a.resposta, "bloco de resposta encerra a vez")

    a = analisar_saida("PASSO: ler\nFERRAMENTA: ler_arquivo\nENTRADA: pedidos.csv\n"
                       "RESULTADO: inventei que o total é 999\n"
                       "RESPOSTA: o total é 999")
    checar(a.ferramenta == "ler_arquivo", "mantém a ação antes do resultado alucinado")
    checar(not a.concluiu, "descarta a resposta construída sobre resultado inventado")

    checar(bool(analisar_saida("").erro_formato), "saída vazia é erro de formato")
    checar(bool(analisar_saida("só um texto solto").erro_formato),
           "texto sem rótulo é erro de formato")
    checar(analisar_saida("passo: x\nferramenta: calculadora\nentrada: 1+1").ferramenta
           == "calculadora", "rótulos em minúsculo são aceitos")


def testar_parser_supervisor() -> None:
    secao("8. Parser do supervisor")

    d = analisar_decisao("ANALISE: falta o total.\n"
                         "HANDOFF: analista | calcule o total do pedido 4471")
    checar(d.especialista == "analista", "extrai o especialista do handoff")
    checar(d.instrucao.startswith("calcule"), "extrai a instrução do handoff")
    checar(not d.encerra, "handoff não encerra a conversa")

    d = analisar_decisao("ANALISE: já temos tudo.\n"
                         "RESPOSTA FINAL: o total é R$ 318,00 [fonte: pedidos.csv]")
    checar(d.encerra and "318" in d.resposta_final, "extrai a resposta final")
    checar(not d.especialista, "resposta final não gera handoff")

    d = analisar_decisao("ANALISE: x\nHANDOFF: **analista**\n"
                         "confira os pedidos da Vetrix")
    checar(d.especialista == "analista",
           "aceita o nome em linha separada e limpa a formatação markdown")
    checar("Vetrix" in d.instrucao, "pega a instrução da linha seguinte")

    d = analisar_decisao("ANALISE: x\nHANDOFF: analista")
    checar(bool(d.erro_formato), "handoff sem instrução é erro de formato")

    d = analisar_decisao("Vou pensar um pouco sobre isso.")
    checar(bool(d.erro_formato), "sem bloco reconhecido é erro de formato")

    checar(bool(analisar_decisao("").erro_formato), "saída vazia é erro de formato")

    # A resposta final tem precedência: se o modelo produzir os dois blocos,
    # encerrar é o comportamento seguro, porque já existe resposta.
    d = analisar_decisao("ANALISE: x\nRESPOSTA FINAL: pronto\n"
                         "HANDOFF: analista | mais uma coisa")
    checar(d.encerra, "com os dois blocos, a resposta final prevalece")


# --------------------------------------------------------------------------- #
# 9. Verificação e relatório
# --------------------------------------------------------------------------- #


def testar_verificacao() -> None:
    secao("9. Verificação de respostas")

    numeros = _extrair_numeros("O 4472 sai por R$ 459,00 e o 4477 por R$ 765,00, "
                               "somando R$ 1.224,00.")
    checar(459.0 in numeros and 765.0 in numeros and 1224.0 in numeros,
           f"extrai todos os números do parágrafo: {numeros}")

    checar(_extrair_numeros("O total é R$ 318,00.")[0] == 318.0,
           "o ponto final da frase não vira separador decimal")
    checar(_extrair_numeros("sem número") == [], "texto sem número dá lista vazia")

    t = Tarefa(id="x", tarefa="", verificacao="numerico", esperado="1224")
    checar(verificar("O 4472 dá R$ 459,00, o 4477 dá R$ 765,00, total R$ 1.224,00.", t),
           "acha o alvo entre os números do parágrafo")
    checar(not verificar("O total é R$ 459,00.", t),
           "recusa quando o alvo não aparece")

    t_cont = Tarefa(id="y", tarefa="", verificacao="contem",
                    esperado=["invers", "repet", "evasiv", "laco"])
    checar(verificar("São inversão de papéis, repetição da instrução, respostas "
                     "evasivas e laço infinito.", t_cont),
           "exige que todos os termos apareçam")
    checar(not verificar("São inversão de papéis e repetição da instrução.", t_cont),
           "resposta parcial não passa")

    for frase in ("Não temos essa informação em nenhuma ferramenta.",
                  "Nenhum especialista alcança esse dado.",
                  "Não é possível responder com as fontes disponíveis."):
        checar(eh_abstencao(frase), f"reconhece abstenção: {frase[:38]}")

    checar(not eh_abstencao("O faturamento foi de R$ 4,2 milhões."),
           "número inventado não é abstenção")

    t_abs = Tarefa(id="z", tarefa="", verificacao="abstencao", esperado="")
    checar(verificar("Não temos esse dado no quadro.", t_abs),
           "abstenção correta conta como acerto")
    checar(not verificar("Foram R$ 4,2 milhões.", t_abs),
           "inventar em grupo conta como erro")

    # Procedência.
    for frase in ("O total é R$ 318,00 [fonte: pedidos.csv]",
                  "Segundo o artigo do CAMEL, são quatro modos de falha.",
                  "Com base nas notas da equipe, o desconto é de 15%.",
                  "Conforme camel#Os quatro modos de falha, são quatro."):
        checar(citou_fonte(frase), f"reconhece citação: {frase[:40]}")

    checar(not citou_fonte("O total é R$ 318,00."),
           "resposta sem procedência não conta como citada")


def testar_relatorio() -> None:
    secao("10. Relatório")

    r1 = ResultadoTarefa(
        tarefa_id="m01", tarefa="t", acertou=True, termino="complete", rodadas=3,
        tokens=1000, especialistas_usados=["analista"],
        especialistas_esperados=["analista"], handoffs=1, fatos=2,
        fatos_com_fonte=2, citou_fonte=True, exigia_fonte=True)
    r2 = ResultadoTarefa(
        tarefa_id="m06", tarefa="t", acertou=False, termino="TLE", rodadas=8,
        tokens=5000, especialistas_usados=["analista"],
        especialistas_esperados=["analista", "pesquisador", "redator"],
        handoffs=4, fatos=1, citou_fonte=False, exigia_fonte=True,
        deteccoes=["laco_infinito", "resposta_evasiva"])

    checar(r1.roteamento_correto and r1.roteamento_suficiente,
           "roteamento exato conta como correto e suficiente")
    checar(not r2.roteamento_suficiente,
           "faltando especialistas, o roteamento não é suficiente")

    r3 = ResultadoTarefa(tarefa_id="m03", tarefa="t",
                         especialistas_usados=["analista", "pesquisador"],
                         especialistas_esperados=["analista"])
    checar(r3.roteamento_suficiente and not r3.roteamento_correto,
           "acionar um especialista a mais é suficiente, mas não exato")

    rel = Relatorio(resultados=[r1, r2], modelo="m", timestamp="t")
    checar(abs(rel.taxa_sucesso - 0.5) < 1e-9, "taxa de acerto de 50%")
    checar(abs(rel.taxa_roteamento - 0.5) < 1e-9, "taxa de roteamento de 50%")
    checar(abs(rel.taxa_procedencia - 0.5) < 1e-9, "taxa de procedência de 50%")
    checar(rel.distribuicao_termino() == {"complete": 1, "TLE": 1},
           "distribuição de término")
    checar(rel.distribuicao_deteccoes()["laco_infinito"] == 1,
           "distribuição de detecções por tipo")
    checar(rel.uso_especialistas()["analista"] == 2,
           "uso de especialistas soma as tarefas")

    custo = rel.custo_de_coordenacao()
    checar(abs(custo["rodadas_por_tarefa"] - 5.5) < 1e-9, "rodadas por tarefa")
    checar(abs(custo["handoffs_por_tarefa"] - 2.5) < 1e-9, "handoffs por tarefa")

    vazio = Relatorio()
    checar(vazio.taxa_sucesso == 0.0 and vazio.custo_de_coordenacao() == {},
           "relatório vazio não quebra")


# --------------------------------------------------------------------------- #
# 11. Conjunto de tarefas e time
# --------------------------------------------------------------------------- #


def testar_tarefas_e_time() -> None:
    secao("11. Conjunto de avaliação e montagem do time")

    tarefas = carregar_tarefas()
    checar(len(tarefas) == 8, f"oito tarefas carregadas ({len(tarefas)})")

    ids = [t.id for t in tarefas]
    checar(len(set(ids)) == len(ids), "ids únicos")

    papeis = set(FERRAMENTAS_POR_PAPEL)
    for t in tarefas:
        for nome in t.especialistas_esperados:
            checar(nome in papeis, f"{t.id} espera um especialista que existe: {nome}")

    multi = [t for t in tarefas if len(t.especialistas_esperados) >= 2]
    checar(len(multi) >= 3,
           f"pelo menos três tarefas exigem dois ou mais especialistas ({len(multi)})")

    trio = [t for t in tarefas if len(t.especialistas_esperados) == 3]
    checar(len(trio) >= 1, "pelo menos uma tarefa exige o time inteiro")

    abstencoes = [t for t in tarefas if t.verificacao == "abstencao"]
    checar(len(abstencoes) == 1, "há exatamente uma tarefa de abstenção")
    checar(abstencoes[0].especialistas_esperados == [],
           "a tarefa de abstenção não espera especialista nenhum")

    checar(all(t.esperado for t in tarefas if t.verificacao == "numerico"),
           "toda tarefa numérica tem gabarito")

    # O time montado precisa bater com os papéis que as tarefas esperam.
    e = EstadoCompartilhado("t")
    time = montar_time(e)
    nomes = {a.nome for a in time}
    checar(nomes == {"pesquisador", "analista", "redator"},
           f"o time montado tem os três especialistas: {sorted(nomes)}")
    checar(len(PERFIS) == len(time), "um perfil para cada agente montado")

    for agente in time:
        ficha = agente.ficha()
        checar(agente.nome in ficha and "ferramentas:" in ficha,
               f"a ficha de {agente.nome} descreve nome e ferramentas")
        prompt = agente.montar_prompt("faça algo")
        checar("FORMATO OBRIGATÓRIO" in prompt and "faça algo" in prompt,
               f"o prompt de {agente.nome} traz formato e instrução")
        checar("NÃO coordena" in prompt,
               f"o prompt de {agente.nome} proíbe a inversão de papéis")


# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 68)
    print("  Testes offline — módulo 06, sistema multiagente")
    print("  Nenhuma chamada de API, nenhum custo.")
    print("=" * 68)

    testar_chunking()
    testar_rag()
    testar_estado()
    testar_divisao_de_papeis()
    testar_ferramentas_de_estado()
    testar_salvaguardas()
    testar_parser_especialista()
    testar_parser_supervisor()
    testar_verificacao()
    testar_relatorio()
    testar_tarefas_e_time()

    print("\n" + "=" * 68)
    if FALHAS:
        print(f"  {len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"    - {f}")
        print("=" * 68)
        return 1

    print("  Todos os testes passaram.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
