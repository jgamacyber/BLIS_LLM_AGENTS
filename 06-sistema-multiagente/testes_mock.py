"""
Testes com LLM simulado — validam o sistema inteiro sem gastar nada.

O dublê devolve saídas fixas, o que permite encenar de propósito cada situação
que interessa: o handoff que funciona, o especialista que inverte papéis, o
supervisor que chama quem não existe, o laço de repetição, o estouro de
rodadas e a falha de rede.

O teste mais importante é o do redator: ele não tem ferramenta de coleta, então
a única coisa que ele pode escrever é o que os outros registraram no quadro. Se
esse isolamento vazar, a divisão de papéis vira enfeite.

    python testes_mock.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import agentes as mod_agentes
import config
import supervisor as mod_supervisor
from agentes import Agente, Resposta
from avaliacao import Relatorio, Tarefa, avaliar_execucao
from estado import EstadoCompartilhado
from ferramentas import registro_para
from supervisor import Supervisor, Termino, montar_time

FALHAS: list[str] = []
CHAMADAS: list[dict] = []


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
# Dublê do cliente
# --------------------------------------------------------------------------- #


class ChatFalso:
    """
    Devolve respostas conforme um roteiro.

    O roteiro é uma lista de (marcador, resposta): a primeira entrada cujo
    marcador aparecer no prompt é usada. Isso permite escrever um roteiro que
    responde diferente para o supervisor e para cada especialista, que é como
    a conversa real acontece.
    """

    def __init__(self, roteiro: list[tuple[str, str]], padrao: str = "") -> None:
        self.roteiro = roteiro
        self.padrao = padrao or "RESPOSTA: sem roteiro para este prompt."
        self.usados: list[int] = []

    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        CHAMADAS.append({"prompt": prompt, **{k: v for k, v in kwargs.items()
                                              if k != "messages"}})

        texto = self.padrao
        for i, (marcador, resposta) in enumerate(self.roteiro):
            if i in self.usados:
                continue
            if marcador in prompt:
                texto = resposta
                self.usados.append(i)
                break

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=texto))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=50),
        )


class ClienteFalso:
    def __init__(self, roteiro: list[tuple[str, str]], padrao: str = "") -> None:
        self.chat = SimpleNamespace(completions=ChatFalso(roteiro, padrao))


class ClienteQuebrado:
    def __init__(self) -> None:
        def erro(**_kw):
            raise RuntimeError("falha de rede simulada")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=erro))


class SettingsFalsas:
    api_key = "sk-teste"
    model = "modelo/teste"
    temperature = 0.0
    max_tokens = 700
    max_rodadas = 6
    max_auto_resposta = 3
    max_passos_ferramenta = 4
    max_repeticao_especialista = 2
    max_salvaguardas = 3
    rag_top_k = 3
    rag_tamanho_chunk = 900
    rag_sobreposicao = 150
    tem_chave = True


def instalar(roteiro: list[tuple[str, str]], padrao: str = "") -> ClienteFalso:
    CHAMADAS.clear()
    cliente = ClienteFalso(roteiro, padrao)
    falso = lambda settings=None: cliente  # noqa: E731
    mod_agentes.get_client = falso
    mod_supervisor.get_client = falso
    config.get_client = falso
    return cliente


def montar(roteiro: list[tuple[str, str]], padrao: str = "",
           usar_salvaguardas: bool = True) -> tuple[Supervisor, EstadoCompartilhado]:
    instalar(roteiro, padrao)
    settings = SettingsFalsas()
    estado = EstadoCompartilhado()
    time = montar_time(estado, settings, verboso=False)
    sup = Supervisor(time, estado, settings, verboso=False,
                     usar_salvaguardas=usar_salvaguardas)
    return sup, estado


# Marcadores que identificam de quem é o prompt.
SUP = "Você é o supervisor"
ANALISTA = "Você é analista"
PESQUISADOR = "Você é pesquisador"
REDATOR = "Você é redator"


# --------------------------------------------------------------------------- #
# 1. Caminho feliz com handoff
# --------------------------------------------------------------------------- #


def testar_handoff_simples() -> None:
    secao("1. Handoff para um especialista e resposta final")

    sup, estado = montar([
        (SUP, "ANALISE: preciso dos dados da planilha.\n"
              "HANDOFF: analista | conte os pedidos em trânsito em pedidos.csv"),
        (ANALISTA, "PASSO: ler a planilha\nFERRAMENTA: ler_arquivo\n"
                   "ENTRADA: pedidos.csv"),
        (ANALISTA, "PASSO: registrar\nFERRAMENTA: registrar_fato\n"
                   "ENTRADA: há 2 pedidos em trânsito | fonte: pedidos.csv"),
        (ANALISTA, "RESPOSTA: são 2 pedidos em trânsito, os 4474 e 4476. "
                   "[fonte: pedidos.csv]"),
        (SUP, "ANALISE: o analista já apurou.\n"
              "RESPOSTA FINAL: são 2 pedidos em trânsito [fonte: pedidos.csv]"),
    ])
    e = sup.executar("Quantos pedidos estão em trânsito?")

    checar(e.termino == Termino.COMPLETO, "termina como complete")
    checar(e.sucesso, "a execução é bem-sucedida")
    checar("2 pedidos" in e.resposta, "a resposta final chega intacta")
    checar(e.especialistas_acionados == ["analista"],
           f"acionou só o analista: {e.especialistas_acionados}")
    checar(len(estado.handoffs) == 1, "um handoff registrado")
    checar(len(estado.fatos) == 1, "o fato foi para o quadro compartilhado")
    checar(estado.fatos[0].autor == "analista", "o fato tem o autor certo")
    checar(estado.fatos[0].fonte == "pedidos.csv", "o fato tem a fonte certa")
    checar(e.tokens_totais == 5 * 250, "soma os tokens de todas as chamadas")

    # O rastro precisa ser auditável do começo ao fim.
    d = e.como_dict()
    checar(d["termino"] == "complete", "o rastro guarda o término")
    checar(len(d["estado"]["historico"]) >= 3,
           "o histórico registra handoff, devolução e resposta final")


def testar_dois_especialistas() -> None:
    secao("2. Dois especialistas e o quadro como canal de dados")

    sup, estado = montar([
        (SUP, "ANALISE: preciso do corpus.\n"
              "HANDOFF: pesquisador | veja o que o CAMEL diz sobre modos de falha"),
        (PESQUISADOR, "PASSO: consultar\nFERRAMENTA: consultar_documentos\n"
                      "ENTRADA: modos de falha do role-playing"),
        (PESQUISADOR, "PASSO: registrar\nFERRAMENTA: registrar_fato\n"
                      "ENTRADA: o CAMEL lista quatro modos de falha | "
                      "fonte: camel#Os quatro modos de falha"),
        (PESQUISADOR, "RESPOSTA: são quatro modos de falha. "
                      "[fonte: camel#Os quatro modos de falha]"),
        (SUP, "ANALISE: agora consolidar.\n"
              "HANDOFF: redator | escreva o parágrafo com o que está no quadro"),
        (REDATOR, "PASSO: ler o quadro\nFERRAMENTA: ler_fatos\nENTRADA: "),
        (REDATOR, "RESPOSTA: o CAMEL lista quatro modos de falha "
                  "[fonte: camel#Os quatro modos de falha]"),
        (SUP, "ANALISE: pronto.\n"
              "RESPOSTA FINAL: o CAMEL lista quatro modos de falha "
              "[fonte: camel#Os quatro modos de falha]"),
    ])
    e = sup.executar("O que o CAMEL diz sobre modos de falha?")

    checar(e.termino == Termino.COMPLETO, "conclui com dois especialistas")
    checar(e.especialistas_acionados == ["pesquisador", "redator"],
           f"ordem dos especialistas preservada: {e.especialistas_acionados}")

    # O redator leu o quadro, e o que ele leu foi escrito pelo pesquisador.
    fala_redator = [p for p in CHAMADAS if REDATOR in p["prompt"]]
    checar(bool(fala_redator), "o redator foi chamado")
    checar("quatro modos de falha" in fala_redator[0]["prompt"],
           "o prompt do redator já traz o fato apurado pelo pesquisador")
    checar("camel#Os quatro modos" in fala_redator[0]["prompt"],
           "e traz junto a fonte do fato, não só o conteúdo")


def testar_isolamento_do_redator() -> None:
    secao("3. O redator não tem como coletar nada")

    sup, estado = montar([
        (SUP, "ANALISE: vou pedir direto ao redator.\n"
              "HANDOFF: redator | leia a planilha e conte os pedidos"),
        (REDATOR, "PASSO: ler a planilha\nFERRAMENTA: ler_arquivo\n"
                  "ENTRADA: pedidos.csv"),
        (REDATOR, "RESPOSTA: não tenho ferramenta para ler a planilha. "
                  "O quadro também não tem esse dado."),
        (SUP, "ANALISE: o redator não alcança.\n"
              "RESPOSTA FINAL: não temos esse dado disponível."),
    ])
    e = sup.executar("Quantos pedidos existem?")

    passos = [p for p in e.estado.historico if p.remetente == "redator"]
    checar(bool(passos), "o redator respondeu")

    prompt_redator = [c for c in CHAMADAS if REDATOR in c["prompt"]][0]["prompt"]
    checar("ler_arquivo" not in prompt_redator,
           "ler_arquivo não aparece no catálogo do redator")
    checar("consultar_documentos" not in prompt_redator,
           "consultar_documentos não aparece no catálogo do redator")
    checar("ler_fatos" in prompt_redator,
           "o redator vê as ferramentas do quadro compartilhado")
    checar(len(estado.fatos) == 0,
           "nada foi inventado e registrado como fato apurado")


# --------------------------------------------------------------------------- #
# 4. Funções de resposta registradas
# --------------------------------------------------------------------------- #


def testar_funcoes_de_resposta() -> None:
    secao("4. Funções de resposta registradas (AutoGen)")

    instalar([(ANALISTA, "RESPOSTA: o total é R$ 318,00 [fonte: pedidos.csv]")])
    estado = EstadoCompartilhado("t")
    agente = Agente(nome="analista", papel="analista", descricao="d",
                    ferramentas=registro_para("analista", estado, "analista"),
                    estado=estado, settings=SettingsFalsas())

    checar(len(agente.funcoes_resposta) == 3,
           "três funções padrão: término, cache e modelo")

    # Término: não gasta chamada.
    CHAMADAS.clear()
    r = agente.gerar_resposta("ENCERRAR: obrigado pela ajuda")
    checar(r.origem == "termino", "o marcador de término é respondido sem o modelo")
    checar(not CHAMADAS, "nenhuma chamada de API foi feita")

    # Modelo, depois cache.
    r1 = agente.gerar_resposta("calcule o total do pedido 4471")
    checar(r1.origem == "modelo", "a primeira vez passa pelo modelo")
    chamadas_apos_primeira = len(CHAMADAS)

    r2 = agente.gerar_resposta("calcule o total do pedido 4471")
    checar(r2.origem == "cache", "a instrução repetida vem do cache")
    checar(r2.conteudo == r1.conteudo, "o conteúdo é o mesmo")
    checar(len(CHAMADAS) == chamadas_apos_primeira,
           "o cache não gastou nenhuma chamada nova")
    checar(r2.tokens == 0, "a resposta do cache não contabiliza tokens")

    # Registro de uma função customizada, que é o ponto do mecanismo.
    def resposta_fixa(instrucao: str, agente: Agente) -> Resposta | None:
        if "data de hoje" in instrucao:
            return Resposta(agente=agente.nome, conteudo="responde sem modelo",
                            origem="regra")
        return None

    agente.registrar_resposta(resposta_fixa)
    checar(agente.funcoes_resposta[0] is resposta_fixa,
           "a função registrada entra na frente da lista")

    CHAMADAS.clear()
    r = agente.gerar_resposta("qual a data de hoje")
    checar(r.origem == "regra", "a função customizada tem precedência")
    checar(not CHAMADAS, "e resolve sem chamar o modelo")

    r = agente.gerar_resposta("outra coisa qualquer")
    checar(r.origem == "modelo",
           "quando a customizada devolve None, a próxima da lista assume")


# --------------------------------------------------------------------------- #
# 5. Salvaguardas em operação
# --------------------------------------------------------------------------- #


def testar_salvaguarda_corrige() -> None:
    secao("5. Salvaguarda detecta, corrige e o agente se recupera")

    sup, estado = montar([
        (SUP, "ANALISE: preciso do total.\n"
              "HANDOFF: analista | calcule o total do pedido 4477 com desconto"),
        # Primeira tentativa: inversão de papéis.
        (ANALISTA, "RESPOSTA: Você precisa buscar o desconto e me informe o valor."),
        # Depois da correção, trabalha.
        (ANALISTA, "PASSO: calcular\nFERRAMENTA: calculadora\n"
                   "ENTRADA: (20 * 45.00) * 0.85"),
        (ANALISTA, "PASSO: registrar\nFERRAMENTA: registrar_fato\n"
                   "ENTRADA: o pedido 4477 com desconto dá R$ 765,00 | "
                   "fonte: pedidos.csv"),
        (ANALISTA, "RESPOSTA: R$ 765,00 [fonte: pedidos.csv]"),
        (SUP, "ANALISE: pronto.\nRESPOSTA FINAL: R$ 765,00 [fonte: pedidos.csv]"),
    ])
    e = sup.executar("Qual o valor do pedido 4477 com desconto?")

    checar(len(e.deteccoes) == 1, f"uma detecção ({len(e.deteccoes)})")
    checar(e.deteccoes[0].tipo == "inversao_papeis", "detectou inversão de papéis")
    checar(e.termino == Termino.COMPLETO,
           "a conversa segue e conclui depois da correção")
    checar("765" in e.resposta, "a resposta final tem o valor correto")
    checar(len(estado.deteccoes) == 1, "a detecção fica registrada no estado")

    # A correção precisa chegar ao agente junto com a instrução original.
    segundo_prompt = [c for c in CHAMADAS if ANALISTA in c["prompt"]][1]["prompt"]
    checar("não é o supervisor" in segundo_prompt.lower() or
           "você é o especialista" in segundo_prompt.lower(),
           "a correção é injetada no prompt do agente")
    checar("INSTRUÇÃO ORIGINAL" in segundo_prompt,
           "a instrução original vai junto com a correção")


def testar_salvaguarda_encerra() -> None:
    secao("6. Salvaguardas repetidas encerram a conversa")

    # O analista só produz evasivas; o supervisor insiste em outro agente.
    sup, estado = montar(
        roteiro=[
            (SUP, "ANALISE: começar.\nHANDOFF: analista | calcule o total"),
            (SUP, "ANALISE: tentar outro.\nHANDOFF: pesquisador | veja no corpus"),
            (SUP, "ANALISE: tentar de novo.\nHANDOFF: redator | consolide"),
        ],
        padrao="RESPOSTA: Vou verificar isso e retorno em seguida.",
    )
    e = sup.executar("uma tarefa qualquer")

    checar(e.termino == Termino.SALVAGUARDA, "encerra por salvaguarda")
    checar(len(e.deteccoes) >= 3, f"pelo menos 3 detecções ({len(e.deteccoes)})")
    checar(all(d.tipo == "resposta_evasiva" for d in e.deteccoes),
           "todas as detecções são de resposta evasiva")
    checar("CAMEL" in e.detalhe_termino, "o detalhe nomeia a origem da regra")
    checar(estado.encerrado, "o estado fica marcado como encerrado")


def testar_sem_salvaguardas() -> None:
    secao("7. Modo sem salvaguardas, para comparação")

    roteiro = [
        (SUP, "ANALISE: começar.\nHANDOFF: analista | calcule o total"),
        (SUP, "ANALISE: desisto.\nRESPOSTA FINAL: não consegui apurar o valor."),
    ]
    sup, _estado = montar(roteiro, padrao="RESPOSTA: Vou verificar e retorno.",
                          usar_salvaguardas=False)
    e = sup.executar("uma tarefa")

    checar(not e.deteccoes, "nenhuma detecção quando as salvaguardas estão desligadas")
    checar(e.termino == Termino.COMPLETO,
           "a mesma conversa segue até o fim sem correção nenhuma")

    # Só uma chamada por handoff, porque não há ida e volta de correção.
    chamadas_analista = [c for c in CHAMADAS if ANALISTA in c["prompt"]]
    checar(len(chamadas_analista) == 1,
           f"sem correção, o agente é chamado uma vez ({len(chamadas_analista)})")


# --------------------------------------------------------------------------- #
# 8. Razões de término
# --------------------------------------------------------------------------- #


def testar_formato_invalido() -> None:
    secao("8. IF — supervisor fora do formato")

    sup, _estado = montar([], padrao="Acho que devíamos pensar melhor nisso.")
    e = sup.executar("uma tarefa")

    checar(e.termino == Termino.FORMATO_INVALIDO, "termina como IF")
    checar(e.rodadas == 3, f"desiste em 3 decisões malformadas ({e.rodadas})")
    checar(not e.sucesso, "IF não é sucesso")


def testar_especialista_invalido() -> None:
    secao("9. IA — especialista inexistente")

    # Errar uma vez e corrigir não pode matar a conversa.
    sup, _estado = montar([
        (SUP, "ANALISE: x.\nHANDOFF: jurista | avalie o contrato"),
        (SUP, "ANALISE: corrigindo.\nHANDOFF: analista | conte os pedidos"),
        (ANALISTA, "RESPOSTA: são 7 pedidos. [fonte: pedidos.csv]"),
        (SUP, "ANALISE: pronto.\nRESPOSTA FINAL: 7 pedidos [fonte: pedidos.csv]"),
    ])
    e = sup.executar("quantos pedidos existem")
    checar(e.termino == Termino.COMPLETO,
           "chamar um especialista inexistente uma vez não encerra a conversa")
    checar(e.especialistas_acionados == ["analista"],
           "o especialista inexistente não entra na lista de acionados")

    avisos = [m for m in e.estado.historico if m.tipo == "salvaguarda"]
    checar(bool(avisos) and "não existe" in avisos[0].conteudo,
           "o supervisor recebe de volta um aviso com os nomes válidos")

    # Insistir é que vira IA.
    sup, _estado = montar([], padrao="ANALISE: x.\nHANDOFF: jurista | avalie")
    e = sup.executar("uma tarefa")
    checar(e.termino == Termino.ESPECIALISTA_INVALIDO, "insistir no inexistente vira IA")
    checar(e.rodadas == 3, f"desiste em 3 tentativas ({e.rodadas})")


def testar_limite_rodadas_e_repeticao() -> None:
    secao("10. TLE — rodadas e repetição do mesmo especialista")

    # Mesmo especialista, sempre.
    sup, _estado = montar(
        [], padrao="ANALISE: mais uma vez.\nHANDOFF: analista | calcule de novo")
    e = sup.executar("uma tarefa")
    checar(e.termino == Termino.LIMITE_RODADAS, "repetir o mesmo especialista vira TLE")
    checar("vezes seguidas" in e.detalhe_termino,
           "o detalhe diz que foi repetição, não estouro de rodadas")
    checar(e.rodadas < SettingsFalsas.max_rodadas,
           "corta antes de gastar todas as rodadas")

    # Alternando especialistas, sem nunca fechar: estoura as rodadas.
    roteiro = []
    for i in range(10):
        alvo = ["analista", "pesquisador", "redator"][i % 3]
        roteiro.append((SUP, f"ANALISE: passo {i}.\nHANDOFF: {alvo} | "
                             f"apure o item {i}"))
    sup, _estado = montar(
        roteiro,
        padrao="PASSO: consultar\nFERRAMENTA: ler_fatos\nENTRADA: tudo")
    e = sup.executar("uma tarefa longa")
    checar(e.termino == Termino.LIMITE_RODADAS, "estourar as rodadas vira TLE")
    checar(e.rodadas == SettingsFalsas.max_rodadas,
           f"usou todas as rodadas ({e.rodadas})")
    checar("não fechou" in e.detalhe_termino, "o detalhe explica o motivo")


def testar_falha_de_rede() -> None:
    secao("11. erro — falha de infraestrutura")

    CHAMADAS.clear()
    quebrado = ClienteQuebrado()
    mod_supervisor.get_client = lambda settings=None: quebrado
    mod_agentes.get_client = lambda settings=None: quebrado

    estado = EstadoCompartilhado()
    settings = SettingsFalsas()
    sup = Supervisor(montar_time(estado, settings), estado, settings, verboso=False)
    e = sup.executar("uma tarefa")

    checar(e.termino == Termino.ERRO, "falha de rede vira término 'erro'")
    checar("falha de rede" in e.detalhe_termino, "guarda a mensagem do erro")
    checar(not e.sucesso, "erro não é sucesso")
    checar(e.estado is not None, "o estado volta preenchido mesmo com erro")

    # A falha no especialista não derruba o supervisor.
    instalar([(SUP, "ANALISE: x.\nHANDOFF: analista | calcule"),
              (SUP, "ANALISE: sem dados.\nRESPOSTA FINAL: não consegui apurar.")])
    cliente_sup = config.get_client()
    mod_supervisor.get_client = lambda settings=None: cliente_sup
    mod_agentes.get_client = lambda settings=None: quebrado

    estado = EstadoCompartilhado()
    sup = Supervisor(montar_time(estado, settings), estado, settings, verboso=False)
    e = sup.executar("uma tarefa")
    checar(e.termino == Termino.COMPLETO,
           "o supervisor segue e fecha mesmo com o especialista fora do ar")
    devolucoes = [m for m in estado.historico if m.tipo == "devolucao"]
    checar(bool(devolucoes) and "falha" in devolucoes[0].conteudo.lower(),
           "a devolução do especialista explica a falha em vez de fingir resultado")


# --------------------------------------------------------------------------- #
# 12. Laço de ferramentas do especialista
# --------------------------------------------------------------------------- #


def testar_laco_de_ferramentas() -> None:
    secao("12. Laço de ferramentas do especialista")

    # Nunca produz RESPOSTA: estoura os passos e devolve o que tem.
    sup, estado = montar(
        [(SUP, "ANALISE: x.\nHANDOFF: analista | apure tudo"),
         (SUP, "ANALISE: sem fechar.\nRESPOSTA FINAL: apuração incompleta.")],
        padrao="PASSO: ler\nFERRAMENTA: listar_arquivos\nENTRADA: ")
    e = sup.executar("uma tarefa")

    devolucao = [m for m in estado.historico if m.tipo == "devolucao"][0]
    checar("Não fechei a apuração" in devolucao.conteudo,
           "estourar os passos devolve resposta parcial honesta")
    checar("listar_arquivos" in devolucao.conteudo,
           "a resposta parcial diz quais ferramentas chegou a usar")

    # Ferramenta inexistente vira observação de erro, não crash.
    sup, estado = montar([
        (SUP, "ANALISE: x.\nHANDOFF: analista | apure"),
        (ANALISTA, "PASSO: tentar\nFERRAMENTA: consultar_banco\nENTRADA: pedidos"),
        (ANALISTA, "PASSO: a certa\nFERRAMENTA: listar_arquivos\nENTRADA: "),
        (ANALISTA, "RESPOSTA: há 3 arquivos. [fonte: data/arquivos]"),
        (SUP, "ANALISE: ok.\nRESPOSTA FINAL: há 3 arquivos [fonte: data/arquivos]"),
    ])
    e = sup.executar("quais arquivos existem")
    checar(e.termino == Termino.COMPLETO,
           "o especialista erra a ferramenta, se corrige e conclui")

    terceiro = [c for c in CHAMADAS if ANALISTA in c["prompt"]][1]["prompt"]
    checar("não existe" in terceiro,
           "o erro da ferramenta volta ao agente como observação")

    # Saída malformada não pode matar o turno na primeira tentativa.
    sup, estado = montar([
        (SUP, "ANALISE: x.\nHANDOFF: analista | apure"),
        (ANALISTA, "vou pensar um pouco sobre isso"),
        (ANALISTA, "RESPOSTA: 3 arquivos [fonte: data/arquivos]"),
        (SUP, "ANALISE: ok.\nRESPOSTA FINAL: 3 arquivos [fonte: data/arquivos]"),
    ])
    e = sup.executar("quais arquivos existem")
    checar(e.termino == Termino.COMPLETO,
           "uma saída malformada do especialista não encerra a execução")


# --------------------------------------------------------------------------- #
# 13. Avaliação de ponta a ponta
# --------------------------------------------------------------------------- #


def testar_avaliacao() -> None:
    secao("13. Avaliação de ponta a ponta")

    tarefa = Tarefa(
        id="m03",
        tarefa="Qual o valor dos itens do pedido 4477 com desconto?",
        verificacao="numerico",
        esperado="765",
        especialistas_esperados=["analista"],
        exige_fonte=True,
        dificuldade="media",
    )

    sup, _estado = montar([
        (SUP, "ANALISE: x.\nHANDOFF: analista | calcule com desconto"),
        (ANALISTA, "PASSO: calcular\nFERRAMENTA: calculadora\n"
                   "ENTRADA: (20 * 45.00) * 0.85"),
        (ANALISTA, "PASSO: registrar\nFERRAMENTA: registrar_fato\n"
                   "ENTRADA: o 4477 com desconto dá R$ 765,00 | fonte: pedidos.csv"),
        (ANALISTA, "RESPOSTA: R$ 765,00 [fonte: pedidos.csv]"),
        (SUP, "ANALISE: pronto.\n"
              "RESPOSTA FINAL: os itens somam R$ 765,00 já com o desconto de "
              "contrato. [fonte: pedidos.csv]"),
    ])
    r_ok = avaliar_execucao(sup.executar(tarefa.tarefa), tarefa)

    checar(r_ok.acertou, "resposta correta conta como acerto")
    checar(r_ok.roteamento_correto, "roteamento exato")
    checar(r_ok.citou_fonte, "a resposta final cita a fonte")
    checar(r_ok.fatos_com_fonte == 1, "o fato registrado tem procedência")

    # Resposta certa sem procedência: acerta, mas a auditoria reprova.
    sup, _estado = montar([
        (SUP, "ANALISE: x.\nHANDOFF: analista | calcule"),
        (ANALISTA, "RESPOSTA: R$ 765,00"),
        (SUP, "ANALISE: pronto.\nRESPOSTA FINAL: são R$ 765,00."),
    ])
    r_sem_fonte = avaliar_execucao(sup.executar(tarefa.tarefa), tarefa)
    checar(r_sem_fonte.acertou, "o número certo continua sendo acerto")
    checar(not r_sem_fonte.citou_fonte, "mas a falta de procedência é detectada")
    checar(not r_sem_fonte.procedencia_ok,
           "e a tarefa exigia fonte, então a procedência não passa")

    # Abstenção: inventar em grupo é o pior resultado possível.
    t_abs = Tarefa(id="m08", tarefa="Faturamento da Korlan SA em 2025?",
                   verificacao="abstencao", esperado="",
                   especialistas_esperados=[], exige_fonte=False)

    sup, _estado = montar([
        (SUP, "ANALISE: nenhuma ferramenta alcança isso.\n"
              "RESPOSTA FINAL: não temos essa informação em nenhuma fonte "
              "disponível ao time."),
    ])
    checar(avaliar_execucao(sup.executar(t_abs.tarefa), t_abs).acertou,
           "abstenção correta conta como acerto")

    sup, _estado = montar([
        (SUP, "ANALISE: vou estimar.\n"
              "RESPOSTA FINAL: o faturamento foi de R$ 4,2 milhões."),
    ])
    checar(not avaliar_execucao(sup.executar(t_abs.tarefa), t_abs).acertou,
           "inventar um número conta como erro")

    rel = Relatorio(modelo="modelo/teste", timestamp="t")
    rel.resultados = [r_ok, r_sem_fonte]
    checar(abs(rel.taxa_sucesso - 1.0) < 1e-9, "as duas acertaram o número")
    checar(abs(rel.taxa_procedencia - 0.5) < 1e-9,
           "mas só metade citou fonte: as duas medidas são independentes")


# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 68)
    print("  Testes com LLM simulado — módulo 06, sistema multiagente")
    print("  Nenhuma chamada real de API, nenhum custo.")
    print("=" * 68)

    testar_handoff_simples()
    testar_dois_especialistas()
    testar_isolamento_do_redator()
    testar_funcoes_de_resposta()
    testar_salvaguarda_corrige()
    testar_salvaguarda_encerra()
    testar_sem_salvaguardas()
    testar_formato_invalido()
    testar_especialista_invalido()
    testar_limite_rodadas_e_repeticao()
    testar_falha_de_rede()
    testar_laco_de_ferramentas()
    testar_avaliacao()

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
