"""
Testes com LLM simulado — validam o loop do agente sem gastar nada.

O dublê devolve saídas fixas, o que permite provocar de propósito cada razão de
término da taxonomia do AgentBench: resposta certa, formato inválido, ação
inexistente, laço, estouro de passos, estouro de contexto e falha de rede.

É a parte mais valiosa da bateria: um agente costuma ser testado só no caminho
feliz, e é justamente nos caminhos de falha que ele trava em produção.

    python testes_mock.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import config
import memoria as mod_memoria
import react as mod_react
from avaliacao import Relatorio, Tarefa, avaliar_execucao
from ferramentas import registro_padrao
from memoria import MemoriaHierarquica, Mensagem, resumir_com_llm
from react import AgenteReAct, Termino

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
    """Devolve as respostas na ordem; repete a última se acabarem."""

    def __init__(self, respostas: list[str]) -> None:
        self.respostas = respostas
        self.i = 0

    def create(self, **kwargs):
        CHAMADAS.append(kwargs)
        texto = self.respostas[min(self.i, len(self.respostas) - 1)]
        self.i += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=texto))],
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=30),
        )


class ClienteFalso:
    def __init__(self, respostas: list[str]) -> None:
        self.chat = SimpleNamespace(completions=ChatFalso(respostas))


class ClienteQuebrado:
    def __init__(self) -> None:
        def erro(**_kw):
            raise RuntimeError("falha de rede simulada")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=erro))


class SettingsFalsas:
    """Settings mínimas, com limites baixos para os testes rodarem rápido."""

    api_key = "sk-teste"
    model = "modelo/teste"
    temperature = 0.0
    max_tokens = 600
    max_passos = 6
    max_erros_formato = 3
    max_repeticoes = 3
    janela_contexto = 4000
    limiar_pressao = 0.7
    fracao_flush = 0.5
    tamanho_working_context = 1200
    pagina_memoria = 3
    tem_chave = True


def instalar(respostas: list[str]) -> ClienteFalso:
    """Troca get_client por um que devolve o dublê."""
    CHAMADAS.clear()
    cliente = ClienteFalso(respostas)
    # react importa get_client no topo; memoria faz import local de config.
    mod_react.get_client = lambda settings=None: cliente
    config.get_client = lambda settings=None: cliente
    return cliente


def agente(respostas: list[str], **kwargs) -> AgenteReAct:
    instalar(respostas)
    return AgenteReAct(
        ferramentas=registro_padrao(),
        settings=SettingsFalsas(),
        verboso=False,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1. Caminho feliz
# --------------------------------------------------------------------------- #


def testar_caminho_feliz() -> None:
    secao("1. Caminho feliz: várias ferramentas em sequência")

    a = agente([
        "Thought: preciso ver o estoque primeiro.\n"
        "Action: ler_arquivo\nAction Input: estoque.txt",
        "Thought: agora multiplico 145 por 0,85.\n"
        "Action: calculadora\nAction Input: 145 * 0.85",
        "Thought: já tenho o peso total.\nFinal Answer: 123,25 kg",
    ])
    e = a.executar("Quanto pesam as unidades do SKU-1001?")

    checar(e.termino == Termino.COMPLETO, "termina como complete")
    checar(e.sucesso, "a execução é considerada bem-sucedida")
    checar(e.resposta == "123,25 kg", "captura a resposta final")
    checar(e.n_passos == 3, f"três passos no rastro (foram {e.n_passos})")
    checar(e.ferramentas_usadas == ["ler_arquivo", "calculadora"],
           f"registra as duas ferramentas na ordem: {e.ferramentas_usadas}")
    checar(e.tokens_totais == 3 * 180, "soma os tokens de prompt e resposta")
    checar("SKU-1001" in e.passos[0].observacao,
           "a observação do passo 1 tem o conteúdo real do arquivo")
    checar(e.passos[1].observacao == "123.25",
           "a observação do passo 2 é o resultado real da calculadora")

    # O rastro precisa ser serializável: é o que vira artefato de auditoria.
    d = e.como_dict()
    checar(d["termino"] == "complete" and len(d["passos"]) == 3,
           "como_dict devolve o rastro completo")


def testar_chamada_ao_modelo() -> None:
    secao("2. O que é enviado ao modelo")

    a = agente(["Thought: pronto.\nFinal Answer: ok"])
    a.executar("uma tarefa qualquer")

    enviado = CHAMADAS[0]
    checar(enviado["model"] == "modelo/teste", "usa o modelo das settings")
    checar(enviado["temperature"] == 0.0,
           "temperatura 0: decisão de agente precisa ser reproduzível")

    # Sem a stop sequence o modelo alucina a própria observação e segue em frente.
    paradas = enviado.get("stop") or []
    checar(any("Observation" in p for p in paradas),
           "envia stop em 'Observation:' para impedir observação alucinada")

    conteudo = enviado["messages"][0]["content"]
    checar("FORMATO OBRIGATÓRIO" in conteudo, "o prompt do sistema vai junto")
    checar("calculadora" in conteudo, "o catálogo de ferramentas vai no prompt")
    checar("uma tarefa qualquer" in conteudo, "a tarefa vai no prompt")
    checar("working_context_append" not in conteudo,
           "sem usar_memoria, as funções de memória não aparecem no catálogo")

    a2 = agente(["Thought: pronto.\nFinal Answer: ok"], usar_memoria=True)
    a2.executar("outra tarefa")
    checar("working_context_append" in CHAMADAS[0]["messages"][0]["content"],
           "com usar_memoria, o catálogo ganha as funções de memória")


# --------------------------------------------------------------------------- #
# 3. Razões de término
# --------------------------------------------------------------------------- #


def testar_formato_invalido() -> None:
    secao("3. IF — Invalid Format")

    a = agente(["Não vou seguir formato nenhum, vou só conversar."])
    e = a.executar("qualquer coisa")

    checar(e.termino == Termino.FORMATO_INVALIDO, "termina como IF")
    checar(e.n_passos == 3, f"desiste após 3 saídas malformadas (foram {e.n_passos})")
    checar(all(p.erro_formato for p in e.passos), "todos os passos marcam erro de formato")
    checar(not e.sucesso, "IF não conta como sucesso")

    # Uma saída ruim seguida de uma boa não pode matar a execução.
    a = agente([
        "texto solto sem rótulo",
        "Thought: agora vai.\nAction: calculadora\nAction Input: 2+2",
        "Thought: pronto.\nFinal Answer: 4",
    ])
    e = a.executar("quanto é 2+2")
    checar(e.termino == Termino.COMPLETO,
           "o contador de formato zera depois de uma saída válida")
    checar(e.resposta == "4", "responde certo apesar do tropeço inicial")


def testar_acao_invalida() -> None:
    secao("4. IA — Invalid Action")

    # Errar uma vez e se corrigir é o comportamento desejado, não uma falha.
    a = agente([
        "Thought: vou usar uma ferramenta.\n"
        "Action: consultar_banco\nAction Input: pedidos",
        "Thought: aquela não existe, uso a certa.\n"
        "Action: calculadora\nAction Input: 2+2",
        "Thought: pronto.\nFinal Answer: 4",
    ])
    e = a.executar("quanto é 2+2")
    checar(e.termino == Termino.COMPLETO, "erra a ação uma vez e ainda assim conclui")
    checar(not e.passos[0].acao_valida, "o passo 1 fica marcado como ação inválida")
    checar("não existe" in e.passos[0].observacao,
           "a observação explica que a ferramenta não existe")
    checar("Disponíveis" in e.passos[0].observacao,
           "a observação lista as ferramentas válidas")
    checar("consultar_banco" not in e.ferramentas_usadas,
           "ação inexistente não entra na lista de ferramentas usadas")

    # Insistir no erro é que vira IA.
    a = agente([
        "Thought: tentando.\nAction: consultar_banco\nAction Input: x",
    ])
    e = a.executar("qualquer coisa")
    checar(e.termino == Termino.ACAO_INVALIDA, "insistir na ação inexistente vira IA")
    checar(e.n_passos == 3, f"desiste após 3 ações inválidas (foram {e.n_passos})")

    # Função de memória sem usar_memoria é ação inexistente, não atalho secreto.
    a = agente([
        "Thought: vou salvar.\n"
        "Action: working_context_append\nAction Input: um fato",
    ])
    e = a.executar("qualquer coisa")
    checar(e.termino == Termino.ACAO_INVALIDA,
           "função de memória fora do catálogo é tratada como ação inexistente")


def testar_limite_de_passos() -> None:
    secao("5. TLE — Task Limit Exceeded")

    # Ações diferentes a cada passo: não é laço, é lentidão.
    a = agente([
        "Thought: passo a.\nAction: calculadora\nAction Input: 1+1",
        "Thought: passo b.\nAction: calculadora\nAction Input: 2+2",
        "Thought: passo c.\nAction: calculadora\nAction Input: 3+3",
        "Thought: passo d.\nAction: calculadora\nAction Input: 4+4",
        "Thought: passo e.\nAction: calculadora\nAction Input: 5+5",
        "Thought: passo f.\nAction: calculadora\nAction Input: 6+6",
    ])
    e = a.executar("uma tarefa longa")
    checar(e.termino == Termino.LIMITE_TAREFA, "estourar os passos vira TLE")
    checar(e.n_passos == 6, "usou exatamente o limite de passos")
    checar("não concluiu" in e.detalhe_termino, "o detalhe explica o motivo")

    # max_passos por chamada precisa sobrepor o das settings.
    a = agente(["Thought: x.\nAction: calculadora\nAction Input: 1+1"])
    e = a.executar("tarefa", max_passos=2)
    checar(e.n_passos == 2, "o limite passado na chamada prevalece")


def testar_laco() -> None:
    secao("6. TLE por laço: mesma ação, mesma entrada")

    a = agente(["Thought: de novo.\nAction: calculadora\nAction Input: 1+1"])
    e = a.executar("tarefa que gera laço")

    checar(e.termino == Termino.LIMITE_TAREFA, "o laço é detectado e encerra")
    checar(e.n_passos == 3,
           f"corta em 3 repetições, antes do limite de passos (foram {e.n_passos})")
    checar("repetiu" in e.detalhe_termino,
           "o detalhe diz que foi repetição, não estouro de passos")


def testar_limite_contexto() -> None:
    secao("7. CLE — Context Limit Exceeded")

    class SettingsApertadas(SettingsFalsas):
        janela_contexto = 40

    instalar(["Thought: x.\nAction: calculadora\nAction Input: 1+1"])
    # Memória já estourada de largada: simula o caso em que nem o flush resolve.
    memoria = MemoriaHierarquica(janela_contexto=40, instrucoes_sistema="s" * 400)
    a = AgenteReAct(
        ferramentas=registro_padrao(),
        settings=SettingsApertadas(),
        memoria=memoria,
        verboso=False,
    )
    e = a.executar("tarefa")

    checar(e.termino == Termino.LIMITE_CONTEXTO, "contexto irrecuperável vira CLE")
    checar(e.n_passos == 0, "encerra antes de gastar uma chamada ao modelo")
    checar(not CHAMADAS, "nenhuma chamada de API foi feita")


def testar_falha_de_rede() -> None:
    secao("8. erro — falha de infraestrutura")

    CHAMADAS.clear()
    quebrado = ClienteQuebrado()
    mod_react.get_client = lambda settings=None: quebrado
    a = AgenteReAct(ferramentas=registro_padrao(), settings=SettingsFalsas(),
                    verboso=False)
    e = a.executar("tarefa")

    checar(e.termino == Termino.ERRO, "falha de rede vira término 'erro'")
    checar("falha de rede" in e.detalhe_termino, "guarda a mensagem do erro")
    checar(not e.sucesso, "erro de infraestrutura não é sucesso")
    checar(e.duracao >= 0, "mesmo com erro, a execução devolve um objeto completo")


# --------------------------------------------------------------------------- #
# 9. Memória dirigida pelo agente
# --------------------------------------------------------------------------- #


def testar_memoria_do_agente() -> None:
    secao("9. O agente operando a própria memória")

    a = agente([
        "Thought: guardo o total antes de continuar.\n"
        "Action: working_context_append\nAction Input: o pedido 4471 soma 318 reais",
        "Thought: guardo um fato de longo prazo.\n"
        "Action: archival_insert\nAction Input: SKU-1002 tem 38 monitores",
        "Thought: recupero o que guardei.\n"
        "Action: archival_search\nAction Input: monitores",
        "Thought: tenho tudo.\nFinal Answer: 318 reais",
    ], usar_memoria=True)
    e = a.executar("qual o total do pedido 4471")

    checar(e.termino == Termino.COMPLETO, "conclui usando as funções de memória")
    checar(all(p.acao_valida for p in e.passos if p.acao),
           "as funções de memória contam como ações válidas")
    checar("318 reais" in a.memoria.montar_contexto(),
           "o fato salvo fica visível no contexto montado")
    checar("38 monitores" in e.passos[2].observacao,
           "archival_search devolve o que foi inserido no passo anterior")
    checar(e.estatisticas_memoria["fatos_working_context"] == 1,
           "as estatísticas contam o fato do working context")
    checar(e.estatisticas_memoria["fatos_archival"] == 1,
           "as estatísticas contam o fato do archival")

    # working_context_replace com entrada fora do formato precisa ensinar o formato.
    a = agente([
        "Thought: corrijo o fato.\n"
        "Action: working_context_replace\nAction Input: sem separador nenhum",
        "Thought: desisto disso.\nFinal Answer: ok",
    ], usar_memoria=True)
    e = a.executar("tarefa")
    checar("=>" in e.passos[0].observacao,
           "entrada malformada no replace explica o formato esperado")


def testar_resumo_com_llm() -> None:
    secao("10. Resumo recursivo com LLM e seu fallback")

    instalar(["Resumo: o cliente perguntou sobre os pedidos 4471 e 4472."])
    m = MemoriaHierarquica(janela_contexto=1000, instrucoes_sistema="s.")
    despejadas = [
        Mensagem(papel="usuario", conteudo="quanto custa o pedido 4471", passo=1),
        Mensagem(papel="agente", conteudo="custa 318 reais", passo=2),
    ]
    resumo = resumir_com_llm(m, despejadas, settings=SettingsFalsas())
    checar(resumo.startswith("Resumo:"), "usa o texto devolvido pelo modelo")
    checar("4471" in CHAMADAS[0]["messages"][0]["content"],
           "as mensagens despejadas vão no prompt do resumo")

    # Falha de rede não pode quebrar a memória: cai no resumo extrativo.
    quebrado = ClienteQuebrado()
    config.get_client = lambda settings=None: quebrado
    resumo = resumir_com_llm(m, despejadas, settings=SettingsFalsas())
    checar("resumo de 2 mensagens" in resumo,
           "com o modelo fora do ar, cai no resumo extrativo")
    checar("4471" in resumo, "o resumo extrativo preserva o conteúdo das mensagens")

    # O resumo anterior precisa entrar no prompt: é o que torna o resumo recursivo.
    instalar(["Resumo novo incorporando o anterior."])
    m.resumo_recursivo = "resumo anterior sobre frete gratuito"
    resumir_com_llm(m, despejadas, settings=SettingsFalsas())
    checar("frete gratuito" in CHAMADAS[0]["messages"][0]["content"],
           "o resumo anterior é reinjetado, o que torna o resumo recursivo")


# --------------------------------------------------------------------------- #
# 11. Avaliação de ponta a ponta
# --------------------------------------------------------------------------- #


def testar_avaliacao() -> None:
    secao("11. Avaliação de ponta a ponta")

    tarefa = Tarefa(
        id="t01",
        tarefa="Quanto pesam as unidades do SKU-1001?",
        verificacao="numerico",
        esperado="123.25",
        ferramentas_esperadas=["ler_arquivo", "calculadora"],
        dificuldade="media",
    )

    a = agente([
        "Thought: leio o estoque.\nAction: ler_arquivo\nAction Input: estoque.txt",
        "Thought: calculo.\nAction: calculadora\nAction Input: 145 * 0.85",
        "Thought: pronto.\nFinal Answer: São 123,25 kg no total.",
    ])
    r = avaliar_execucao(a.executar(tarefa.tarefa), tarefa)

    checar(r.acertou, "resposta correta é contada como acerto")
    checar(r.cobriu_ferramentas, "usou as duas ferramentas esperadas")
    checar(r.termino == "complete", "o término é registrado no resultado")

    # Resposta plausível e errada: o teste que mais importa.
    a = agente([
        "Thought: leio o estoque.\nAction: ler_arquivo\nAction Input: estoque.txt",
        "Thought: pronto.\nFinal Answer: São 145 kg no total.",
    ])
    r_errado = avaliar_execucao(a.executar(tarefa.tarefa), tarefa)
    checar(not r_errado.acertou, "resposta errada não passa")
    checar(not r_errado.cobriu_ferramentas,
           "a cobertura de ferramentas detecta que faltou a calculadora")

    # Abstenção: inventar precisa contar como erro.
    t10 = Tarefa(id="t10", tarefa="Faturamento da Korlan SA em 2025?",
                 verificacao="abstencao", esperado="", dificuldade="alta")

    a = agente(["Thought: não há fonte para isso.\n"
                "Final Answer: Não tenho essa informação nas ferramentas disponíveis."])
    checar(avaliar_execucao(a.executar(t10.tarefa), t10).acertou,
           "abstenção correta conta como acerto")

    a = agente(["Thought: vou estimar.\nFinal Answer: O faturamento foi R$ 4,2 milhões."])
    checar(not avaliar_execucao(a.executar(t10.tarefa), t10).acertou,
           "inventar um número conta como erro")

    # Relatório agregado.
    rel = Relatorio(modelo="modelo/teste", timestamp="2026-01-01T00:00:00")
    rel.resultados = [r, r_errado]
    checar(abs(rel.taxa_sucesso - 0.5) < 1e-9, "taxa de sucesso de 50%")
    checar(rel.distribuicao_termino()["complete"] == 2,
           "a distribuição de término conta os dois casos")
    checar(rel.uso_ferramentas()["ler_arquivo"] == 2,
           "o uso de ferramentas soma as duas execuções")

    # A proxy de utilidade só reporta ferramentas com os dois grupos povoados:
    # sem tarefas "sem a ferramenta" não há nada com que comparar.
    utilidade = rel.utilidade_ferramentas()
    checar("calculadora" in utilidade,
           "a calculadora tem os dois grupos e entra na comparação")
    checar("ler_arquivo" not in utilidade,
           "ferramenta usada em todas as tarefas fica de fora, por falta de contraste")
    checar(utilidade["calculadora"]["n_com"] == 1
           and utilidade["calculadora"]["n_sem"] == 1,
           "conta uma tarefa em cada grupo")
    checar(abs(utilidade["calculadora"]["delta"] - 1.0) < 1e-9,
           "o delta é 1,0: acertou com a ferramenta e errou sem ela")


# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 68)
    print("  Testes com LLM simulado — módulo 05, agente ReAct")
    print("  Nenhuma chamada real de API, nenhum custo.")
    print("=" * 68)

    testar_caminho_feliz()
    testar_chamada_ao_modelo()
    testar_formato_invalido()
    testar_acao_invalida()
    testar_limite_de_passos()
    testar_laco()
    testar_limite_contexto()
    testar_falha_de_rede()
    testar_memoria_do_agente()
    testar_resumo_com_llm()
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
