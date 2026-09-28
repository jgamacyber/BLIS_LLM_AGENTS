"""
Testes offline — validam a lógica sem chamar a API nem gastar créditos.

Cobrem as sete ferramentas (inclusive a segurança da calculadora e do sandbox
de arquivos), o índice BM25, a hierarquia de memória do MemGPT, o parser da
saída do modelo e a verificação de respostas da avaliação.

O parser é a peça mais frágil de um agente ReAct: se ele aceitar uma saída
malformada, o agente age sobre lixo; se recusar uma saída correta, a execução
morre com IF sem motivo. Por isso ele tem a maior bateria aqui.

    python testes_offline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from avaliacao import (
    Tarefa,
    _extrair_numero,
    carregar_tarefas,
    eh_abstencao,
    verificar,
)
from ferramentas import (
    IndiceCorpus,
    Ferramenta,
    RegistroFerramentas,
    Resultado,
    converter_unidades,
    registro_padrao,
)
from memoria import MemoriaHierarquica, estimar_tokens
from react import FUNCOES_MEMORIA, Passo, Termino, analisar_saida, catalogo_com_memoria

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


def perto(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def obs(registro: RegistroFerramentas, nome: str, entrada: str) -> str:
    return registro.executar(nome, entrada).como_observacao()


# --------------------------------------------------------------------------- #
# 1. Calculadora
# --------------------------------------------------------------------------- #


def testar_calculadora() -> None:
    secao("1. Calculadora: AST, nunca eval()")
    r = registro_padrao()

    checar(obs(r, "calculadora", "2 + 3 * 4") == "14", "precedência de operadores")
    checar(obs(r, "calculadora", "(3 * 98.50) + 22.50") == "318",
           "expressão do pedido 4471 dá 318")
    checar(obs(r, "calculadora", "2^10") == "1024", "aceita ^ como potência")
    checar(obs(r, "calculadora", "sqrt(144)") == "12", "função sqrt")
    checar(obs(r, "calculadora", "round(2.71828, 2)") == "2.72",
           "round com dois argumentos não quebra por causa da vírgula")
    checar(obs(r, "calculadora", "max(3, 5)") == "5", "max com dois argumentos")

    # A ambiguidade da vírgula, que é o caso real de um agente em português.
    checar(obs(r, "calculadora", "(3 * 98,50) + 22,50") == "318",
           "vírgula decimal brasileira é reinterpretada quando a 1ª leitura falha")
    checar(obs(r, "calculadora", "145 * 0,85") == "123.25",
           "145 x 0,85 dá 123.25")

    # Segurança: o LLM escolhe a string, então a superfície de ataque é real.
    for maligna in ('__import__("os").system("ls")', "open('/etc/passwd')",
                    "().__class__", "lambda: 1", "[1,2][0]"):
        resultado = r.executar("calculadora", maligna)
        checar(not resultado.sucesso, f"recusa construção perigosa: {maligna[:28]}")

    checar(not r.executar("calculadora", "9**9**9").sucesso,
           "recusa expoente gigante em vez de travar o processo")
    checar(not r.executar("calculadora", "10 / 0").sucesso, "divisão por zero é erro")
    checar(not r.executar("calculadora", "").sucesso, "expressão vazia é erro")
    checar(not r.executar("calculadora", "abc").sucesso, "nome desconhecido é erro")


# --------------------------------------------------------------------------- #
# 2. Sandbox de arquivos
# --------------------------------------------------------------------------- #


def testar_arquivos() -> None:
    secao("2. Sandbox de arquivos")
    r = registro_padrao()

    listagem = obs(r, "listar_arquivos", "")
    for esperado in ("pedidos.csv", "estoque.txt", "notas.md"):
        checar(esperado in listagem, f"listar_arquivos mostra {esperado}")

    conteudo = obs(r, "ler_arquivo", "estoque.txt")
    checar("SKU-1001" in conteudo, "ler_arquivo devolve o conteúdo do estoque")

    # Traversal: o agente pode escrever qualquer caminho, então o teste importa.
    for fuga in ("../../config.py", "/etc/passwd", "../react.py",
                 "data/../../main.py", "..%2F..%2Fconfig.py"):
        resultado = r.executar("ler_arquivo", fuga)
        checar(not resultado.sucesso, f"recusa caminho fora do sandbox: {fuga[:24]}")

    checar(not r.executar("ler_arquivo", "nao_existe.txt").sucesso,
           "arquivo inexistente vira erro, não exceção")


# --------------------------------------------------------------------------- #
# 3. Conversão de unidades
# --------------------------------------------------------------------------- #


def testar_conversao() -> None:
    secao("3. Conversão de unidades")
    r = registro_padrao()

    saida = obs(r, "converter_unidades", "197.6 kg para lb")
    checar("435.6" in saida, f"197,6 kg dá ~435,6 lb (saída: {saida})")

    checar("1000" in obs(r, "converter_unidades", "1 km para m"), "1 km = 1000 m")
    checar("150" in obs(r, "converter_unidades", "2.5 h para min"), "2,5 h = 150 min")

    # Ida e volta não pode acumular erro perceptível.
    ida = converter_unidades("10 kg para lb").metadados.get("valor")
    volta = converter_unidades(f"{ida} lb para kg").metadados.get("valor")
    checar(volta is not None and perto(volta, 10.0, 1e-6),
           "kg -> lb -> kg volta ao valor original")

    checar(not r.executar("converter_unidades", "5 kg para litros").sucesso,
           "recusa conversão entre dimensões diferentes")
    checar(not r.executar("converter_unidades", "abacaxi").sucesso,
           "recusa entrada sem formato de conversão")


# --------------------------------------------------------------------------- #
# 4. BM25 e corpus
# --------------------------------------------------------------------------- #


def testar_bm25() -> None:
    secao("4. Índice BM25")

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        (pasta / "a.md").write_text(
            "# Agentes\nO loop ReAct alterna pensamento e ação.\n", encoding="utf-8")
        (pasta / "b.md").write_text(
            "# Memória\nO MemGPT pagina memória entre contexto e disco.\n",
            encoding="utf-8")
        (pasta / "c.md").write_text(
            "# Outro\nTexto sem relação com o assunto buscado.\n", encoding="utf-8")

        indice = IndiceCorpus(pasta)
        checar(len(indice) == 3, "indexou os três documentos")

        top = indice.buscar("paginação de memória no MemGPT", top_k=3)
        checar(bool(top) and top[0]["id"] == "b",
               "a busca por memória traz o documento do MemGPT primeiro")

        scores = [d["score"] for d in top]
        checar(scores == sorted(scores, reverse=True),
               "resultados vêm ordenados por score decrescente")

        checar(indice.buscar("zzzz palavra inexistente qqqq") == [],
               "consulta sem casamento devolve lista vazia")

    r = registro_padrao()
    corpus = obs(r, "listar_corpus", "")
    for doc in ("react", "memgpt", "coala", "toolformer"):
        checar(doc in corpus, f"corpus contém o resumo de {doc}")

    busca = obs(r, "buscar_corpus", "por que o pensamento nao gera observacao")
    checar("ReAct" in busca, "busca sobre pensamento cai no documento do ReAct")


# --------------------------------------------------------------------------- #
# 5. Registro de ferramentas
# --------------------------------------------------------------------------- #


def testar_registro() -> None:
    secao("5. Registro de ferramentas")
    r = registro_padrao()

    checar(len(r) == 7, "sete ferramentas registradas")
    checar("calculadora" in r, "__contains__ encontra ferramenta existente")
    checar("inexistente" not in r, "__contains__ recusa ferramenta inexistente")

    erro = r.executar("ferramenta_inventada", "x")
    checar(not erro.sucesso, "ferramenta inexistente devolve erro")
    checar("Disponíveis" in erro.erro,
           "o erro lista as ferramentas válidas, para o agente se corrigir")

    catalogo = r.catalogo()
    for nome in r.nomes():
        checar(nome in catalogo, f"catálogo descreve {nome}")

    # Exceção dentro da ferramenta nunca pode escapar para o loop do agente.
    def explode(_entrada: str) -> Resultado:
        raise RuntimeError("boom")

    registro = RegistroFerramentas([
        Ferramenta(nome="explosiva", descricao="falha sempre", funcao=explode)
    ])
    resultado = registro.executar("explosiva", "x")
    checar(not resultado.sucesso and "boom" in resultado.erro,
           "exceção na ferramenta vira Resultado com erro, não crash")


# --------------------------------------------------------------------------- #
# 6. Memória hierárquica
# --------------------------------------------------------------------------- #


def testar_memoria() -> None:
    secao("6. Memória hierárquica (MemGPT)")

    m = MemoriaHierarquica(janela_contexto=400, instrucoes_sistema="sistema.")
    checar(m.ocupacao > 0, "as instruções já ocupam contexto")

    # Empurra até passar de 70% sem chegar a 100%: o aviso deve sair uma vez,
    # não uma vez por mensagem. Repetir o alerta gasta o contexto que ele
    # existe para proteger.
    i = 0
    while m.ocupacao < 0.85 and i < 40:
        i += 1
        m.adicionar_mensagem("usuario", "texto de teste com algum tamanho " * 2, i)

    checar(m.sob_pressao, "a memória entrou em regime de pressão")
    avisos = [e for e in m.eventos if e.tipo == "pressao"]
    checar(len(avisos) == 1,
           f"o aviso de pressão saiu uma única vez na subida ({len(avisos)} avisos)")
    checar(len(m.alertas_pendentes) == 1, "há exatamente um alerta pendente")

    # montar_contexto consome os alertas: o mesmo aviso não volta no passo seguinte.
    m.montar_contexto()
    checar(not m.alertas_pendentes, "montar_contexto drena os alertas pendentes")

    # O recall guarda tudo, inclusive o que já saiu da fila.
    checar(len(m.recall) == i, "recall storage mantém todas as mensagens")
    checar(len(m.fila) <= i, "a fila do main context é menor ou igual ao recall")

    # Flush de verdade: forçar estouro e conferir que o contexto encolheu.
    m2 = MemoriaHierarquica(janela_contexto=120, instrucoes_sistema="s.")
    for i in range(12):
        m2.adicionar_mensagem("usuario", f"mensagem número {i} com texto de enchimento", i)
    checar(any(e.tipo == "flush" for e in m2.eventos), "o flush disparou")
    checar(bool(m2.resumo_recursivo), "o flush produziu resumo recursivo")
    checar(len(m2.fila) < len(m2.recall), "a fila encolheu e o recall não")
    checar(m2.tokens_usados <= m2.janela_contexto * 1.5,
           "depois do flush o contexto voltou para dentro de um limite razoável")

    # O resumo não pode crescer sem limite, senão reintroduz o problema.
    limite = int(m2.janela_contexto * 0.25) * 4
    checar(len(m2.resumo_recursivo) <= limite + 60,
           "o resumo recursivo respeita o teto de tamanho")
    checar("\n" not in m2.resumo_recursivo[:1] and
           not m2.resumo_recursivo.startswith(" "),
           "o resumo truncado não começa no meio de uma linha")

    # Funções de memória do agente.
    m3 = MemoriaHierarquica(janela_contexto=2000, instrucoes_sistema="s.")
    m3.working_context_append("o pedido 4471 custa 318 reais")
    checar("4471" in m3.montar_contexto(), "fato do working context aparece no contexto")

    m3.working_context_replace("318 reais", "347,90 reais")
    contexto = m3.montar_contexto()
    checar("347,90" in contexto and "318 reais" not in contexto,
           "working_context_replace corrige o fato no lugar")

    m3.archival_insert("SKU-1002 tem 38 monitores", rotulo="estoque")
    achado = m3.archival_search("monitores")
    checar("38 monitores" in achado, "archival_search encontra o fato guardado")
    checar("nada" in m3.archival_search("assunto inexistente aqui").lower(),
           "busca sem resultado diz que não achou")

    m3.adicionar_mensagem("usuario", "qual era o valor do frete do pedido 4471", 1)
    checar("frete" in m3.recall_search("frete"), "recall_search acha no histórico")

    # Paginação: o artigo insiste nisso para não estourar o contexto de volta.
    m4 = MemoriaHierarquica(janela_contexto=4000, pagina=2, instrucoes_sistema="s.")
    for i in range(7):
        m4.archival_insert(f"fato número {i} sobre pedidos")
    pagina1 = m4.archival_search("pedidos", pagina=0)
    checar("página 1/" in pagina1, "a busca informa a página atual")
    checar(pagina1.count("\n  ") <= 3, "a página respeita o tamanho configurado")

    checar(estimar_tokens("abcd" * 10) == 10, "estimador de tokens: 4 chars por token")


# --------------------------------------------------------------------------- #
# 7. Parser da saída do modelo
# --------------------------------------------------------------------------- #


def testar_parser() -> None:
    secao("7. Parser da saída do modelo")

    p = analisar_saida(
        "Thought: preciso do arquivo de estoque\n"
        "Action: ler_arquivo\n"
        "Action Input: estoque.txt"
    )
    checar(p.acao == "ler_arquivo", "forma de três linhas: nome da ação")
    checar(p.entrada_acao == "estoque.txt", "forma de três linhas: entrada")
    checar(not p.erro_formato, "forma de três linhas é válida")

    p = analisar_saida("Thought: vou calcular\nAction: calculadora[3 * 98.50]")
    checar(p.acao == "calculadora" and p.entrada_acao == "3 * 98.50",
           "forma compacta nome[entrada] também é aceita")

    p = analisar_saida("Thought: já tenho tudo\nFinal Answer: 318 reais")
    checar(p.concluiu and p.resposta_final == "318 reais", "resposta final")
    checar(not p.acao, "resposta final não gera ação")

    # Observação alucinada: o modelo inventa o resultado antes de agir.
    p = analisar_saida(
        "Thought: vou ler\nAction: ler_arquivo\nAction Input: estoque.txt\n"
        "Observation: SKU-1001 tem 999 unidades\n"
        "Thought: pronto\nFinal Answer: 999"
    )
    checar(p.acao == "ler_arquivo", "corta a alucinação e mantém a ação")
    checar(not p.concluiu,
           "a resposta final inventada depois da observação falsa é descartada")

    # Variações de forma que um modelo real produz.
    checar(analisar_saida("thought: x\naction: calculadora\naction input: 1+1").acao
           == "calculadora", "rótulos em minúsculo")
    checar(analisar_saida("Thought: x\nAction : calculadora\nAction Input : 1+1").acao
           == "calculadora", "espaço antes dos dois pontos")

    p = analisar_saida("Thought: não sei o que fazer agora.")
    checar(bool(p.erro_formato), "pensamento sem ação e sem resposta é erro de formato")

    p = analisar_saida("")
    checar(bool(p.erro_formato), "saída vazia é erro de formato")

    p = analisar_saida("Só um texto solto sem nenhum rótulo.")
    checar(bool(p.erro_formato), "texto sem rótulo é erro de formato")

    p = analisar_saida(
        "Thought: multilinha\nprimeira parte\nsegunda parte\n"
        "Action: calculadora\nAction Input: 2+2"
    )
    checar("segunda parte" in p.pensamento, "pensamento de várias linhas é preservado")


# --------------------------------------------------------------------------- #
# 8. Taxonomia de término e catálogo
# --------------------------------------------------------------------------- #


def testar_taxonomia() -> None:
    secao("8. Taxonomia de término (AgentBench)")

    for t in Termino:
        checar(bool(t.descricao()), f"{t.value} tem descrição legível")

    checar(Termino.FORMATO_INVALIDO.value == "IF", "IF é Invalid Format")
    checar(Termino.ACAO_INVALIDA.value == "IA", "IA é Invalid Action")
    checar(Termino.LIMITE_TAREFA.value == "TLE", "TLE é Task Limit Exceeded")
    checar(Termino.LIMITE_CONTEXTO.value == "CLE", "CLE é Context Limit Exceeded")

    catalogo = catalogo_com_memoria(registro_padrao())
    for funcao in FUNCOES_MEMORIA:
        checar(funcao in catalogo,
               f"o catálogo com memória descreve {funcao}")


# --------------------------------------------------------------------------- #
# 9. Verificação de respostas
# --------------------------------------------------------------------------- #


def testar_verificacao() -> None:
    secao("9. Verificação de respostas")

    checar(_extrair_numero("o total é R$ 318,00") == 318.0,
           "extrai número com vírgula decimal e cifrão")
    checar(_extrair_numero("são 1.234,50 reais") == 1234.50,
           "extrai número com separador de milhar")
    checar(_extrair_numero("435.63 lb") == 435.63, "extrai número com ponto decimal")
    checar(_extrair_numero("nenhum número aqui") is None, "texto sem número dá None")

    # O ponto final da frase não pode virar separador decimal: um agente
    # termina quase toda resposta com ponto, e isso reprovaria acertos.
    checar(_extrair_numero("O total é R$ 318,00.") == 318.0,
           "ponto final da frase não é confundido com separador")
    checar(_extrair_numero("O peso é 435,63 lb.") == 435.63,
           "ponto final depois da unidade também é ignorado")
    checar(_extrair_numero("-12,5.") == -12.5, "número negativo com ponto final")

    t_num = Tarefa(id="x", tarefa="", verificacao="numerico", esperado="318")
    checar(verificar("O total é R$ 318,00.", t_num), "aceita o número correto")
    checar(verificar("O total é 318,05.", t_num), "aceita dentro da tolerância de 2%")
    checar(not verificar("O total é 295,50.", t_num), "recusa número errado")
    checar(not verificar("Não consegui calcular.", t_num), "recusa resposta sem número")

    t_cont = Tarefa(id="y", tarefa="", verificacao="contem", esperado=["sim"])
    checar(verificar("Sim, tem direito a frete gratuito.", t_cont),
           "verificação por conteúdo, sem diferenciar maiúsculas")
    checar(not verificar("Não tem direito.", t_cont), "recusa quando o termo falta")

    # Abstenção: a t10 do conjunto não tem resposta em nenhuma ferramenta.
    for frase in ("Não sei responder com as ferramentas disponíveis.",
                  "Não tenho essa informação.",
                  "Não é possível determinar com os dados disponíveis."):
        checar(eh_abstencao(frase), f"reconhece abstenção: {frase[:34]}")

    checar(not eh_abstencao("O faturamento foi de R$ 4,2 milhões."),
           "resposta inventada não conta como abstenção")

    t_abs = Tarefa(id="z", tarefa="", verificacao="abstencao", esperado="")
    checar(verificar("Não tenho essa informação nas ferramentas.", t_abs),
           "abstenção correta conta como acerto")
    checar(not verificar("O faturamento foi de 4,2 milhões.", t_abs),
           "inventar em vez de se abster conta como erro")


# --------------------------------------------------------------------------- #
# 10. Conjunto de tarefas
# --------------------------------------------------------------------------- #


def testar_tarefas() -> None:
    secao("10. Conjunto de avaliação")

    tarefas = carregar_tarefas()
    checar(len(tarefas) == 10, "dez tarefas carregadas")

    ids = [t.id for t in tarefas]
    checar(len(set(ids)) == len(ids), "ids únicos")

    validos = {"numerico", "contem", "exato", "abstencao"}
    checar(all(t.verificacao in validos for t in tarefas),
           "todo tipo de verificação é conhecido")

    registro = registro_padrao()
    for t in tarefas:
        for ferramenta in t.ferramentas_esperadas:
            checar(ferramenta in registro,
                   f"{t.id} espera uma ferramenta que existe: {ferramenta}")

    abstencoes = [t for t in tarefas if t.verificacao == "abstencao"]
    checar(len(abstencoes) == 1, "há exatamente uma tarefa de abstenção")

    # Uma tarefa numérica sem gabarito nunca poderia ser acertada.
    checar(all(t.esperado for t in tarefas if t.verificacao == "numerico"),
           "toda tarefa numérica tem gabarito")

    dificuldades = {t.dificuldade for t in tarefas}
    checar(dificuldades <= {"baixa", "media", "alta"},
           f"dificuldades conhecidas: {sorted(dificuldades)}")

    multi = [t for t in tarefas if len(t.ferramentas_esperadas) >= 2]
    checar(len(multi) >= 5,
           "pelo menos metade das tarefas exige combinar duas ferramentas ou mais")


# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 68)
    print("  Testes offline — módulo 05, agente ReAct")
    print("  Nenhuma chamada de API, nenhum custo.")
    print("=" * 68)

    testar_calculadora()
    testar_arquivos()
    testar_conversao()
    testar_bm25()
    testar_registro()
    testar_memoria()
    testar_parser()
    testar_taxonomia()
    testar_verificacao()
    testar_tarefas()

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
