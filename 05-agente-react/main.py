"""
CLI do módulo 05: agente ReAct com ferramentas.

Comandos que rodam SEM chave de API e sem custo:

    python main.py config        mostra a configuração ativa
    python main.py ferramentas   exercita as sete ferramentas
    python main.py memoria       demonstra a hierarquia de memória do MemGPT
    python main.py tarefas       lista o conjunto de avaliação
    python main.py prompt        imprime o prompt do sistema montado

Comandos que chamam o modelo pelo OpenRouter (custam tokens):

    python main.py agente "sua pergunta"
    python main.py agente "sua pergunta" --memoria
    python main.py avaliar
    python main.py avaliar --tarefas t01,t04,t10

Os números de custo por comando estão em REPRODUTIBILIDADE.md.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from avaliacao import Relatorio, avaliar_execucao, carregar_tarefas
from config import SETTINGS, resumo_config
from ferramentas import registro_padrao
from memoria import MemoriaHierarquica
from react import PROMPT_SISTEMA, AgenteReAct, catalogo_com_memoria

DIR_SAIDA = Path("saidas")


def titulo(texto: str) -> None:
    print(f"\n{'=' * 74}\n  {texto}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def cmd_config(_args: argparse.Namespace) -> int:
    titulo("Configuração")
    print(f"\n  {resumo_config()}")
    print(f"\n  limiar de pressão:  {SETTINGS.tokens_pressao} tokens "
          f"({SETTINGS.limiar_pressao:.0%} da janela)")
    print(f"  fração do flush:    {SETTINGS.fracao_flush:.0%} da fila")
    print(f"  máx. erros formato: {SETTINGS.max_erros_formato}")
    print(f"  máx. repetições:    {SETTINGS.max_repeticoes}")

    registro = registro_padrao()
    print(f"\n  ferramentas ({len(registro)}): {', '.join(registro.nomes())}")

    if not SETTINGS.tem_chave:
        print("\n  Sem OPENROUTER_API_KEY. Os comandos ferramentas, memoria,")
        print("  tarefas e prompt funcionam assim mesmo.")
    return 0


# --------------------------------------------------------------------------- #
# ferramentas
# --------------------------------------------------------------------------- #

# Cada par é uma chamada real, escolhida para mostrar o contrato da ferramenta
# e não só o caso feliz: há uma divisão por zero e um caminho fora do sandbox.
DEMOS = [
    ("calculadora", "(3 * 98.50) + 22.50"),
    ("calculadora", "sqrt(144) + round(2.71828, 2)"),
    ("calculadora", "10 / 0"),
    ("converter_unidades", "197.6 kg para lb"),
    ("converter_unidades", "2.5 h para min"),
    ("listar_corpus", ""),
    ("buscar_corpus", "por que o pensamento nao gera observacao"),
    ("listar_arquivos", ""),
    ("ler_arquivo", "estoque.txt"),
    ("ler_arquivo", "../../config.py"),
    ("data_hora", ""),
]


def cmd_ferramentas(_args: argparse.Namespace) -> int:
    titulo("As sete ferramentas, exercitadas offline")

    registro = registro_padrao()
    falhas_esperadas = 0

    for nome, entrada in DEMOS:
        resultado = registro.executar(nome, entrada)
        marca = "ok " if resultado.sucesso else "ERR"
        print(f"\n  [{marca}] {nome}[{entrada}]")

        saida = resultado.como_observacao()
        for linha in saida.splitlines()[:8]:
            print(f"        {linha}")
        if len(saida.splitlines()) > 8:
            print(f"        ... (+{len(saida.splitlines()) - 8} linhas)")

        if not resultado.sucesso:
            falhas_esperadas += 1

    print(f"\n  {len(DEMOS)} chamadas, {falhas_esperadas} com erro tratado.")
    print("  Erro é dado, não exceção: o agente recebe o texto do erro como")
    print("  observação e pode corrigir o curso no passo seguinte.")

    titulo("Catálogo como o agente o vê")
    print()
    print(registro.catalogo())
    return 0


# --------------------------------------------------------------------------- #
# memoria
# --------------------------------------------------------------------------- #


def cmd_memoria(_args: argparse.Namespace) -> int:
    titulo("Memória hierárquica (MemGPT)")

    # Janela absurdamente pequena de propósito: o ponto da demonstração é ver a
    # pressão e o flush acontecerem em poucas mensagens, não simular um modelo
    # real. A janela de verdade está em SETTINGS.janela_contexto.
    janela = 150
    memoria = MemoriaHierarquica(
        janela_contexto=janela,
        limiar_pressao=SETTINGS.limiar_pressao,
        fracao_flush=SETTINGS.fracao_flush,
        instrucoes_sistema="Você é um agente de apoio ao time de logística.",
    )

    print(f"\n  Janela de {janela} tokens, aviso em 70%, flush em 100%.")
    print("  Vamos empurrar mensagens até o contexto estourar.\n")

    conversas = [
        ("usuario", "Preciso conferir os pedidos da semana passada."),
        ("agente", "Claro. O pedido 4471 tem 3 itens de R$ 98,50 e frete de R$ 22,50."),
        ("usuario", "E o 4472?"),
        ("agente", "O 4472 tem 12 itens de R$ 45,00, total de R$ 540,00 nos itens."),
        ("usuario", "Qual deles tem frete gratuito pela regra do time?"),
        ("agente", "A regra dá frete gratuito acima de R$ 300,00, então o 4472 tem."),
        ("usuario", "Anota que o cliente Vetrix Logística tem 15% de desconto."),
        ("agente", "Anotado: Vetrix Logística, 15% de desconto em contrato."),
        ("usuario", "Quantas unidades do SKU-1001 ainda estão no estoque?"),
        ("agente", "São 145 unidades, cada uma com 0,85 kg."),
        ("usuario", "E do SKU-1002?"),
        ("agente", "38 monitores, 5,20 kg cada um."),
        ("usuario", "Preciso do peso total dos monitores em libras."),
        ("agente", "38 x 5,20 dá 197,6 kg, que são aproximadamente 435,63 libras."),
    ]

    for i, (papel, conteudo) in enumerate(conversas, start=1):
        antes = memoria.ocupacao
        memoria.adicionar_mensagem(papel, conteudo, passo=i)
        depois = memoria.ocupacao
        aviso = ""
        if antes < memoria.limiar_pressao <= depois:
            aviso = "   <- aviso de pressão"
        if depois < antes:
            aviso = "   <- FLUSH: a fila foi despejada para o recall"
        print(f"  msg {i:>2}  ocupação {depois:>5.0%}{aviso}")

    memoria.imprimir_estado()

    # Fatos salvos à mão, como o agente faria com working_context_append.
    titulo("O agente escrevendo na própria memória")
    print("\n  working_context_append:")
    print("   ", memoria.working_context_append(
        "Vetrix Logística tem 15% de desconto em contrato", passo=99))
    print("\n  archival_insert:")
    print("   ", memoria.archival_insert(
        "SKU-1002: 38 monitores, 5,20 kg cada", rotulo="estoque", passo=99))

    titulo("Recuperando o que saiu do contexto")
    print("\n  recall_search | frete gratuito")
    print(memoria.recall_search("frete gratuito", passo=99))
    print("\n  archival_search | monitores")
    print(memoria.archival_search("monitores", passo=99))

    if memoria.resumo_recursivo:
        titulo("Resumo recursivo gerado no flush")
        print(f"\n{memoria.resumo_recursivo}")
        print("\n  Este resumo é extrativo e determinístico, para que o teste")
        print("  offline possa verificá-lo. O artigo usa o próprio LLM: a")
        print("  função resumir_com_llm faz isso quando há chave.")

    titulo("Main context montado (o que iria ao modelo)")
    print()
    print(memoria.montar_contexto())

    DIR_SAIDA.mkdir(exist_ok=True)
    caminho = memoria.salvar(DIR_SAIDA / "memoria_demo.json")
    print(f"\n  Estado completo salvo em {caminho}")
    return 0


# --------------------------------------------------------------------------- #
# tarefas e prompt
# --------------------------------------------------------------------------- #


def cmd_tarefas(_args: argparse.Namespace) -> int:
    titulo("Conjunto de avaliação")
    tarefas = carregar_tarefas()

    for t in tarefas:
        esperadas = ", ".join(t.ferramentas_esperadas) or "nenhuma"
        print(f"\n  {t.id}  [{t.dificuldade}]  verificação: {t.verificacao}")
        print(f"        {t.tarefa}")
        print(f"        ferramentas esperadas: {esperadas}")
        if t.nota:
            print(f"        gabarito: {t.nota}")

    print(f"\n  {len(tarefas)} tarefas.")
    print("  A t10 não tem resposta em nenhuma ferramenta: ela mede abstenção,")
    print("  ou seja, se o agente admite que não sabe em vez de inventar.")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    titulo("Prompt do sistema")
    registro = registro_padrao()
    catalogo = (
        catalogo_com_memoria(registro) if args.memoria else registro.catalogo()
    )
    print()
    print(PROMPT_SISTEMA.format(ferramentas=catalogo))
    return 0


# --------------------------------------------------------------------------- #
# agente
# --------------------------------------------------------------------------- #


def cmd_agente(args: argparse.Namespace) -> int:
    if not SETTINGS.tem_chave:
        print("\n  Este comando chama o modelo e precisa de OPENROUTER_API_KEY.")
        print("  Copie .env.example para .env e preencha a chave.")
        print("  Sem chave, rode: python main.py ferramentas | memoria | tarefas")
        return 1

    titulo("Agente ReAct")
    print(f"\n  {resumo_config()}")
    if args.memoria:
        print("  Modo com memória explícita: o agente decide o que salvar.")

    agente = AgenteReAct(verboso=True, usar_memoria=args.memoria)
    execucao = agente.executar(args.tarefa, max_passos=args.passos)
    execucao.imprimir(detalhado=False)
    agente.memoria.imprimir_estado()

    DIR_SAIDA.mkdir(exist_ok=True)
    import json

    caminho = DIR_SAIDA / "ultima_execucao.json"
    caminho.write_text(
        json.dumps(execucao.como_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Rastro completo em {caminho}")
    return 0 if execucao.sucesso else 1


# --------------------------------------------------------------------------- #
# avaliar
# --------------------------------------------------------------------------- #


def cmd_avaliar(args: argparse.Namespace) -> int:
    if not SETTINGS.tem_chave:
        print("\n  Este comando chama o modelo e precisa de OPENROUTER_API_KEY.")
        print("  Copie .env.example para .env e preencha a chave.")
        return 1

    tarefas = carregar_tarefas()
    if args.tarefas:
        escolhidas = {t.strip() for t in args.tarefas.split(",")}
        tarefas = [t for t in tarefas if t.id in escolhidas]
        if not tarefas:
            print(f"\n  Nenhuma tarefa com id em {sorted(escolhidas)}.")
            return 1

    titulo(f"Avaliação: {len(tarefas)} tarefas")
    print(f"\n  {resumo_config()}")

    relatorio = Relatorio(
        modelo=SETTINGS.model,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )

    for i, tarefa in enumerate(tarefas, start=1):
        print(f"\n  [{i}/{len(tarefas)}] {tarefa.id}: {tarefa.tarefa}")
        # Um agente novo por tarefa: nada de uma tarefa contaminar a próxima.
        agente = AgenteReAct(verboso=args.verboso, usar_memoria=args.memoria)
        execucao = agente.executar(tarefa.tarefa)
        resultado = avaliar_execucao(execucao, tarefa)
        relatorio.resultados.append(resultado)

        marca = "ACERTOU" if resultado.acertou else "ERROU  "
        ferramentas = ", ".join(resultado.ferramentas_usadas) or "nenhuma"
        print(f"        {marca}  {resultado.termino}  "
              f"{resultado.n_passos} passos  [{ferramentas}]")
        print(f"        resposta: {resultado.resposta[:120]}")

    relatorio.imprimir()
    relatorio.imprimir_tabela()
    relatorio.imprimir_utilidade()

    DIR_SAIDA.mkdir(exist_ok=True)
    caminho = relatorio.salvar(DIR_SAIDA / "relatorio.json")
    print(f"\n  Relatório salvo em {caminho}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Agente ReAct com ferramentas — BLIS Trilha Prática, módulo 05",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando")

    sub.add_parser("config", help="mostra a configuração ativa")
    sub.add_parser("ferramentas", help="exercita as ferramentas, sem custo")
    sub.add_parser("memoria", help="demonstra a memória hierárquica, sem custo")
    sub.add_parser("tarefas", help="lista o conjunto de avaliação, sem custo")

    p_prompt = sub.add_parser("prompt", help="imprime o prompt do sistema")
    p_prompt.add_argument("--memoria", action="store_true",
                          help="inclui as funções de memória no catálogo")

    p_agente = sub.add_parser("agente", help="roda o agente em uma tarefa")
    p_agente.add_argument("tarefa", help="a pergunta ou tarefa")
    p_agente.add_argument("--memoria", action="store_true",
                          help="dá ao agente as funções de memória")
    p_agente.add_argument("--passos", type=int, default=None,
                          help=f"limite de passos (padrão {SETTINGS.max_passos})")

    p_aval = sub.add_parser("avaliar", help="roda o conjunto inteiro e mede")
    p_aval.add_argument("--tarefas", default="",
                        help="ids separados por vírgula, ex: t01,t04,t10")
    p_aval.add_argument("--memoria", action="store_true",
                        help="dá ao agente as funções de memória")
    p_aval.add_argument("--verboso", action="store_true",
                        help="imprime cada passo de cada tarefa")

    return parser


COMANDOS = {
    "config": cmd_config,
    "ferramentas": cmd_ferramentas,
    "memoria": cmd_memoria,
    "tarefas": cmd_tarefas,
    "prompt": cmd_prompt,
    "agente": cmd_agente,
    "avaliar": cmd_avaliar,
}


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    if not args.comando:
        parser.print_help()
        print("\nComece por: python main.py ferramentas")
        return 0

    return COMANDOS[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
