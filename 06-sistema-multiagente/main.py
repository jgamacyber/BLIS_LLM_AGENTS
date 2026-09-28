"""
CLI do módulo 06: sistema multiagente com supervisor e especialistas.

Comandos que rodam SEM chave de API e sem custo:

    python main.py config         mostra a configuração ativa
    python main.py rag            exercita o RAG usado como ferramenta
    python main.py time           mostra os papéis e quem tem qual ferramenta
    python main.py estado         demonstra o quadro compartilhado
    python main.py salvaguardas   demonstra os quatro modos de falha do CAMEL
    python main.py tarefas        lista o conjunto de avaliação
    python main.py prompt         imprime os prompts do supervisor e dos agentes

Comandos que chamam o modelo pelo OpenRouter (custam tokens):

    python main.py sistema "sua pergunta"
    python main.py sistema "sua pergunta" --sem-salvaguardas
    python main.py avaliar
    python main.py avaliar --tarefas m01,m06
    python main.py avaliar --comparar

Os custos por comando estão em REPRODUTIBILIDADE.md.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import salvaguardas
from agentes import PROMPT_ESPECIALISTA, Agente
from avaliacao import (
    Relatorio,
    avaliar_execucao,
    carregar_tarefas,
    citou_fonte,
    comparar,
)
from config import SETTINGS, resumo_config
from estado import EstadoCompartilhado
from ferramentas import FERRAMENTAS_POR_PAPEL, registro_para
from rag import obter_rag
from supervisor import PERFIS, PROMPT_SUPERVISOR, montar_sistema, montar_time

DIR_SAIDA = Path("saidas")


def titulo(texto: str) -> None:
    print(f"\n{'=' * 74}\n  {texto}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
# config e time
# --------------------------------------------------------------------------- #


def cmd_config(_args: argparse.Namespace) -> int:
    titulo("Configuração")
    print(f"\n  {resumo_config()}")
    print(f"\n  max_auto_resposta:          {SETTINGS.max_auto_resposta} "
          f"(idas e voltas com o mesmo agente)")
    print(f"  max_passos_ferramenta:      {SETTINGS.max_passos_ferramenta} "
          f"(por especialista, por handoff)")
    print(f"  max_repeticao_especialista: {SETTINGS.max_repeticao_especialista}")
    print(f"  max_salvaguardas:           {SETTINGS.max_salvaguardas}")

    rag = obter_rag()
    print(f"\n  corpus: {len(rag.documentos())} documentos, {len(rag)} trechos")

    if not SETTINGS.tem_chave:
        print("\n  Sem OPENROUTER_API_KEY. Os comandos rag, time, estado,")
        print("  salvaguardas, tarefas e prompt funcionam assim mesmo.")
    return 0


def cmd_time(_args: argparse.Namespace) -> int:
    titulo("O time e a divisão de papéis")

    estado = EstadoCompartilhado("tarefa de exemplo")
    for agente in montar_time(estado):
        print(f"\n  {agente.nome.upper()} ({agente.papel})")
        print(f"    {agente.descricao}")
        print(f"    ferramentas: {', '.join(agente.ferramentas.nomes())}")

    print("\n  O supervisor não aparece acima porque não tem ferramenta nenhuma:")
    print("  ele só decide quem trabalha e fecha a resposta.")

    titulo("Por que a divisão importa")
    print()
    print("  A tabela de ferramentas por papel é a divisão de papéis de verdade.")
    print("  O redator não tem como buscar nada: se um fato não estiver no quadro")
    print("  compartilhado, ele não tem de onde tirar, e o buraco fica visível em")
    print("  vez de ser preenchido com invenção. O pesquisador não calcula e o")
    print("  analista não lê o corpus, então nenhuma tarefa que cruze as duas")
    print("  fontes se resolve sem handoff.")
    print()
    for papel, ferramentas in FERRAMENTAS_POR_PAPEL.items():
        print(f"    {papel:<14} {', '.join(ferramentas) or '(nenhuma de coleta)'}")
    return 0


# --------------------------------------------------------------------------- #
# rag
# --------------------------------------------------------------------------- #

CONSULTAS_DEMO = [
    "quais são os quatro modos de falha do role-playing",
    "como o gerente escolhe quem fala na conversa em grupo",
    "para que serve o limite de respostas automáticas consecutivas",
    "o que é um pensamento no ReAct",
    "qual a receita de bolo de cenoura",
]


def cmd_rag(_args: argparse.Namespace) -> int:
    titulo("O RAG usado como ferramenta")

    rag = obter_rag()
    print(f"\n  {len(rag.documentos())} documentos, {len(rag)} trechos indexados.")
    for doc in rag.documentos():
        n = sum(1 for c in rag.chunks if c.doc == doc)
        print(f"    {doc:<12} {n:>2} trechos — {rag.titulo_de(doc)}")

    for consulta in CONSULTAS_DEMO:
        print(f"\n  {'─' * 70}")
        print(f"  consulta: {consulta}")
        resultados = rag.consultar(consulta, top_k=2)
        if not resultados:
            print("  (nenhum trecho; a ferramenta manda o agente admitir que não sabe)")
            continue
        for chunk, score in resultados:
            print(f"    [{score:5.2f}] {chunk.fonte}")
        primeiro = rag.formatar(resultados[:1], tamanho_trecho=260)
        for linha in primeiro.splitlines():
            print(f"      {linha}")

    print(f"\n  {'─' * 70}")
    print("  A última consulta não tem resposta no corpus e volta vazia de")
    print("  propósito: é o caso em que a ferramenta precisa dizer que não achou,")
    print("  em vez de devolver o trecho menos ruim como se servisse.")
    return 0


# --------------------------------------------------------------------------- #
# estado
# --------------------------------------------------------------------------- #


def cmd_estado(_args: argparse.Namespace) -> int:
    titulo("O quadro compartilhado")

    estado = EstadoCompartilhado("Resumir os pedidos da Vetrix com desconto")
    estado.rodada = 1

    print("\n  Dois especialistas escrevem no quadro, cada um com a sua fonte:\n")
    print("   ", estado.registrar_fato(
        "o pedido 4472 tem 12 itens de R$ 45,00", "analista", "pedidos.csv"))
    print("   ", estado.registrar_fato(
        "a Vetrix tem 15% de desconto em contrato", "analista", "notas.md"))
    print("   ", estado.registrar_fato(
        "o CAMEL lista quatro modos de falha do role-playing",
        "pesquisador", "camel#Os quatro modos de falha"))

    print("\n  O mesmo fato registrado de novo não duplica:")
    print("   ", estado.registrar_fato(
        "o pedido 4472 tem 12 itens de R$ 45,00", "redator", "ouvi dizer"))

    estado.rodada = 2
    print("\n  Fatos no quadro:")
    print(estado.fatos_formatados())

    print("\n  O redator lê o quadro e escreve o artefato:")
    print("   ", estado.escrever_artefato(
        "resumo", "Vetrix: 4472 com desconto sai por R$ 459,00.", "redator"))
    print("    leitura de volta:", estado.ler_artefato("resumo"))
    print("    artefato que não existe:", estado.ler_artefato("inexistente"))

    estado.registrar_handoff("supervisor", "analista", "levante os pedidos da Vetrix")
    estado.registrar_mensagem("analista", "supervisor", "apurado, veja o quadro",
                              tipo="devolucao")

    titulo("O que o supervisor vê antes de decidir")
    print()
    print(estado.resumo())

    estado.imprimir()

    DIR_SAIDA.mkdir(exist_ok=True)
    caminho = estado.salvar(DIR_SAIDA / "estado_demo.json")
    print(f"\n  Estado completo salvo em {caminho}")

    print("\n  Repare no que NÃO é transmitido: a fala de cada agente fica no")
    print("  histórico, e o que circula entre eles são os fatos com procedência.")
    print("  Copiar toda fala para todos é o que faz o contexto explodir.")
    return 0


# --------------------------------------------------------------------------- #
# salvaguardas
# --------------------------------------------------------------------------- #

CASOS_SALVAGUARDA = [
    (
        "inversão de papéis",
        "Você precisa buscar o arquivo de pedidos e me informe o total apurado.",
        salvaguardas.Contexto(agente="analista",
                              instrucao_recebida="calcule o total do pedido 4471"),
    ),
    (
        "repetição da instrução",
        "Entendi: devo calcular o total do pedido 4471 somando os itens e o frete.",
        salvaguardas.Contexto(agente="analista",
                              instrucao_recebida="calcule o total do pedido 4471 "
                                                 "somando os itens e o frete"),
    ),
    (
        "resposta evasiva",
        "Vou calcular o total do pedido agora mesmo e retorno com o número.",
        salvaguardas.Contexto(agente="analista",
                              instrucao_recebida="calcule o total do pedido 4471"),
    ),
    (
        "laço infinito",
        "Muito obrigado pela colaboração! Fico à disposição.",
        salvaguardas.Contexto(agente="redator",
                              instrucao_recebida="consolide os fatos"),
    ),
    (
        "resposta legítima (não deve disparar nada)",
        "O total do pedido 4471 é R$ 318,00: 3 itens de R$ 98,50 mais R$ 22,50 "
        "de frete. [fonte: pedidos.csv]",
        salvaguardas.Contexto(agente="analista",
                              instrucao_recebida="calcule o total do pedido 4471",
                              ferramentas_usadas=["ler_arquivo", "calculadora"],
                              fatos_registrados=1),
    ),
    (
        "promessa seguida de entrega (não deve disparar)",
        "Vou calcular: 3 x 98,50 = 295,50, mais 22,50 de frete, total R$ 318,00. "
        "[fonte: pedidos.csv]",
        salvaguardas.Contexto(agente="analista",
                              instrucao_recebida="calcule o total do pedido 4471",
                              ferramentas_usadas=["calculadora"],
                              fatos_registrados=1),
    ),
]


def cmd_salvaguardas(_args: argparse.Namespace) -> int:
    titulo("Os quatro modos de falha do CAMEL, detectados fora do prompt")

    for rotulo, mensagem, contexto in CASOS_SALVAGUARDA:
        deteccao = salvaguardas.avaliar(mensagem, contexto)
        print(f"\n  {rotulo}")
        print(f"    mensagem: {mensagem[:96]}")
        trabalho = (f"{len(contexto.ferramentas_usadas)} ferramentas, "
                    f"{contexto.fatos_registrados} fatos")
        print(f"    trabalho feito: {trabalho}")
        if deteccao is None:
            print("    -> nada detectado")
        else:
            print(f"    -> {deteccao.tipo}: {deteccao.evidencia}")
            print(f"       correção: {deteccao.correcao[:110]}")

    titulo("O critério dos detectores")
    print()
    print("  Os dois últimos casos são o que separa um detector útil de um")
    print("  detector barulhento. O quinto tem vocabulário parecido com a")
    print("  instrução, e ainda assim não é repetição, porque houve trabalho.")
    print("  O sexto começa com 'vou calcular', que é o padrão da evasiva, e")
    print("  ainda assim entrega o número na mesma mensagem.")
    print()
    print("  Daí a evidência dupla: o padrão textual E a ausência de trabalho.")
    print("  São heurísticas sobre texto, então erram nos dois sentidos; o")
    print("  projeto prefere deixar passar a corrigir um agente que ia bem.")
    return 0


# --------------------------------------------------------------------------- #
# tarefas e prompt
# --------------------------------------------------------------------------- #


def cmd_tarefas(_args: argparse.Namespace) -> int:
    titulo("Conjunto de avaliação")

    tarefas = carregar_tarefas()
    for t in tarefas:
        esperados = ", ".join(t.especialistas_esperados) or "nenhum"
        print(f"\n  {t.id}  [{t.dificuldade}]  verificação: {t.verificacao}")
        print(f"        {t.tarefa}")
        print(f"        especialistas esperados: {esperados}")
        if t.nota:
            print(f"        gabarito: {t.nota}")

    multi = [t for t in tarefas if len(t.especialistas_esperados) >= 2]
    print(f"\n  {len(tarefas)} tarefas, {len(multi)} exigindo dois ou mais")
    print("  especialistas. A m08 não tem resposta em ferramenta nenhuma: ela")
    print("  mede se o sistema admite que não sabe em vez de inventar em grupo.")
    return 0


def cmd_prompt(_args: argparse.Namespace) -> int:
    estado = EstadoCompartilhado("exemplo de tarefa")
    time = montar_time(estado)

    titulo("Prompt do supervisor")
    print()
    print(PROMPT_SUPERVISOR.format(
        especialistas="\n".join(a.ficha() for a in time),
        situacao=estado.resumo(),
    ))

    titulo("Prompt de um especialista (o analista)")
    print()
    print(time[1].montar_prompt("calcule o total do pedido 4471"))
    return 0


# --------------------------------------------------------------------------- #
# sistema
# --------------------------------------------------------------------------- #


def exigir_chave() -> bool:
    if SETTINGS.tem_chave:
        return True
    print("\n  Este comando chama o modelo e precisa de OPENROUTER_API_KEY.")
    print("  Copie .env.example para .env e preencha a chave.")
    print("  Sem chave, rode: python main.py rag | estado | salvaguardas | time")
    return False


def cmd_sistema(args: argparse.Namespace) -> int:
    if not exigir_chave():
        return 1

    titulo("Sistema multiagente")
    print(f"\n  {resumo_config()}")
    if not args.sem_salvaguardas:
        print("  Salvaguardas do CAMEL ligadas.")
    else:
        print("  Salvaguardas DESLIGADAS: modo de comparação.")

    supervisor, estado = montar_sistema(
        args.tarefa, verboso=True, usar_salvaguardas=not args.sem_salvaguardas
    )
    execucao = supervisor.executar(args.tarefa, max_rodadas=args.rodadas)
    execucao.imprimir()

    if execucao.resposta:
        marca = "sim" if citou_fonte(execucao.resposta) else "NÃO"
        print(f"\n  A resposta final cita fonte: {marca}")

    DIR_SAIDA.mkdir(exist_ok=True)
    caminho = execucao.salvar(DIR_SAIDA / "ultima_execucao.json")
    print(f"  Rastro completo em {caminho}")
    return 0 if execucao.sucesso else 1


# --------------------------------------------------------------------------- #
# avaliar
# --------------------------------------------------------------------------- #


def rodar_conjunto(tarefas, com_salvaguardas: bool, verboso: bool) -> Relatorio:
    relatorio = Relatorio(
        modelo=SETTINGS.model,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        com_salvaguardas=com_salvaguardas,
    )

    for i, tarefa in enumerate(tarefas, start=1):
        print(f"\n  [{i}/{len(tarefas)}] {tarefa.id}: {tarefa.tarefa[:88]}")
        # Um sistema novo por tarefa: quadro compartilhado limpo, senão o fato
        # de uma tarefa resolveria a próxima e a medição não valeria nada.
        supervisor, _estado = montar_sistema(
            tarefa.tarefa, verboso=verboso, usar_salvaguardas=com_salvaguardas
        )
        execucao = supervisor.executar(tarefa.tarefa)
        resultado = avaliar_execucao(execucao, tarefa)
        relatorio.resultados.append(resultado)

        marca = "ACERTOU" if resultado.acertou else "ERROU  "
        print(f"        {marca}  {resultado.termino}  "
              f"{resultado.rodadas} rodadas  "
              f"[{', '.join(resultado.especialistas_usados) or 'nenhum'}]"
              f"{'  fonte ok' if resultado.citou_fonte else ''}")
        print(f"        resposta: {resultado.resposta[:130]}")

    return relatorio


def cmd_avaliar(args: argparse.Namespace) -> int:
    if not exigir_chave():
        return 1

    tarefas = carregar_tarefas()
    if args.tarefas:
        escolhidas = {t.strip() for t in args.tarefas.split(",")}
        tarefas = [t for t in tarefas if t.id in escolhidas]
        if not tarefas:
            print(f"\n  Nenhuma tarefa com id em {sorted(escolhidas)}.")
            return 1

    DIR_SAIDA.mkdir(exist_ok=True)

    titulo(f"Avaliação: {len(tarefas)} tarefas, com salvaguardas")
    print(f"\n  {resumo_config()}")
    com = rodar_conjunto(tarefas, com_salvaguardas=True, verboso=args.verboso)
    com.imprimir()
    com.imprimir_tabela()
    print(f"\n  Relatório salvo em {com.salvar(DIR_SAIDA / 'relatorio_com.json')}")

    if not args.comparar:
        return 0

    titulo(f"Avaliação: {len(tarefas)} tarefas, SEM salvaguardas")
    sem = rodar_conjunto(tarefas, com_salvaguardas=False, verboso=args.verboso)
    sem.imprimir()
    sem.imprimir_tabela()
    print(f"\n  Relatório salvo em {sem.salvar(DIR_SAIDA / 'relatorio_sem.json')}")

    print(comparar(com, sem))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sistema multiagente com supervisor — BLIS Trilha Prática, módulo 06",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando")

    sub.add_parser("config", help="mostra a configuração ativa")
    sub.add_parser("rag", help="exercita o RAG usado como ferramenta, sem custo")
    sub.add_parser("time", help="mostra os papéis e as ferramentas de cada um")
    sub.add_parser("estado", help="demonstra o quadro compartilhado, sem custo")
    sub.add_parser("salvaguardas", help="demonstra os modos de falha do CAMEL")
    sub.add_parser("tarefas", help="lista o conjunto de avaliação")
    sub.add_parser("prompt", help="imprime os prompts do sistema")

    p_sis = sub.add_parser("sistema", help="roda o sistema em uma tarefa")
    p_sis.add_argument("tarefa", help="a pergunta ou tarefa")
    p_sis.add_argument("--sem-salvaguardas", action="store_true",
                       dest="sem_salvaguardas",
                       help="desliga os detectores do CAMEL, para comparação")
    p_sis.add_argument("--rodadas", type=int, default=None,
                       help=f"limite de rodadas (padrão {SETTINGS.max_rodadas})")

    p_aval = sub.add_parser("avaliar", help="roda o conjunto inteiro e mede")
    p_aval.add_argument("--tarefas", default="",
                        help="ids separados por vírgula, ex: m01,m06")
    p_aval.add_argument("--comparar", action="store_true",
                        help="roda também sem salvaguardas e compara")
    p_aval.add_argument("--verboso", action="store_true",
                        help="imprime cada rodada de cada tarefa")

    return parser


COMANDOS = {
    "config": cmd_config,
    "rag": cmd_rag,
    "time": cmd_time,
    "estado": cmd_estado,
    "salvaguardas": cmd_salvaguardas,
    "tarefas": cmd_tarefas,
    "prompt": cmd_prompt,
    "sistema": cmd_sistema,
    "avaliar": cmd_avaliar,
}


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    if not args.comando:
        parser.print_help()
        print("\nComece por: python main.py time")
        return 0

    return COMANDOS[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
