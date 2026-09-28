"""
Avaliação do sistema multiagente.

A taxa de acerto sozinha diz pouco sobre um sistema com supervisor, porque um
sistema pode acertar pelo motivo errado: um especialista respondeu de cabeça,
sem consultar nada, e por sorte acertou. Por isso o relatório mede quatro
coisas diferentes:

1. **Acerto** — a resposta final bate com o gabarito.
2. **Roteamento** — os especialistas acionados são os que a tarefa exigia. Um
   handoff a mais custa dinheiro; um handoff a menos costuma significar que
   alguém respondeu sem ter a ferramenta para saber.
3. **Procedência** — a resposta final cita fonte, e os fatos do quadro têm
   autor e origem. Resposta certa sem procedência é resposta que ninguém pode
   auditar.
4. **Término** — na taxonomia do AgentBench, mais as detecções das salvaguardas
   do CAMEL.

A separação importa. Uma queda na taxa de acerto acompanhada de muitos TLE é um
problema de coordenação; acompanhada de muitas detecções de laço, é um problema
de comportamento dos agentes; com roteamento errado, é o prompt do supervisor
que está mal escrito. São três consertos diferentes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from supervisor import Execucao

CAMINHO_TAREFAS = Path("data/tarefas.json")


@dataclass
class Tarefa:
    id: str
    tarefa: str
    verificacao: str = "contem"
    esperado: object = ""
    especialistas_esperados: list[str] = field(default_factory=list)
    exige_fonte: bool = True
    dificuldade: str = "media"
    nota: str = ""


def carregar_tarefas(caminho: str | Path = CAMINHO_TAREFAS) -> list[Tarefa]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [
        Tarefa(
            id=d["id"],
            tarefa=d["tarefa"],
            verificacao=d.get("verificacao", "contem"),
            esperado=d.get("esperado", ""),
            especialistas_esperados=d.get("especialistas_esperados", []),
            exige_fonte=d.get("exige_fonte", True),
            dificuldade=d.get("dificuldade", "media"),
            nota=d.get("nota", ""),
        )
        for d in dados
    ]


# --------------------------------------------------------------------------- #
# Verificação
# --------------------------------------------------------------------------- #


def _sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", _sem_acentos(texto)).strip()


def _extrair_numeros(texto: str) -> list[float]:
    """
    Todos os números do texto, tolerando o formato pt-BR.

    A resposta final de um sistema multiagente é um parágrafo, não um número
    solto: ela costuma citar os valores intermediários antes do total. Por isso
    a verificação procura o alvo entre todos os números, em vez de exigir que
    ele seja o primeiro.
    """
    limpo = re.sub(r"[R$€£%]", " ", str(texto))
    achados = re.findall(r"-?\d[\d.,]*", limpo)
    numeros: list[float] = []

    for bruto in achados:
        bruto = bruto.rstrip(".,")
        if not bruto or bruto == "-":
            continue
        tem_ponto, tem_virgula = "." in bruto, "," in bruto

        if tem_ponto and tem_virgula:
            if bruto.rfind(",") > bruto.rfind("."):
                bruto = bruto.replace(".", "").replace(",", ".")
            else:
                bruto = bruto.replace(",", "")
        elif tem_virgula:
            partes = bruto.split(",")
            bruto = (bruto.replace(",", ".")
                     if len(partes) == 2 and len(partes[-1]) <= 2
                     else bruto.replace(",", ""))
        elif tem_ponto:
            partes = bruto.split(".")
            if len(partes) > 2 or (len(partes) == 2 and len(partes[-1]) == 3):
                bruto = bruto.replace(".", "")

        try:
            numeros.append(float(bruto))
        except ValueError:
            continue

    return numeros


PADROES_ABSTENCAO = [
    r"n[ãa]o\s+(sei|tenho|dispomos|disponho|consigo|posso|h[áa]|encontr|foi poss)",
    r"n[ãa]o\s+(é|e)\s+poss[íi]vel",
    r"sem\s+informa[çc][ãa]o",
    r"informa[çc][ãa]o\s+(insuficiente|indispon[íi]vel|n[ãa]o dispon)",
    r"nenhum[ao]?\s+(das\s+)?(ferramenta|especialista|fonte|documento)",
    r"n[ãa]o\s+(est[áa]|esta|consta|aparece)\s+(dispon|em nenhum)",
    r"n[ãa]o\s+(temos|h[áa])\s+(esse|essa|este|esta)\s+dado",
]


def eh_abstencao(texto: str) -> bool:
    return any(re.search(p, _sem_acentos(texto)) for p in PADROES_ABSTENCAO)


# Marcas de procedência que contam como citação de fonte na resposta final.
_MARCAS_FONTE = [
    r"\[fonte\s*:",
    r"\bfonte\s*:",
    r"\bsegundo (o|a) (artigo|documento|corpus|planilha|arquivo|notas)",
    r"\b(pedidos\.csv|estoque\.txt|notas\.md)\b",
    r"\b(camel|autogen|react|memgpt|coala|toolformer)#",
    r"\bcom base (n[oa]s?|em) (arquivo|planilha|documento|notas|corpus)",
]


def citou_fonte(texto: str) -> bool:
    normalizado = _sem_acentos(texto)
    return any(re.search(p, normalizado) for p in _MARCAS_FONTE)


def verificar(resposta: str, tarefa: Tarefa, tolerancia: float = 0.02) -> bool:
    resposta = resposta or ""

    if tarefa.verificacao == "abstencao":
        return eh_abstencao(resposta)

    if tarefa.verificacao == "numerico":
        alvos = _extrair_numeros(str(tarefa.esperado))
        if not alvos:
            return False
        alvo = alvos[0]
        margem = max(tolerancia, abs(alvo) * tolerancia)
        return any(abs(n - alvo) <= margem for n in _extrair_numeros(resposta))

    alternativas = (tarefa.esperado if isinstance(tarefa.esperado, list)
                    else [tarefa.esperado])
    normalizada = _normalizar(resposta)

    if tarefa.verificacao == "exato":
        return any(_normalizar(a) == normalizada for a in alternativas)

    return all(_normalizar(a) in normalizada for a in alternativas)


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoTarefa:
    tarefa_id: str
    tarefa: str
    resposta: str = ""
    acertou: bool = False
    termino: str = ""
    rodadas: int = 0
    duracao: float = 0.0
    tokens: int = 0
    especialistas_usados: list[str] = field(default_factory=list)
    especialistas_esperados: list[str] = field(default_factory=list)
    handoffs: int = 0
    fatos: int = 0
    fatos_com_fonte: int = 0
    citou_fonte: bool = False
    exigia_fonte: bool = True
    deteccoes: list[str] = field(default_factory=list)
    dificuldade: str = "media"

    @property
    def roteamento_correto(self) -> bool:
        """Acionou exatamente quem a tarefa exigia, sem sobra nem falta."""
        if not self.especialistas_esperados:
            return True
        return set(self.especialistas_esperados) == set(self.especialistas_usados)

    @property
    def roteamento_suficiente(self) -> bool:
        """Acionou pelo menos quem a tarefa exigia; pode ter acionado mais."""
        if not self.especialistas_esperados:
            return True
        return set(self.especialistas_esperados).issubset(set(self.especialistas_usados))

    @property
    def procedencia_ok(self) -> bool:
        return (not self.exigia_fonte) or self.citou_fonte


@dataclass
class Relatorio:
    resultados: list[ResultadoTarefa] = field(default_factory=list)
    modelo: str = ""
    timestamp: str = ""
    com_salvaguardas: bool = True

    @property
    def n(self) -> int:
        return len(self.resultados)

    @property
    def taxa_sucesso(self) -> float:
        return sum(r.acertou for r in self.resultados) / self.n if self.n else 0.0

    @property
    def taxa_roteamento(self) -> float:
        if not self.n:
            return 0.0
        return sum(r.roteamento_suficiente for r in self.resultados) / self.n

    @property
    def taxa_procedencia(self) -> float:
        exigem = [r for r in self.resultados if r.exigia_fonte]
        if not exigem:
            return 0.0
        return sum(r.citou_fonte for r in exigem) / len(exigem)

    def distribuicao_termino(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self.resultados:
            dist[r.termino] = dist.get(r.termino, 0) + 1
        return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))

    def distribuicao_deteccoes(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self.resultados:
            for tipo in r.deteccoes:
                dist[tipo] = dist.get(tipo, 0) + 1
        return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))

    def uso_especialistas(self) -> dict[str, int]:
        uso: dict[str, int] = {}
        for r in self.resultados:
            for nome in r.especialistas_usados:
                uso[nome] = uso.get(nome, 0) + 1
        return dict(sorted(uso.items(), key=lambda kv: kv[1], reverse=True))

    def custo_de_coordenacao(self) -> dict:
        """
        Quanto se paga pela coordenação, separado do trabalho útil.

        Um sistema multiagente é mais caro que um agente só, e vale a pena
        medir o quanto. Handoffs por tarefa e rodadas por tarefa são o preço;
        a taxa de acerto é o que se compra com ele.
        """
        if not self.n:
            return {}
        return {
            "rodadas_por_tarefa": sum(r.rodadas for r in self.resultados) / self.n,
            "handoffs_por_tarefa": sum(r.handoffs for r in self.resultados) / self.n,
            "tokens_por_tarefa": sum(r.tokens for r in self.resultados) / self.n,
            "fatos_por_tarefa": sum(r.fatos for r in self.resultados) / self.n,
        }

    # ------------------------------ impressão ------------------------------ #

    def imprimir(self) -> None:
        print(f"\n{'=' * 78}")
        modo = "com" if self.com_salvaguardas else "sem"
        print(f"  RELATÓRIO — {self.n} tarefas, modelo {self.modelo}, "
              f"{modo} salvaguardas")
        print("=" * 78)

        acertos = sum(r.acertou for r in self.resultados)
        print(f"\n  Acerto:      {self.taxa_sucesso:.1%} ({acertos}/{self.n})")
        print(f"  Roteamento:  {self.taxa_roteamento:.1%} das tarefas acionaram "
              f"pelo menos os especialistas necessários")
        print(f"  Procedência: {self.taxa_procedencia:.1%} das respostas que "
              f"exigiam fonte citaram uma")

        print("\n  Razões de término (AgentBench):")
        for razao, n in self.distribuicao_termino().items():
            print(f"    {razao:<10} {n:>3}  ({n / self.n:.0%})")

        deteccoes = self.distribuicao_deteccoes()
        if deteccoes:
            print("\n  Salvaguardas disparadas (modos de falha do CAMEL):")
            for tipo, n in deteccoes.items():
                print(f"    {tipo:<22} {n:>3}")
        else:
            print("\n  Nenhuma salvaguarda disparada.")

        print("\n  Especialistas acionados:")
        for nome, n in self.uso_especialistas().items():
            print(f"    {nome:<14} em {n} tarefa(s)")

        custo = self.custo_de_coordenacao()
        if custo:
            print("\n  Custo de coordenação, por tarefa:")
            print(f"    {custo['rodadas_por_tarefa']:.1f} rodadas, "
                  f"{custo['handoffs_por_tarefa']:.1f} handoffs, "
                  f"{custo['tokens_por_tarefa']:.0f} tokens, "
                  f"{custo['fatos_por_tarefa']:.1f} fatos registrados")

    def imprimir_tabela(self) -> None:
        print(f"\n  {'id':<6} {'ok':<4} {'rot':<5} {'fonte':<7} {'term':<9} "
              f"{'rod':<5} especialistas")
        print("  " + "-" * 74)
        for r in self.resultados:
            print(f"  {r.tarefa_id:<6} "
                  f"{'sim' if r.acertou else 'nao':<4} "
                  f"{'sim' if r.roteamento_suficiente else 'nao':<5} "
                  f"{('sim' if r.citou_fonte else 'nao') if r.exigia_fonte else '-':<7} "
                  f"{r.termino:<9} {r.rodadas:<5} "
                  f"{', '.join(r.especialistas_usados) or '-'}")

    def salvar(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps({
            "modelo": self.modelo,
            "timestamp": self.timestamp,
            "com_salvaguardas": self.com_salvaguardas,
            "taxa_sucesso": self.taxa_sucesso,
            "taxa_roteamento": self.taxa_roteamento,
            "taxa_procedencia": self.taxa_procedencia,
            "distribuicao_termino": self.distribuicao_termino(),
            "distribuicao_deteccoes": self.distribuicao_deteccoes(),
            "uso_especialistas": self.uso_especialistas(),
            "custo_de_coordenacao": self.custo_de_coordenacao(),
            "resultados": [
                {
                    "tarefa_id": r.tarefa_id, "tarefa": r.tarefa,
                    "resposta": r.resposta, "acertou": r.acertou,
                    "termino": r.termino, "rodadas": r.rodadas,
                    "duracao": round(r.duracao, 3), "tokens": r.tokens,
                    "especialistas_usados": r.especialistas_usados,
                    "especialistas_esperados": r.especialistas_esperados,
                    "roteamento_suficiente": r.roteamento_suficiente,
                    "roteamento_correto": r.roteamento_correto,
                    "handoffs": r.handoffs, "fatos": r.fatos,
                    "fatos_com_fonte": r.fatos_com_fonte,
                    "citou_fonte": r.citou_fonte,
                    "deteccoes": r.deteccoes,
                    "dificuldade": r.dificuldade,
                }
                for r in self.resultados
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return caminho


def avaliar_execucao(execucao: Execucao, tarefa: Tarefa) -> ResultadoTarefa:
    """Converte uma execução do sistema em resultado avaliado."""
    estado = execucao.estado
    estatisticas = estado.estatisticas() if estado else {}

    return ResultadoTarefa(
        tarefa_id=tarefa.id,
        tarefa=tarefa.tarefa,
        resposta=execucao.resposta,
        acertou=verificar(execucao.resposta, tarefa),
        termino=execucao.termino.value,
        rodadas=execucao.rodadas,
        duracao=execucao.duracao,
        tokens=execucao.tokens_totais,
        especialistas_usados=execucao.especialistas_acionados,
        especialistas_esperados=tarefa.especialistas_esperados,
        handoffs=estatisticas.get("handoffs", 0),
        fatos=estatisticas.get("fatos", 0),
        fatos_com_fonte=estatisticas.get("fatos_com_fonte", 0),
        citou_fonte=citou_fonte(execucao.resposta),
        exigia_fonte=tarefa.exige_fonte,
        deteccoes=[d.tipo for d in execucao.deteccoes],
        dificuldade=tarefa.dificuldade,
    )


def comparar(com: Relatorio, sem: Relatorio) -> str:
    """
    Compara as duas execuções do conjunto, com e sem salvaguardas.

    Uma advertência sobre ler esta comparação: com oito tarefas e uma execução
    de cada, a diferença entre as duas taxas tem intervalo de confiança largo o
    bastante para incluir zero em quase qualquer resultado. Isso serve para ver
    o mecanismo funcionando e para gerar hipótese, não para concluir que as
    salvaguardas ajudam. Concluir isso exigiria repetições e um conjunto maior.
    """
    linhas = [
        "",
        "  Comparação com e sem salvaguardas",
        "  " + "-" * 60,
        f"  {'':<22}{'com':<12}{'sem':<12}",
        f"  {'acerto':<22}{com.taxa_sucesso:<12.1%}{sem.taxa_sucesso:<12.1%}",
        f"  {'roteamento':<22}{com.taxa_roteamento:<12.1%}{sem.taxa_roteamento:<12.1%}",
        f"  {'procedência':<22}{com.taxa_procedencia:<12.1%}{sem.taxa_procedencia:<12.1%}",
    ]

    custo_com = com.custo_de_coordenacao()
    custo_sem = sem.custo_de_coordenacao()
    if custo_com and custo_sem:
        linhas.append(
            f"  {'tokens/tarefa':<22}{custo_com['tokens_por_tarefa']:<12.0f}"
            f"{custo_sem['tokens_por_tarefa']:<12.0f}"
        )
        linhas.append(
            f"  {'rodadas/tarefa':<22}{custo_com['rodadas_por_tarefa']:<12.1f}"
            f"{custo_sem['rodadas_por_tarefa']:<12.1f}"
        )

    linhas += [
        "",
        f"  Detecções na execução com salvaguardas: "
        f"{sum(com.distribuicao_deteccoes().values())}",
        "",
        "  Com 8 tarefas e uma execução de cada, esta diferença não sustenta",
        "  conclusão sobre eficácia: o intervalo de confiança inclui zero em",
        "  quase qualquer resultado. Serve para ver o mecanismo e levantar",
        "  hipótese, não para afirmar que as salvaguardas melhoram o sistema.",
    ]
    return "\n".join(linhas)
