"""
Avaliação do agente.

Métrica principal: **taxa de sucesso** (Success Rate), como no AgentBench
(Liu et al., 2024). Mas a taxa sozinha não diz nada acionável — um agente com
40% de sucesso pode estar falhando por três razões muito diferentes, e a
correção é diferente em cada caso:

- muitos **IF** (formato inválido): o prompt está mal especificado;
- muitos **IA** (ação inválida): o catálogo de ferramentas está confuso;
- muitos **TLE** (limite estourado): o agente não consegue planejar em vários
  turnos, ou entra em laço.

Por isso o relatório sempre traz a distribuição das razões de término ao lado
da taxa de sucesso.

Há também uma medida inspirada no **critério de utilidade do Toolformer**
(Schick et al., 2023). O artigo mantém uma chamada de API apenas quando ela
reduz a perda de predição. Aqui, sem acesso a logprobs do modelo avaliado,
a adaptação é comportamental: uma ferramenta é considerada útil quando as
tarefas que a usaram têm taxa de sucesso maior do que as que não a usaram.
É uma proxy grosseira, e está rotulada como tal.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from react import Execucao, Termino

CAMINHO_TAREFAS = Path("data/tarefas.json")


@dataclass
class Tarefa:
    id: str
    tarefa: str
    verificacao: str = "contem"
    esperado: object = ""
    ferramentas_esperadas: list[str] = field(default_factory=list)
    dificuldade: str = "media"
    nota: str = ""


def carregar_tarefas(caminho: str | Path = CAMINHO_TAREFAS) -> list[Tarefa]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Tarefas não encontradas: {caminho}")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return [
        Tarefa(
            id=d.get("id", f"t{i:02d}"),
            tarefa=d["tarefa"],
            verificacao=d.get("verificacao", "contem"),
            esperado=d.get("esperado", ""),
            ferramentas_esperadas=d.get("ferramentas_esperadas", []),
            dificuldade=d.get("dificuldade", "media"),
            nota=d.get("nota", ""),
        )
        for i, d in enumerate(dados)
    ]


# --------------------------------------------------------------------------- #
# Verificação
# --------------------------------------------------------------------------- #


def _sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalizar(texto: str) -> str:
    texto = _sem_acentos(str(texto).lower())
    return re.sub(r"\s+", " ", texto).strip(" .,;:!?\"'`")


def _extrair_numero(texto: str) -> float | None:
    """Extrai um número, tolerando formatos pt-BR e símbolos de moeda."""
    t = re.sub(r"[R$€£%\s ]", "", str(texto))
    match = re.search(r"-?\d[\d.,]*", t)
    if not match:
        return None

    # O casamento é ganancioso e engole a pontuação final da frase: em
    # "o total é R$ 318,00." ele traria "318,00.", e o ponto final seria lido
    # como separador decimal, virando 31800. Um agente termina quase toda
    # resposta com ponto, então isso reprovaria respostas corretas em silêncio.
    bruto = match.group(0).rstrip(".,")
    if not bruto or bruto == "-":
        return None

    tem_ponto, tem_virgula = "." in bruto, "," in bruto

    if tem_ponto and tem_virgula:
        if bruto.rfind(",") > bruto.rfind("."):
            bruto = bruto.replace(".", "").replace(",", ".")
        else:
            bruto = bruto.replace(",", "")
    elif tem_virgula:
        partes = bruto.split(",")
        bruto = (bruto.replace(",", ".") if len(partes) == 2 and len(partes[-1]) <= 2
                 else bruto.replace(",", ""))
    elif tem_ponto:
        partes = bruto.split(".")
        if len(partes) > 2 or (len(partes) == 2 and len(partes[-1]) == 3):
            bruto = bruto.replace(".", "")

    try:
        return float(bruto)
    except ValueError:
        return None


PADROES_ABSTENCAO = [
    r"n[ãa]o\s+(sei|tenho|disponho|consigo|posso|h[áa]|encontrei|foi poss)",
    r"n[ãa]o\s+(é|e)\s+poss[íi]vel",
    r"sem\s+informa[çc][ãa]o",
    r"informa[çc][ãa]o\s+(insuficiente|indispon[íi]vel|n[ãa]o dispon)",
    r"nenhuma\s+(das\s+)?ferramenta",
    r"n[ãa]o\s+(está|esta|consta)\s+dispon",
]


def eh_abstencao(texto: str) -> bool:
    normalizado = _sem_acentos(str(texto).lower())
    return any(re.search(p, normalizado) for p in PADROES_ABSTENCAO)


def verificar(resposta: str, tarefa: Tarefa, tolerancia: float = 0.02) -> bool:
    """Verifica a resposta do agente contra o gabarito da tarefa."""
    resposta = resposta or ""

    if tarefa.verificacao == "abstencao":
        return eh_abstencao(resposta)

    if tarefa.verificacao == "numerico":
        obtido = _extrair_numero(resposta)
        alvo = _extrair_numero(tarefa.esperado)
        if obtido is None or alvo is None:
            return False
        # Tolerância relativa: converter unidades introduz arredondamento.
        return abs(obtido - alvo) <= max(tolerancia, abs(alvo) * tolerancia)

    alternativas = (tarefa.esperado if isinstance(tarefa.esperado, list)
                    else [tarefa.esperado])
    normalizada = _normalizar(resposta)

    if tarefa.verificacao == "exato":
        return any(_normalizar(a) == normalizada for a in alternativas)

    # "contem": todas as alternativas listadas precisam aparecer, o que evita
    # dar acerto para uma resposta que menciona só um dos conceitos pedidos.
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
    n_passos: int = 0
    duracao: float = 0.0
    tokens: int = 0
    ferramentas_usadas: list[str] = field(default_factory=list)
    ferramentas_esperadas: list[str] = field(default_factory=list)
    dificuldade: str = "media"

    @property
    def cobriu_ferramentas(self) -> bool:
        """Usou todas as ferramentas que a tarefa esperava?"""
        if not self.ferramentas_esperadas:
            return True
        return set(self.ferramentas_esperadas).issubset(set(self.ferramentas_usadas))


@dataclass
class Relatorio:
    resultados: list[ResultadoTarefa] = field(default_factory=list)
    modelo: str = ""
    timestamp: str = ""

    @property
    def n(self) -> int:
        return len(self.resultados)

    @property
    def taxa_sucesso(self) -> float:
        return sum(r.acertou for r in self.resultados) / self.n if self.n else 0.0

    def distribuicao_termino(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self.resultados:
            dist[r.termino] = dist.get(r.termino, 0) + 1
        return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))

    def por_dificuldade(self) -> dict[str, tuple[int, int]]:
        agrupado: dict[str, list[ResultadoTarefa]] = {}
        for r in self.resultados:
            agrupado.setdefault(r.dificuldade, []).append(r)
        return {
            nivel: (sum(x.acertou for x in itens), len(itens))
            for nivel, itens in sorted(agrupado.items())
        }

    def uso_ferramentas(self) -> dict[str, int]:
        contagem: dict[str, int] = {}
        for r in self.resultados:
            for f in r.ferramentas_usadas:
                contagem[f] = contagem.get(f, 0) + 1
        return dict(sorted(contagem.items(), key=lambda kv: kv[1], reverse=True))

    def utilidade_ferramentas(self) -> dict[str, dict]:
        """
        Proxy comportamental do critério de utilidade do Toolformer.

        Compara a taxa de sucesso das tarefas em que a ferramenta foi usada
        com a das tarefas em que não foi. NÃO é o critério do artigo, que mede
        redução de perda em tokens; é uma aproximação por comportamento, e só
        é interpretável com um número razoável de tarefas em cada grupo.
        """
        todas = {f for r in self.resultados for f in r.ferramentas_usadas}
        saida: dict[str, dict] = {}

        for ferramenta in sorted(todas):
            com = [r for r in self.resultados if ferramenta in r.ferramentas_usadas]
            sem = [r for r in self.resultados if ferramenta not in r.ferramentas_usadas]
            if not com or not sem:
                continue
            taxa_com = sum(r.acertou for r in com) / len(com)
            taxa_sem = sum(r.acertou for r in sem) / len(sem)
            saida[ferramenta] = {
                "n_com": len(com),
                "n_sem": len(sem),
                "taxa_com": taxa_com,
                "taxa_sem": taxa_sem,
                "delta": taxa_com - taxa_sem,
            }
        return saida

    # ------------------------------ impressão ------------------------------ #

    def imprimir(self) -> None:
        print(f"\n{'=' * 78}")
        print(f"  RELATÓRIO  —  {self.n} tarefas, modelo {self.modelo}")
        print("=" * 78)

        print(f"\n  Taxa de sucesso: {self.taxa_sucesso:.1%} "
              f"({sum(r.acertou for r in self.resultados)}/{self.n})")

        print("\n  Razões de término (taxonomia do AgentBench):")
        for razao, n in self.distribuicao_termino().items():
            try:
                descricao = Termino(razao).descricao()
            except ValueError:
                descricao = razao
            print(f"    {razao:<10} {n:>3}  {descricao}")

        print("\n  Por dificuldade:")
        for nivel, (acertos, total) in self.por_dificuldade().items():
            taxa = acertos / total if total else 0.0
            print(f"    {nivel:<10} {acertos}/{total}  ({taxa:.0%})")

        uso = self.uso_ferramentas()
        if uso:
            print("\n  Uso de ferramentas:")
            for ferramenta, n in uso.items():
                print(f"    {ferramenta:<22} {n:>3} tarefa(s)")

        cobertura = [r for r in self.resultados if r.ferramentas_esperadas]
        if cobertura:
            cobriram = sum(r.cobriu_ferramentas for r in cobertura)
            print(f"\n  Cobertura de ferramentas esperadas: "
                  f"{cobriram}/{len(cobertura)} tarefas usaram todas as previstas")

        custo_medio = sum(r.tokens for r in self.resultados) / self.n if self.n else 0
        passos_medio = sum(r.n_passos for r in self.resultados) / self.n if self.n else 0
        print(f"\n  Média: {passos_medio:.1f} passos, {custo_medio:.0f} tokens por tarefa")

    def imprimir_tabela(self) -> None:
        print(f"\n  {'id':<6} {'ok':<4} {'término':<10} {'passos':>7} "
              f"{'tokens':>8}  ferramentas")
        print("  " + "-" * 74)
        for r in self.resultados:
            marca = "OK" if r.acertou else "X"
            ferramentas = ", ".join(r.ferramentas_usadas[:3]) or "nenhuma"
            print(f"  {r.tarefa_id:<6} {marca:<4} {r.termino:<10} {r.n_passos:>7} "
                  f"{r.tokens:>8}  {ferramentas[:36]}")

    def imprimir_utilidade(self) -> None:
        utilidade = self.utilidade_ferramentas()
        if not utilidade:
            print("\n  (sem dados suficientes para estimar utilidade das ferramentas)")
            return

        print("\n  Utilidade das ferramentas (proxy comportamental, não o")
        print("  critério de perda do Toolformer):")
        print(f"    {'ferramenta':<22} {'com':>10} {'sem':>10} {'delta':>9}")
        print("    " + "-" * 54)
        for ferramenta, d in sorted(utilidade.items(),
                                    key=lambda kv: kv[1]["delta"], reverse=True):
            print(f"    {ferramenta:<22} {d['taxa_com']:>9.0%} {d['taxa_sem']:>10.0%} "
                  f"{d['delta']:>+9.0%}")
        print("\n    Atenção: com poucas tarefas por grupo, esses deltas são ruído.")

    def salvar(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps({
            "modelo": self.modelo,
            "timestamp": self.timestamp,
            "taxa_sucesso": self.taxa_sucesso,
            "distribuicao_termino": self.distribuicao_termino(),
            "uso_ferramentas": self.uso_ferramentas(),
            "resultados": [
                {
                    "tarefa_id": r.tarefa_id, "tarefa": r.tarefa,
                    "resposta": r.resposta, "acertou": r.acertou,
                    "termino": r.termino, "n_passos": r.n_passos,
                    "duracao": round(r.duracao, 3), "tokens": r.tokens,
                    "ferramentas_usadas": r.ferramentas_usadas,
                    "dificuldade": r.dificuldade,
                }
                for r in self.resultados
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return caminho


def avaliar_execucao(execucao: Execucao, tarefa: Tarefa) -> ResultadoTarefa:
    """Converte uma execução em resultado avaliado."""
    return ResultadoTarefa(
        tarefa_id=tarefa.id,
        tarefa=tarefa.tarefa,
        resposta=execucao.resposta,
        acertou=verificar(execucao.resposta, tarefa),
        termino=execucao.termino.value,
        n_passos=execucao.n_passos,
        duracao=execucao.duracao,
        tokens=execucao.tokens_totais,
        ferramentas_usadas=execucao.ferramentas_usadas,
        ferramentas_esperadas=tarefa.ferramentas_esperadas,
        dificuldade=tarefa.dificuldade,
    )
