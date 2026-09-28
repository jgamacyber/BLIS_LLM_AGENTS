"""
Ferramentas do agente: o espaço de ações externas.

No vocabulário do CoALA (Sumers et al., 2024), estas são as **ações de
grounding**: as que afetam o mundo fora do agente e produzem observações. As
ações internas (raciocínio, recuperação de memória, aprendizado) estão em
`memoria.py` e no loop de `react.py`.

Formalização da chamada, seguindo Toolformer (Schick et al., 2023): uma chamada
é uma tupla c = (a_c, i_c), onde a_c é o nome da ferramenta e i_c a entrada. O
artigo linearizava isso como `<API> a_c(i_c) → r </API>`; aqui usamos o formato
textual do ReAct (`Action:` / `Action Input:`), que é mais legível para quem lê
o rastro.

Todas as ferramentas são **determinísticas e offline**, de propósito:

- dá para testar o agente inteiro sem rede e sem custo;
- quando o agente erra, o erro é do agente, não de uma API instável;
- o mesmo rastro reproduz na sua máquina.

O único acesso a disco é uma sandbox restrita a `data/arquivos/`.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DIR_SANDBOX = Path("data/arquivos")
DIR_CORPUS = Path("data/corpus")


# --------------------------------------------------------------------------- #
# Estrutura de uma ferramenta
# --------------------------------------------------------------------------- #


@dataclass
class Resultado:
    """O que uma ferramenta devolve. Erro é dado, não exceção."""

    saida: str
    sucesso: bool = True
    erro: str = ""
    metadados: dict = field(default_factory=dict)

    def como_observacao(self) -> str:
        """Texto que volta ao agente como Observation."""
        if not self.sucesso:
            return f"ERRO: {self.erro}"
        return self.saida


@dataclass
class Ferramenta:
    """Uma ferramenta disponível ao agente."""

    nome: str
    descricao: str
    funcao: Callable[[str], Resultado]
    exemplo: str = ""
    categoria: str = "geral"

    def executar(self, entrada: str) -> Resultado:
        """
        Executa a ferramenta, convertendo qualquer exceção em Resultado.

        Um agente nunca deve travar porque uma ferramenta levantou exceção: ele
        precisa VER o erro como observação para poder corrigir o curso. É o
        mesmo princípio do feedback loop do MemGPT, em que erros de runtime
        voltam para o processador.
        """
        try:
            return self.funcao(entrada)
        except Exception as erro:  # noqa: BLE001
            return Resultado(saida="", sucesso=False, erro=f"{type(erro).__name__}: {erro}")

    def ficha(self) -> str:
        """Descrição da ferramenta para o prompt do agente."""
        linha = f"- {self.nome}: {self.descricao}"
        if self.exemplo:
            linha += f"\n    exemplo: {self.exemplo}"
        return linha


# --------------------------------------------------------------------------- #
# 1. Calculadora
# --------------------------------------------------------------------------- #

# Operadores permitidos. A avaliação é feita sobre a AST, NUNCA com eval():
# o agente escreve a expressão, e um `eval` aqui seria execução de código
# arbitrário decidida por um LLM.
_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCOES = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "min": min, "max": max, "sum": sum,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
}

_CONSTANTES = {"pi": math.pi, "e": math.e}


def _avaliar_no(no: ast.AST) -> float:
    """Avalia um nó da AST, recusando qualquer construção não prevista."""
    if isinstance(no, ast.Constant):
        if isinstance(no.value, (int, float)):
            return no.value
        raise ValueError(f"constante não numérica: {no.value!r}")

    if isinstance(no, ast.BinOp):
        tipo = type(no.op)
        if tipo not in _OPERADORES:
            raise ValueError(f"operador não permitido: {tipo.__name__}")
        esquerda, direita = _avaliar_no(no.left), _avaliar_no(no.right)
        if tipo in (ast.Div, ast.FloorDiv, ast.Mod) and direita == 0:
            raise ZeroDivisionError("divisão por zero")
        # Limita expoentes para não travar o processo com 9**9**9.
        if tipo is ast.Pow and (abs(direita) > 128 or abs(esquerda) > 1e6):
            raise ValueError("expoente ou base grande demais")
        return _OPERADORES[tipo](esquerda, direita)

    if isinstance(no, ast.UnaryOp):
        tipo = type(no.op)
        if tipo not in _OPERADORES:
            raise ValueError(f"operador unário não permitido: {tipo.__name__}")
        return _OPERADORES[tipo](_avaliar_no(no.operand))

    if isinstance(no, ast.Call):
        if not isinstance(no.func, ast.Name) or no.func.id not in _FUNCOES:
            nome = getattr(no.func, "id", "?")
            raise ValueError(f"função não permitida: {nome}")
        argumentos = [_avaliar_no(a) for a in no.args]
        return _FUNCOES[no.func.id](*argumentos)

    if isinstance(no, ast.Name):
        if no.id in _CONSTANTES:
            return _CONSTANTES[no.id]
        raise ValueError(f"nome desconhecido: {no.id}")

    raise ValueError(f"expressão não permitida: {type(no).__name__}")


def _avaliar_expressao(expressao: str) -> float:
    """Parseia e avalia, deixando o erro subir para quem chamou decidir."""
    arvore = ast.parse(expressao, mode="eval")
    return _avaliar_no(arvore.body)


# Vírgula entre dígitos, sem espaço em volta: o jeito brasileiro de escrever
# decimal. É esse o padrão trocado por ponto, e só como segunda tentativa.
_VIRGULA_DECIMAL = re.compile(r"(?<=\d),(?=\d)")


def calculadora(entrada: str) -> Resultado:
    """
    Avalia uma expressão aritmética com segurança.

    A vírgula é ambígua: em `98,50` ela separa a parte decimal, em
    `round(2.71828, 2)` ela separa argumentos. A estratégia é avaliar a
    expressão como veio e, só se isso falhar, reinterpretar as vírgulas entre
    dígitos como ponto decimal. Assim `round(x, 2)` continua funcionando e
    `(3 * 98,50) + 22,50`, que um agente raciocinando em português produz,
    também.
    """
    expressao = (entrada or "").strip().strip("`")
    if not expressao:
        return Resultado("", False, "expressão vazia")

    # Aceita '^' como potência, que é o que as pessoas (e os LLMs) escrevem.
    expressao = expressao.replace("^", "**")

    alternativa = _VIRGULA_DECIMAL.sub(".", expressao)
    tentativas = [expressao] if alternativa == expressao else [expressao, alternativa]

    valor = None
    primeiro_erro: Exception | None = None

    for candidata in tentativas:
        try:
            valor = _avaliar_expressao(candidata)
            expressao = candidata
            break
        except ZeroDivisionError:
            # Divisão por zero não é ambiguidade de vírgula: erra na hora.
            raise
        except SyntaxError as erro:
            primeiro_erro = primeiro_erro or ValueError(
                f"expressão inválida: {erro.msg}"
            )
        except Exception as erro:  # noqa: BLE001
            primeiro_erro = primeiro_erro or erro

    if valor is None:
        erro = primeiro_erro or ValueError("expressão inválida")
        if isinstance(erro, ValueError) and str(erro).startswith("expressão inválida"):
            return Resultado("", False, str(erro))
        raise erro

    if isinstance(valor, float):
        if valor != valor or valor in (float("inf"), float("-inf")):
            return Resultado("", False, "resultado não é um número finito")
        if abs(valor - round(valor)) < 1e-10:
            valor = round(valor)
        else:
            valor = round(valor, 10)

    return Resultado(saida=str(valor), metadados={"expressao": expressao})


# --------------------------------------------------------------------------- #
# 2. Busca no corpus (BM25)
# --------------------------------------------------------------------------- #


def _sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "em", "no", "na",
    "um", "uma", "e", "que", "com", "por", "para", "se", "ao", "aos", "as",
    "ou", "the", "of", "is", "to",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenizar(texto: str) -> list[str]:
    tokens = _TOKEN.findall(_sem_acentos(texto.lower()))
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class IndiceCorpus:
    """
    Índice BM25 sobre os documentos de `data/corpus/`.

    É a "memória semântica" do CoALA em forma mínima, e o gancho com o módulo
    06: lá, esta mesma busca vira a ferramenta que o especialista de pesquisa
    usa. Implementado do zero para manter o módulo sem dependências.
    """

    def __init__(self, diretorio: str | Path = DIR_CORPUS, k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.diretorio = Path(diretorio)
        self.k1, self.b = k1, b
        self.documentos: list[dict] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._carregar()

    def _carregar(self) -> None:
        if not self.diretorio.exists():
            return

        for caminho in sorted(self.diretorio.glob("*.md")):
            texto = caminho.read_text(encoding="utf-8", errors="replace")
            titulo = caminho.stem
            for linha in texto.splitlines():
                if linha.startswith("# "):
                    titulo = linha[2:].strip()
                    break
            tokens = _tokenizar(f"{titulo} {texto}")
            self.documentos.append({
                "id": caminho.stem,
                "titulo": titulo,
                "texto": texto,
                "tokens": tokens,
                "freq": Counter(tokens),
                "comprimento": len(tokens),
            })

        if not self.documentos:
            return

        self.avgdl = sum(d["comprimento"] for d in self.documentos) / len(self.documentos)

        df: Counter = Counter()
        for doc in self.documentos:
            df.update(doc["freq"].keys())
        n = len(self.documentos)
        self.idf = {
            termo: math.log((n - k + 0.5) / (k + 0.5) + 1.0) for termo, k in df.items()
        }

    def buscar(self, consulta: str, top_k: int = 3) -> list[dict]:
        tokens = _tokenizar(consulta)
        if not tokens or not self.documentos:
            return []

        pontuados = []
        for doc in self.documentos:
            score = 0.0
            norma = self.k1 * (1 - self.b + self.b * doc["comprimento"] / self.avgdl)
            for termo in tokens:
                f = doc["freq"].get(termo, 0)
                if f:
                    score += self.idf.get(termo, 0.0) * (f * (self.k1 + 1)) / (f + norma)
            if score > 0:
                pontuados.append((score, doc))

        pontuados.sort(key=lambda par: par[0], reverse=True)
        return [
            {"id": d["id"], "titulo": d["titulo"], "texto": d["texto"], "score": s}
            for s, d in pontuados[:top_k]
        ]

    def __len__(self) -> int:
        return len(self.documentos)


_INDICE: IndiceCorpus | None = None


def obter_indice(diretorio: str | Path = DIR_CORPUS) -> IndiceCorpus:
    """Índice carregado uma vez e reaproveitado."""
    global _INDICE
    if _INDICE is None or str(_INDICE.diretorio) != str(diretorio):
        _INDICE = IndiceCorpus(diretorio)
    return _INDICE


def _primeiro_trecho(texto: str, consulta: str, tamanho: int = 400) -> str:
    """Trecho do documento em torno do primeiro termo da consulta encontrado."""
    tokens = _tokenizar(consulta)
    texto_normalizado = _sem_acentos(texto.lower())

    melhor = -1
    for token in tokens:
        pos = texto_normalizado.find(token)
        if pos >= 0 and (melhor < 0 or pos < melhor):
            melhor = pos

    if melhor < 0:
        return texto[:tamanho].strip()

    inicio = max(0, melhor - tamanho // 3)
    return ("..." if inicio > 0 else "") + texto[inicio : inicio + tamanho].strip() + "..."


def buscar_corpus(entrada: str) -> Resultado:
    """Busca lexical no corpus local e devolve os trechos mais relevantes."""
    consulta = (entrada or "").strip()
    if not consulta:
        return Resultado("", False, "consulta vazia")

    indice = obter_indice()
    if len(indice) == 0:
        return Resultado("", False, f"corpus vazio ou ausente em {DIR_CORPUS}")

    achados = indice.buscar(consulta, top_k=3)
    if not achados:
        return Resultado(
            f"Nenhum documento do corpus menciona '{consulta}'. "
            f"O corpus tem {len(indice)} documentos sobre agentes e LLMs."
        )

    partes = []
    for i, doc in enumerate(achados, start=1):
        partes.append(
            f"[{i}] {doc['titulo']} (doc: {doc['id']}, score {doc['score']:.2f})\n"
            f"{_primeiro_trecho(doc['texto'], consulta)}"
        )

    return Resultado(
        saida="\n\n".join(partes),
        metadados={"n_achados": len(achados),
                   "docs": [d["id"] for d in achados]},
    )


def listar_corpus(entrada: str = "") -> Resultado:
    """Lista os documentos disponíveis no corpus."""
    indice = obter_indice()
    if len(indice) == 0:
        return Resultado("", False, f"corpus vazio ou ausente em {DIR_CORPUS}")
    linhas = [f"- {d['id']}: {d['titulo']}" for d in indice.documentos]
    return Resultado(f"{len(indice)} documentos no corpus:\n" + "\n".join(linhas))


# --------------------------------------------------------------------------- #
# 3. Data e hora
# --------------------------------------------------------------------------- #


def data_hora(entrada: str = "") -> Resultado:
    """
    Data e hora atuais em UTC.

    Ferramenta simples, mas conceitualmente importante: é conhecimento que o
    modelo não tem nos pesos e não pode inferir. O Toolformer usa exatamente um
    calendário entre suas ferramentas, pelo mesmo motivo.
    """
    agora = datetime.now(timezone.utc)
    dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
            "sexta-feira", "sábado", "domingo"]
    return Resultado(
        saida=(f"{agora.strftime('%d/%m/%Y %H:%M:%S')} UTC "
               f"({dias[agora.weekday()]})"),
        metadados={"iso": agora.isoformat()},
    )


# --------------------------------------------------------------------------- #
# 4. Sandbox de arquivos
# --------------------------------------------------------------------------- #


def _caminho_seguro(nome: str) -> Path:
    """
    Resolve um nome de arquivo dentro da sandbox, recusando escapes.

    O agente escolhe o nome do arquivo, então `../../etc/passwd` é uma entrada
    plausível — por acidente ou por injeção via conteúdo recuperado. A checagem
    é feita após `resolve()`, que é o que neutraliza `..` e links simbólicos.
    """
    if not nome or not nome.strip():
        raise ValueError("nome de arquivo vazio")

    limpo = nome.strip().strip("`\"'")
    raiz = DIR_SANDBOX.resolve()
    alvo = (DIR_SANDBOX / limpo).resolve()

    if not str(alvo).startswith(str(raiz) + "/") and alvo != raiz:
        raise ValueError(
            f"acesso negado: '{limpo}' está fora da pasta permitida "
            f"({DIR_SANDBOX})"
        )
    return alvo


def listar_arquivos(entrada: str = "") -> Resultado:
    """Lista os arquivos da sandbox."""
    if not DIR_SANDBOX.exists():
        return Resultado(f"A pasta {DIR_SANDBOX} não existe ou está vazia.")

    arquivos = sorted(p for p in DIR_SANDBOX.rglob("*") if p.is_file())
    if not arquivos:
        return Resultado(f"Nenhum arquivo em {DIR_SANDBOX}.")

    linhas = [
        f"- {p.relative_to(DIR_SANDBOX)} ({p.stat().st_size} bytes)" for p in arquivos
    ]
    return Resultado(f"{len(arquivos)} arquivo(s):\n" + "\n".join(linhas))


def ler_arquivo(entrada: str) -> Resultado:
    """Lê um arquivo de texto da sandbox."""
    caminho = _caminho_seguro(entrada)

    if not caminho.exists():
        disponiveis = (
            ", ".join(sorted(p.name for p in DIR_SANDBOX.glob("*") if p.is_file()))
            if DIR_SANDBOX.exists() else "(nenhum)"
        )
        return Resultado(
            "", False,
            f"arquivo não encontrado: {caminho.name}. Disponíveis: {disponiveis}"
        )

    if caminho.stat().st_size > 50_000:
        return Resultado("", False, "arquivo grande demais (limite de 50 KB)")

    texto = caminho.read_text(encoding="utf-8", errors="replace")
    return Resultado(texto, metadados={"arquivo": caminho.name, "chars": len(texto)})


# --------------------------------------------------------------------------- #
# 5. Conversão de unidades
# --------------------------------------------------------------------------- #

# Fatores para a unidade base de cada família.
_UNIDADES = {
    "comprimento": {"base": "m",
                    "fatores": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                                "pol": 0.0254, "pe": 0.3048, "mi": 1609.344}},
    "massa": {"base": "kg",
              "fatores": {"mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0,
                          "lb": 0.45359237, "oz": 0.028349523}},
    "tempo": {"base": "s",
              "fatores": {"ms": 0.001, "s": 1.0, "min": 60.0, "h": 3600.0,
                          "dia": 86400.0}},
    "dados": {"base": "byte",
              "fatores": {"byte": 1.0, "kb": 1024.0, "mb": 1024.0**2,
                          "gb": 1024.0**3, "tb": 1024.0**4}},
}

_ALIASES = {
    "metro": "m", "metros": "m", "quilometro": "km", "quilometros": "km",
    "km/h": "km", "centimetro": "cm", "centimetros": "cm", "polegada": "pol",
    "polegadas": "pol", "pes": "pe", "milha": "mi", "milhas": "mi",
    "grama": "g", "gramas": "g", "quilo": "kg", "quilos": "kg",
    "quilograma": "kg", "quilogramas": "kg", "libra": "lb", "libras": "lb",
    "tonelada": "t", "toneladas": "t", "onca": "oz",
    "segundo": "s", "segundos": "s", "minuto": "min", "minutos": "min",
    "hora": "h", "horas": "h", "dias": "dia",
    "bytes": "byte", "kib": "kb", "mib": "mb", "gib": "gb",
}


def _normalizar_unidade(u: str) -> str:
    u = _sem_acentos(u.strip().lower())
    return _ALIASES.get(u, u)


def converter_unidades(entrada: str) -> Resultado:
    """
    Converte entre unidades. Formato: "10 km para mi" ou "5 kg em lb".
    """
    texto = _sem_acentos((entrada or "").strip().lower()).replace(",", ".")
    padrao = re.compile(
        r"(-?\d+\.?\d*)\s*([a-z/]+)\s*(?:para|em|to|->|>)\s*([a-z/]+)"
    )
    match = padrao.search(texto)
    if not match:
        return Resultado(
            "", False,
            "formato inválido. Use algo como '10 km para mi' ou '5 kg em lb'."
        )

    valor = float(match.group(1))
    origem = _normalizar_unidade(match.group(2))
    destino = _normalizar_unidade(match.group(3))

    for familia, dados in _UNIDADES.items():
        fatores = dados["fatores"]
        if origem in fatores and destino in fatores:
            resultado = valor * fatores[origem] / fatores[destino]
            arredondado = round(resultado, 6)
            if abs(arredondado - round(arredondado)) < 1e-9:
                arredondado = round(arredondado)
            return Resultado(
                f"{valor} {origem} = {arredondado} {destino}",
                metadados={"familia": familia, "valor": resultado},
            )

    conhecidas = sorted({u for d in _UNIDADES.values() for u in d["fatores"]})
    return Resultado(
        "", False,
        f"não sei converter '{origem}' para '{destino}'. "
        f"Unidades conhecidas: {', '.join(conhecidas)}"
    )


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


class RegistroFerramentas:
    """Conjunto de ferramentas disponíveis ao agente."""

    def __init__(self, ferramentas: list[Ferramenta] | None = None) -> None:
        self._ferramentas: dict[str, Ferramenta] = {}
        for f in ferramentas or []:
            self.registrar(f)

    def registrar(self, ferramenta: Ferramenta) -> None:
        self._ferramentas[ferramenta.nome] = ferramenta

    def obter(self, nome: str) -> Ferramenta | None:
        return self._ferramentas.get((nome or "").strip())

    def nomes(self) -> list[str]:
        return list(self._ferramentas)

    def catalogo(self) -> str:
        """Descrição de todas as ferramentas, para o prompt do agente."""
        return "\n".join(f.ficha() for f in self._ferramentas.values())

    def executar(self, nome: str, entrada: str) -> Resultado:
        """
        Executa uma ferramenta pelo nome.

        Nome inválido devolve um Resultado com erro e a lista de nomes válidos.
        Isso corresponde ao `Invalid Action` da taxonomia do AgentBench: o
        agente seguiu o formato mas escolheu uma ação inexistente, e precisa
        VER isso para se corrigir.
        """
        ferramenta = self.obter(nome)
        if ferramenta is None:
            return Resultado(
                "", False,
                f"ferramenta '{nome}' não existe. "
                f"Disponíveis: {', '.join(self.nomes())}"
            )
        return ferramenta.executar(entrada)

    def __len__(self) -> int:
        return len(self._ferramentas)

    def __contains__(self, nome: str) -> bool:
        return nome in self._ferramentas


def registro_padrao() -> RegistroFerramentas:
    """As sete ferramentas funcionais do módulo."""
    return RegistroFerramentas([
        Ferramenta(
            nome="calculadora",
            descricao="Calcula uma expressão aritmética. Suporta + - * / ** % e "
                      "sqrt, log, abs, round, min, max, floor, ceil.",
            funcao=calculadora,
            exemplo="calculadora | (1250 * 0.15) + 300",
            categoria="computacao",
        ),
        Ferramenta(
            nome="buscar_corpus",
            descricao="Busca documentos no corpus local sobre agentes e LLMs. "
                      "Devolve os 3 trechos mais relevantes.",
            funcao=buscar_corpus,
            exemplo="buscar_corpus | o que é o loop ReAct",
            categoria="conhecimento",
        ),
        Ferramenta(
            nome="listar_corpus",
            descricao="Lista os documentos disponíveis no corpus, com seus títulos.",
            funcao=listar_corpus,
            exemplo="listar_corpus |",
            categoria="conhecimento",
        ),
        Ferramenta(
            nome="ler_arquivo",
            descricao="Lê um arquivo de texto da pasta de dados do agente.",
            funcao=ler_arquivo,
            exemplo="ler_arquivo | pedidos.csv",
            categoria="arquivos",
        ),
        Ferramenta(
            nome="listar_arquivos",
            descricao="Lista os arquivos disponíveis na pasta de dados.",
            funcao=listar_arquivos,
            exemplo="listar_arquivos |",
            categoria="arquivos",
        ),
        Ferramenta(
            nome="data_hora",
            descricao="Devolve a data e a hora atuais em UTC.",
            funcao=data_hora,
            exemplo="data_hora |",
            categoria="ambiente",
        ),
        Ferramenta(
            nome="converter_unidades",
            descricao="Converte entre unidades de comprimento, massa, tempo e "
                      "dados. Formato: '<valor> <de> para <para>'.",
            funcao=converter_unidades,
            exemplo="converter_unidades | 42 km para mi",
            categoria="computacao",
        ),
    ])
