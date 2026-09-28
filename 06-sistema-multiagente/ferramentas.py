"""
Ferramentas dos especialistas.

Cada especialista recebe um subconjunto diferente: é assim que a divisão de
papéis deixa de ser só uma frase no prompt e passa a ser uma restrição real.

- O **pesquisador** consulta o corpus através do RAG e não tem acesso aos
  arquivos operacionais nem à calculadora.
- O **analista** lê os arquivos e calcula, e não tem acesso ao corpus.
- O **redator** não tem nenhuma ferramenta de coleta: só lê o que os outros
  registraram no estado compartilhado e escreve o artefato final.

Isso torna o handoff necessário em vez de opcional. Um agente que pudesse fazer
tudo sozinho nunca exercitaria o mecanismo que o módulo quer demonstrar. E,
como o redator não consegue buscar nada, ele não tem como completar um buraco
com invenção sem que isso apareça: o fato simplesmente não está no quadro.

Segurança: a calculadora avalia a AST da expressão e nunca chama eval(), e a
leitura de arquivos é confinada a uma sandbox. Quem escolhe a string é o
modelo, então essas duas superfícies são de ataque, não de conveniência.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from estado import EstadoCompartilhado
from rag import obter_rag

DIR_SANDBOX = Path("data/arquivos")

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
# 5. RAG como ferramenta
# --------------------------------------------------------------------------- #


def consultar_documentos(entrada: str) -> Resultado:
    """
    Consulta o corpus através do RAG e devolve trechos com a fonte à vista.

    A observação inclui o rótulo [fonte: doc#secao] de cada trecho, que é o que
    o especialista copia ao registrar o fato. Sem isso, o fato chega ao redator
    como afirmação sem rastro.
    """
    pergunta = (entrada or "").strip()
    if not pergunta:
        return Resultado("", False, "consulta vazia")

    rag = obter_rag()
    if len(rag) == 0:
        return Resultado("", False, "o corpus está vazio")

    resultados = rag.consultar(pergunta, top_k=3)
    return Resultado(
        saida=rag.formatar(resultados),
        metadados={
            "n_resultados": len(resultados),
            "fontes": [c.fonte for c, _ in resultados],
        },
    )


def listar_documentos(entrada: str = "") -> Resultado:
    rag = obter_rag()
    if len(rag) == 0:
        return Resultado("", False, "o corpus está vazio")

    linhas = [f"{len(rag.documentos())} documentos, {len(rag)} trechos indexados:"]
    linhas += [f"- {doc}: {rag.titulo_de(doc)}" for doc in rag.documentos()]
    return Resultado("\n".join(linhas))


# --------------------------------------------------------------------------- #
# 6. Ferramentas do estado compartilhado
# --------------------------------------------------------------------------- #


def ferramentas_de_estado(estado: EstadoCompartilhado, autor: str) -> list[Ferramenta]:
    """
    Cria as ferramentas de leitura e escrita do quadro compartilhado, já ligadas
    ao agente que as vai usar.

    O autor é fixado aqui, e não vem da entrada do modelo: um agente não pode
    registrar um fato em nome de outro. Procedência que o próprio agente escolhe
    não é procedência.
    """

    def registrar_fato(entrada: str) -> Resultado:
        texto = (entrada or "").strip()
        if not texto:
            return Resultado("", False, "fato vazio")

        # Formato: "<fato> | fonte: <origem>". A fonte é opcional na sintaxe,
        # mas a ausência dela fica registrada e aparece no relatório.
        fonte = ""
        if "|" in texto:
            texto, resto = texto.split("|", 1)
            fonte = resto.strip()
            for prefixo in ("fonte:", "fonte"):
                if fonte.lower().startswith(prefixo):
                    fonte = fonte[len(prefixo):].strip(": ").strip()
                    break

        return Resultado(estado.registrar_fato(texto.strip(), autor, fonte))

    def ler_fatos(entrada: str = "") -> Resultado:
        achados = estado.buscar_fatos(entrada or "")
        if not achados:
            return Resultado("Nenhum fato apurado ainda sobre isso.")
        return Resultado("\n".join(f.formatar() for f in achados))

    def escrever_artefato(entrada: str) -> Resultado:
        if "|" not in (entrada or ""):
            return Resultado("", False,
                             "use 'nome | conteúdo' na entrada de escrever_artefato")
        nome, conteudo = entrada.split("|", 1)
        return Resultado(estado.escrever_artefato(nome.strip(), conteudo.strip(), autor))

    def ler_artefato(entrada: str) -> Resultado:
        return Resultado(estado.ler_artefato(entrada))

    return [
        Ferramenta(
            nome="registrar_fato",
            descricao="Registra no quadro compartilhado um fato que você apurou. "
                      "Formato: '<o fato> | fonte: <de onde veio>'. Sempre "
                      "informe a fonte.",
            funcao=registrar_fato,
            exemplo="registrar_fato | o pedido 4471 soma R$ 318,00 | fonte: pedidos.csv",
            categoria="estado",
        ),
        Ferramenta(
            nome="ler_fatos",
            descricao="Lê os fatos que os outros agentes já registraram. Entrada "
                      "vazia traz todos.",
            funcao=ler_fatos,
            exemplo="ler_fatos | frete",
            categoria="estado",
        ),
        Ferramenta(
            nome="escrever_artefato",
            descricao="Salva um resultado nomeado no quadro. Formato: "
                      "'<nome> | <conteúdo>'.",
            funcao=escrever_artefato,
            exemplo="escrever_artefato | relatorio | Resumo dos pedidos...",
            categoria="estado",
        ),
        Ferramenta(
            nome="ler_artefato",
            descricao="Lê um artefato salvo por outro agente, pelo nome.",
            funcao=ler_artefato,
            exemplo="ler_artefato | relatorio",
            categoria="estado",
        ),
    ]


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #


class RegistroFerramentas:
    """Conjunto de ferramentas disponíveis a um agente."""

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
        return "\n".join(f.ficha() for f in self._ferramentas.values())

    def executar(self, nome: str, entrada: str) -> Resultado:
        """
        Executa uma ferramenta pelo nome.

        Nome inválido devolve erro com a lista de nomes válidos, para o agente
        se corrigir. É o `Invalid Action` do AgentBench: seguiu o formato, mas
        escolheu uma ação que não existe.
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


# Ferramentas que não dependem do estado, prontas para montar cada papel.
FERRAMENTAS_BASE = {
    "calculadora": lambda: Ferramenta(
        nome="calculadora",
        descricao="Calcula uma expressão aritmética. Suporta + - * / ** % e "
                  "sqrt, log, abs, round, min, max, floor, ceil.",
        funcao=calculadora,
        exemplo="calculadora | (12 * 45.00) * 0.85",
        categoria="computacao",
    ),
    "converter_unidades": lambda: Ferramenta(
        nome="converter_unidades",
        descricao="Converte entre unidades de comprimento, massa, tempo e dados. "
                  "Formato: '<valor> <de> para <para>'.",
        funcao=converter_unidades,
        exemplo="converter_unidades | 197.6 kg para lb",
        categoria="computacao",
    ),
    "ler_arquivo": lambda: Ferramenta(
        nome="ler_arquivo",
        descricao="Lê um arquivo da pasta de dados operacionais.",
        funcao=ler_arquivo,
        exemplo="ler_arquivo | pedidos.csv",
        categoria="dados",
    ),
    "listar_arquivos": lambda: Ferramenta(
        nome="listar_arquivos",
        descricao="Lista os arquivos operacionais disponíveis.",
        funcao=listar_arquivos,
        exemplo="listar_arquivos |",
        categoria="dados",
    ),
    "data_hora": lambda: Ferramenta(
        nome="data_hora",
        descricao="Devolve a data e a hora atuais em UTC.",
        funcao=data_hora,
        exemplo="data_hora |",
        categoria="geral",
    ),
    "consultar_documentos": lambda: Ferramenta(
        nome="consultar_documentos",
        descricao="Consulta o corpus de documentos (RAG) e devolve os trechos "
                  "mais relevantes, cada um com a sua fonte.",
        funcao=consultar_documentos,
        exemplo="consultar_documentos | quais são os modos de falha do role-playing",
        categoria="conhecimento",
    ),
    "listar_documentos": lambda: Ferramenta(
        nome="listar_documentos",
        descricao="Lista os documentos do corpus, com título e número de trechos.",
        funcao=listar_documentos,
        exemplo="listar_documentos |",
        categoria="conhecimento",
    ),
}


# Quem pode o quê. Esta tabela É a divisão de papéis: mudar uma linha aqui muda
# o comportamento do sistema inteiro, inclusive a necessidade de handoff.
FERRAMENTAS_POR_PAPEL = {
    "pesquisador": ["consultar_documentos", "listar_documentos"],
    "analista": ["ler_arquivo", "listar_arquivos", "calculadora",
                 "converter_unidades", "data_hora"],
    "redator": [],
}


def registro_para(papel: str, estado: EstadoCompartilhado,
                  autor: str | None = None) -> RegistroFerramentas:
    """Monta o registro do papel, somando as ferramentas do estado."""
    nomes = FERRAMENTAS_POR_PAPEL.get(papel)
    if nomes is None:
        raise ValueError(
            f"papel desconhecido: '{papel}'. "
            f"Conhecidos: {', '.join(FERRAMENTAS_POR_PAPEL)}"
        )

    ferramentas = [FERRAMENTAS_BASE[n]() for n in nomes]
    ferramentas += ferramentas_de_estado(estado, autor or papel)
    return RegistroFerramentas(ferramentas)
