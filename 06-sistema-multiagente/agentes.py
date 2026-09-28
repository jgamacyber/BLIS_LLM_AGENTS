"""
Agente conversável, no sentido do AutoGen.

O artigo define o bloco básico por três propriedades: o agente é conversável
(sabe enviar e receber mensagens), é customizável (pode ser movido por modelo,
por código ou por pessoa) e tem estado (guarda o que trocou). A interface é
sempre a mesma: enviar, receber e gerar resposta.

A parte que mais rende na prática é a **lista de funções de resposta**. O
AutoGen permite registrar funções que são tentadas em ordem; a primeira que
devolver algo define a resposta daquele turno. Isso é o que o artigo chama de
controle por linguagem de programação, em contraste com o controle por
linguagem natural, que é escrever a regra dentro do prompt.

Aqui as funções padrão são três, nesta ordem:

1. **terminação** — se a mensagem recebida encerra a conversa, responde com um
   reconhecimento curto e não chama o modelo. Uma chamada economizada.
2. **cache** — se este agente já respondeu exatamente a esta instrução nesta
   execução, devolve a resposta anterior. Além do custo, evita o laço em que
   um supervisor confuso reencaminha a mesma coisa três vezes.
3. **modelo** — o caminho normal: um laço curto de ferramentas até o agente
   produzir a resposta.

Cada especialista roda um laço de ferramentas limitado, no espírito do ReAct
mas mais curto: ele existe para o especialista apurar o que lhe cabe, não para
resolver a tarefa inteira sozinho. Resolver a tarefa inteira é do sistema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from config import SETTINGS, Settings, get_client
from estado import EstadoCompartilhado
from ferramentas import RegistroFerramentas

# --------------------------------------------------------------------------- #
# Protocolo do especialista
# --------------------------------------------------------------------------- #

PROMPT_ESPECIALISTA = """Você é {nome}, {descricao}

Você faz parte de um time. Um supervisor coordena o trabalho e vai te passar
uma instrução por vez. Você NÃO coordena, NÃO delega e NÃO faz perguntas ao
supervisor: você executa a sua parte com as suas ferramentas e devolve o
resultado.

SUAS FERRAMENTAS:
{ferramentas}

FORMATO OBRIGATÓRIO. A cada vez, produza EXATAMENTE um destes dois blocos:

Para usar uma ferramenta:
PASSO: <o que você quer descobrir com esta chamada>
FERRAMENTA: <nome exato da ferramenta>
ENTRADA: <a entrada da ferramenta>

Para encerrar a sua vez:
RESPOSTA: <o que você apurou, com os números e as fontes>

REGRAS:
1. Um bloco por vez, e pare. O sistema executa e devolve "RESULTADO:".
2. Nunca invente um resultado de ferramenta. Espere o do sistema.
3. Antes de dar a RESPOSTA, registre o que apurou com registrar_fato, sempre
   informando a fonte. O que não estiver no quadro compartilhado não chega aos
   outros agentes.
4. Se as suas ferramentas não alcançarem o que foi pedido, diga isso na
   RESPOSTA. Não invente e não peça para outro agente fazer.
5. Não prometa fazer: faça na mesma resposta.

QUADRO COMPARTILHADO (fatos já apurados pelo time):
{fatos}

INSTRUÇÃO DO SUPERVISOR:
{instrucao}"""


_RE_PASSO = re.compile(r"PASSO\s*:\s*(.+?)(?=\n\s*(?:FERRAMENTA|RESPOSTA)\s*:|\Z)",
                       re.IGNORECASE | re.DOTALL)
_RE_FERRAMENTA = re.compile(r"FERRAMENTA\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE)
_RE_ENTRADA = re.compile(r"ENTRADA\s*:\s*(.+?)(?=\n\s*(?:PASSO|FERRAMENTA|RESPOSTA|RESULTADO)\s*:|\Z)",
                         re.IGNORECASE | re.DOTALL)
_RE_RESPOSTA = re.compile(r"RESPOSTA\s*:\s*(.+?)(?=\n\s*(?:PASSO|FERRAMENTA)\s*:|\Z)",
                          re.IGNORECASE | re.DOTALL)


@dataclass
class Acao:
    """O que o especialista pediu neste turno."""

    passo: str = ""
    ferramenta: str = ""
    entrada: str = ""
    resposta: str = ""
    erro_formato: str = ""

    @property
    def concluiu(self) -> bool:
        return bool(self.resposta)


def analisar_saida(texto: str) -> Acao:
    """
    Parser tolerante da saída do especialista.

    Aceita a forma de três linhas e a compacta `FERRAMENTA: nome[entrada]`, e
    corta qualquer "RESULTADO:" alucinado: se o modelo inventou o retorno da
    ferramenta, tudo que vem depois foi raciocinado sobre dado falso.
    """
    bruto = (texto or "").strip()
    if not bruto:
        return Acao(erro_formato="saída vazia")

    # Corta a alucinação do resultado, mas só quando ela vem depois de uma ação.
    corte = re.search(r"\n\s*RESULTADO\s*:", bruto, re.IGNORECASE)
    if corte and re.search(r"FERRAMENTA\s*:", bruto[:corte.start()], re.IGNORECASE):
        bruto = bruto[:corte.start()]

    acao = Acao()

    m = _RE_PASSO.search(bruto)
    if m:
        acao.passo = m.group(1).strip()

    m = _RE_FERRAMENTA.search(bruto)
    if m:
        alvo = m.group(1).strip().strip("`")
        compacta = re.match(r"^([\w_]+)\s*[\[(](.*)[\])]\s*$", alvo, re.DOTALL)
        if compacta:
            acao.ferramenta = compacta.group(1).strip()
            acao.entrada = compacta.group(2).strip()
        else:
            acao.ferramenta = alvo.split("|")[0].strip()
            if "|" in alvo:
                acao.entrada = alvo.split("|", 1)[1].strip()

    if not acao.entrada:
        m = _RE_ENTRADA.search(bruto)
        if m:
            acao.entrada = m.group(1).strip()

    m = _RE_RESPOSTA.search(bruto)
    if m:
        acao.resposta = m.group(1).strip()

    if not acao.ferramenta and not acao.resposta:
        acao.erro_formato = "nenhum bloco FERRAMENTA ou RESPOSTA reconhecido"

    return acao


# --------------------------------------------------------------------------- #
# Resultado de um turno
# --------------------------------------------------------------------------- #


@dataclass
class Resposta:
    """O que o agente produziu em um turno completo."""

    agente: str
    conteudo: str = ""
    ferramentas_usadas: list[str] = field(default_factory=list)
    fatos_registrados: int = 0
    passos: list[dict] = field(default_factory=list)
    tokens: int = 0
    origem: str = "modelo"       # qual função de resposta produziu
    erro: str = ""

    @property
    def houve_trabalho(self) -> bool:
        return bool(self.ferramentas_usadas) or self.fatos_registrados > 0


# --------------------------------------------------------------------------- #
# O agente
# --------------------------------------------------------------------------- #

MARCADORES_TERMINO = ("ENCERRAR", "TAREFA_CONCLUIDA", "CAMEL_TASK_DONE")


class Agente:
    """
    Agente conversável: envia, recebe e gera resposta.

    A customização acontece por funções de resposta registradas, tentadas em
    ordem até uma devolver algo, como no AutoGen.
    """

    def __init__(
        self,
        nome: str,
        papel: str,
        descricao: str,
        ferramentas: RegistroFerramentas,
        estado: EstadoCompartilhado,
        settings: Settings | None = None,
        verboso: bool = False,
    ) -> None:
        self.nome = nome
        self.papel = papel
        self.descricao = descricao
        self.ferramentas = ferramentas
        self.estado = estado
        self.settings = settings or SETTINGS
        self.verboso = verboso

        self.caixa_entrada: list[tuple[str, str]] = []     # (remetente, conteúdo)
        self.enviadas: list[str] = []
        self.cache: dict[str, Resposta] = {}
        self.tokens_gastos = 0

        # A lista do AutoGen: tentadas em ordem, a primeira que responder vence.
        self.funcoes_resposta: list[Callable[[str, "Agente"], Resposta | None]] = [
            resposta_por_termino,
            resposta_por_cache,
            resposta_do_modelo,
        ]

    # --------------------------- interface base ---------------------------- #

    def registrar_resposta(self, funcao, posicao: int = 0) -> None:
        """Registra uma função de resposta. Posição 0 a torna prioritária."""
        self.funcoes_resposta.insert(posicao, funcao)

    def receber(self, conteudo: str, remetente: str) -> None:
        self.caixa_entrada.append((remetente, conteudo))

    def enviar(self, conteudo: str, destinatario: str, tipo: str = "fala") -> None:
        self.enviadas.append(conteudo)
        self.estado.registrar_mensagem(self.nome, destinatario, conteudo, tipo)

    def gerar_resposta(self, instrucao: str) -> Resposta:
        """Percorre as funções registradas e devolve a primeira resposta."""
        for funcao in self.funcoes_resposta:
            resposta = funcao(instrucao, self)
            if resposta is not None:
                if resposta.conteudo and resposta.origem == "modelo":
                    self.cache[instrucao.strip()] = resposta
                return resposta

        return Resposta(agente=self.nome, conteudo="", origem="nenhuma",
                        erro="nenhuma função de resposta produziu resultado")

    # ------------------------------ auxiliares ----------------------------- #

    def ficha(self) -> str:
        """Como o supervisor vê este agente na hora de escolher quem fala."""
        return (f"- {self.nome} ({self.papel}): {self.descricao}\n"
                f"    ferramentas: {', '.join(self.ferramentas.nomes())}")

    def montar_prompt(self, instrucao: str) -> str:
        return PROMPT_ESPECIALISTA.format(
            nome=self.nome,
            descricao=self.descricao,
            ferramentas=self.ferramentas.catalogo(),
            fatos=self.estado.fatos_formatados(),
            instrucao=instrucao,
        )


# --------------------------------------------------------------------------- #
# Funções de resposta padrão
# --------------------------------------------------------------------------- #


def resposta_por_termino(instrucao: str, agente: Agente) -> Resposta | None:
    """Encerramento reconhecido sem gastar uma chamada ao modelo."""
    texto = (instrucao or "").upper()
    if any(marcador in texto for marcador in MARCADORES_TERMINO):
        return Resposta(
            agente=agente.nome,
            conteudo=f"{agente.nome}: encerrando a participação.",
            origem="termino",
        )
    return None


def resposta_por_cache(instrucao: str, agente: Agente) -> Resposta | None:
    """
    Mesma instrução, mesma resposta, sem pagar de novo.

    Não é só economia: um supervisor confuso reencaminha a mesma instrução
    várias vezes, e sem isso cada repetição custa uma chamada e produz uma
    resposta ligeiramente diferente, o que confunde ainda mais a coordenação.
    """
    chave = (instrucao or "").strip()
    anterior = agente.cache.get(chave)
    if anterior is None:
        return None

    return Resposta(
        agente=agente.nome,
        conteudo=anterior.conteudo,
        ferramentas_usadas=list(anterior.ferramentas_usadas),
        fatos_registrados=anterior.fatos_registrados,
        passos=list(anterior.passos),
        tokens=0,
        origem="cache",
    )


def resposta_do_modelo(instrucao: str, agente: Agente) -> Resposta:
    """
    Laço curto de ferramentas até o especialista produzir a RESPOSTA.

    O limite de passos é baixo de propósito: o especialista apura a parte dele,
    não a tarefa inteira. Estourar o limite não é crash, é resultado, e vira
    uma resposta parcial honesta com o que foi apurado até ali.
    """
    resposta = Resposta(agente=agente.nome)
    fatos_antes = len(agente.estado.fatos)

    conversa = [agente.montar_prompt(instrucao)]

    for passo in range(1, agente.settings.max_passos_ferramenta + 1):
        try:
            bruto, tokens = _chamar_modelo(agente, "\n\n".join(conversa))
        except Exception as erro:  # noqa: BLE001
            resposta.erro = str(erro)
            resposta.conteudo = (
                f"Não consegui concluir: falha ao consultar o modelo ({erro})."
            )
            return resposta

        resposta.tokens += tokens
        agente.tokens_gastos += tokens
        acao = analisar_saida(bruto)

        if agente.verboso:
            print(f"      {agente.nome} passo {passo}: "
                  f"{acao.ferramenta or 'RESPOSTA'} {acao.entrada[:50]}")

        if acao.erro_formato:
            conversa.append(
                "RESULTADO: sua saída não seguiu o formato. Produza exatamente "
                "'PASSO/FERRAMENTA/ENTRADA' ou 'RESPOSTA:'."
            )
            resposta.passos.append({"passo": passo, "erro_formato": acao.erro_formato})
            continue

        if acao.concluiu:
            resposta.conteudo = acao.resposta
            resposta.passos.append({"passo": passo, "resposta": acao.resposta})
            break

        resultado = agente.ferramentas.executar(acao.ferramenta, acao.entrada)
        observacao = resultado.como_observacao()

        if resultado.sucesso and acao.ferramenta not in resposta.ferramentas_usadas:
            resposta.ferramentas_usadas.append(acao.ferramenta)

        resposta.passos.append({
            "passo": passo,
            "ferramenta": acao.ferramenta,
            "entrada": acao.entrada,
            "sucesso": resultado.sucesso,
            "observacao": observacao[:400],
        })

        conversa.append(
            f"PASSO: {acao.passo}\nFERRAMENTA: {acao.ferramenta}\n"
            f"ENTRADA: {acao.entrada}\nRESULTADO: {observacao}"
        )
    else:
        # Estourou os passos sem RESPOSTA: devolve o que deu, sem fingir.
        resposta.conteudo = (
            f"Não fechei a apuração em {agente.settings.max_passos_ferramenta} "
            f"passos. Ferramentas usadas: "
            f"{', '.join(resposta.ferramentas_usadas) or 'nenhuma'}. "
            f"Veja no quadro os fatos que consegui registrar."
        )

    resposta.fatos_registrados = len(agente.estado.fatos) - fatos_antes
    return resposta


def _chamar_modelo(agente: Agente, conteudo: str) -> tuple[str, int]:
    client = get_client(agente.settings)
    resposta = client.chat.completions.create(
        model=agente.settings.model,
        messages=[{"role": "user", "content": conteudo}],
        temperature=agente.settings.temperature,
        max_tokens=agente.settings.max_tokens,
        stop=["\nRESULTADO:", "RESULTADO:"],
    )
    texto = (resposta.choices[0].message.content or "").strip()
    uso = getattr(resposta, "usage", None)
    tokens = ((getattr(uso, "prompt_tokens", 0) or 0)
              + (getattr(uso, "completion_tokens", 0) or 0))
    return texto, tokens
