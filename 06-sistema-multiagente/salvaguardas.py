"""
As quatro salvaguardas do CAMEL, implementadas fora do prompt.

O artigo trata os quatro modos de falha com regras no prompt de concepção:
"nunca inverta papéis", "sempre comece com Solution:", "não prometa, faça".
Isso reduz a frequência, mas não elimina: o prompt é um pedido, não uma
garantia. Um sistema que depende só dele descobre a falha quando a conversa já
gastou dez rodadas trocando gentilezas.

Aqui as quatro falhas viram detectores que olham a mensagem produzida e
decidem. Cada detecção devolve uma correção, que é injetada de volta no agente,
e fica registrada no estado para aparecer no relatório.

**Sobre o que estes detectores são.** São heurísticas sobre texto, e por isso
erram nos dois sentidos: uma pergunta legítima de esclarecimento pode ser lida
como inversão de papéis, e uma evasiva bem escrita pode passar. O critério de
projeto foi preferir o falso negativo ao falso positivo, porque uma correção
injetada sem motivo atrapalha um agente que estava indo bem. Onde há dúvida, o
detector usa evidência dupla: o padrão textual E a ausência de trabalho feito
(nenhuma ferramenta usada, nenhum fato registrado). Prometer e entregar não é
evasiva; prometer e não entregar é.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _normalizar(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokens(texto: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalizar(texto)))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cobertura(mensagem: str, referencia: str) -> float:
    """Quanto da referência a mensagem cobre."""
    tm, tr = _tokens(mensagem), _tokens(referencia)
    if not tm or not tr:
        return 0.0
    return len(tm & tr) / len(tr)


def novidade(mensagem: str, referencia: str) -> float:
    """Que fração da mensagem é conteúdo que não estava na referência."""
    tm, tr = _tokens(mensagem), _tokens(referencia)
    if not tm:
        return 0.0
    return len(tm - tr) / len(tm)


@dataclass
class Contexto:
    """O que o detector precisa saber além do texto da mensagem."""

    agente: str
    instrucao_recebida: str = ""
    ferramentas_usadas: list[str] = field(default_factory=list)
    fatos_registrados: int = 0
    mensagens_anteriores: list[str] = field(default_factory=list)


@dataclass
class Deteccao:
    tipo: str
    agente: str
    evidencia: str
    correcao: str

    def formatar(self) -> str:
        return f"[salvaguarda: {self.tipo}] {self.agente} — {self.evidencia}"


# --------------------------------------------------------------------------- #
# 1. Inversão de papéis
# --------------------------------------------------------------------------- #

# Um especialista que devolve instrução ou pergunta, em vez de solução, assumiu
# o papel do supervisor. A conversa para de progredir porque ninguém executa.
_PADROES_INVERSAO = [
    r"\bvoc[eê] (deve|precisa|poderia|pode) (fazer|buscar|calcular|verificar|consultar)",
    r"\bme (informe|diga|passe|envie|forne[çc]a)\b",
    r"\bqual (é|e) (a|o) (pr[oó]xim[ao]|sua)\b",
    r"\bo que (voc[eê]|eu) (dev[oe]|preciso) fazer\b",
    r"^\s*handoff\s*:",
    r"\bdelego\b|\bdelegue\b",
]


def detectar_inversao_papeis(mensagem: str, ctx: Contexto) -> Deteccao | None:
    texto = _normalizar(mensagem)

    for padrao in _PADROES_INVERSAO:
        achado = re.search(padrao, texto, re.MULTILINE)
        if achado:
            return Deteccao(
                tipo="inversao_papeis",
                agente=ctx.agente,
                evidencia=f"instruiu ou perguntou em vez de executar: "
                          f"'{achado.group(0)[:60]}'",
                correcao=(
                    "Você é o especialista, não o supervisor. Não faça perguntas "
                    "nem dê instruções: execute com as ferramentas que você tem e "
                    "responda com o resultado. Se faltar informação que suas "
                    "ferramentas não alcançam, diga exatamente o que falta e "
                    "encerre sua vez."
                ),
            )

    # Mensagem que é só pergunta, sem nenhum trabalho feito: mesmo efeito.
    perguntas = mensagem.count("?")
    if perguntas >= 2 and not ctx.ferramentas_usadas and ctx.fatos_registrados == 0:
        return Deteccao(
            tipo="inversao_papeis",
            agente=ctx.agente,
            evidencia=f"{perguntas} perguntas e nenhuma ferramenta usada",
            correcao=(
                "Responda com o que suas ferramentas alcançam, em vez de "
                "devolver perguntas. Use as ferramentas antes de pedir ajuda."
            ),
        )
    return None


# --------------------------------------------------------------------------- #
# 2. Repetição da instrução
# --------------------------------------------------------------------------- #


def detectar_repeticao_instrucao(mensagem: str, ctx: Contexto,
                                 limiar_cobertura: float = 0.7,
                                 limiar_novidade: float = 0.45) -> Deteccao | None:
    """
    O agente devolve a instrução parafraseada, sem solução.

    A medida não é a similaridade simétrica entre os dois textos, e a razão é
    prática: uma instrução curta ("calcule o total do pedido 4471") tem poucos
    tokens, então qualquer palavra a mais na resposta derruba o Jaccard e a
    repetição passa batido. O que caracteriza a falha são duas coisas
    assimétricas: a mensagem **cobre** quase toda a instrução e **acrescenta**
    pouca coisa nova.

    Some-se a evidência de comportamento: nenhuma ferramenta usada e nenhum
    fato registrado. Uma resposta correta sobre o mesmo assunto compartilha
    vocabulário com a pergunta por natureza, e por isso a similaridade sozinha
    nunca bastaria.
    """
    if not ctx.instrucao_recebida:
        return None
    if ctx.ferramentas_usadas or ctx.fatos_registrados:
        return None

    cob = cobertura(mensagem, ctx.instrucao_recebida)
    nov = novidade(mensagem, ctx.instrucao_recebida)

    if cob < limiar_cobertura or nov > limiar_novidade:
        return None

    return Deteccao(
        tipo="repeticao_instrucao",
        agente=ctx.agente,
        evidencia=f"cobre {cob:.0%} da instrução acrescentando {nov:.0%} de "
                  f"conteúdo novo, sem trabalho feito",
        correcao=(
            "Você repetiu a instrução em vez de cumpri-la. Use uma ferramenta "
            "agora e responda com o resultado concreto que ela devolveu."
        ),
    )


# --------------------------------------------------------------------------- #
# 3. Resposta evasiva
# --------------------------------------------------------------------------- #

# O padrão que o artigo chama de "flake reply": a promessa no lugar da ação.
_PADROES_EVASIVA = [
    r"\bvou (buscar|consultar|calcular|verificar|analisar|ler|fazer|providenciar)",
    r"\b(irei|vou) (em seguida|agora|proceder)",
    r"\bpretendo\b|\bplanejo\b",
    r"\bposso ajudar com isso\b",
    r"\bassim que (poss[ií]vel|eu puder)\b",
    r"\bem breve (eu )?(te )?(retorno|respondo|envio)\b",
    r"\bestou (trabalhando|providenciando)\b",
]


def detectar_resposta_evasiva(mensagem: str, ctx: Contexto) -> Deteccao | None:
    # Prometer e entregar não é evasiva. Só a promessa sem entrega é.
    if ctx.ferramentas_usadas or ctx.fatos_registrados:
        return None

    texto = _normalizar(mensagem)
    for padrao in _PADROES_EVASIVA:
        achado = re.search(padrao, texto)
        if achado:
            return Deteccao(
                tipo="resposta_evasiva",
                agente=ctx.agente,
                evidencia=f"prometeu sem executar: '{achado.group(0)[:50]}'",
                correcao=(
                    "Não anuncie o que você vai fazer: faça. Chame a ferramenta "
                    "nesta mesma resposta e devolva o resultado dela."
                ),
            )

    # Resposta curta demais também é evasiva na prática, mas curto não é
    # sinônimo de vazio: "são 2 pedidos [fonte: pedidos.csv]" entrega um número
    # e uma procedência em 34 caracteres. O que caracteriza a evasiva é a
    # ausência de conteúdo verificável, não o tamanho.
    limpa = mensagem.strip()
    tem_numero = bool(re.search(r"\d", limpa))
    tem_fonte = bool(re.search(r"\[?fonte\s*:|segundo (o|a) ", _normalizar(limpa)))
    admite_limite = _normalizar(limpa).startswith(("erro", "nao ", "não "))

    if len(limpa) < 40 and not (tem_numero or tem_fonte or admite_limite):
        return Deteccao(
            tipo="resposta_evasiva",
            agente=ctx.agente,
            evidencia=f"resposta de {len(limpa)} caracteres, sem número, sem "
                      f"fonte e sem trabalho feito",
            correcao=(
                "Sua resposta não trouxe conteúdo. Use uma ferramenta e responda "
                "com o que ela devolveu."
            ),
        )
    return None


# --------------------------------------------------------------------------- #
# 4. Laço infinito
# --------------------------------------------------------------------------- #

_PADROES_CORTESIA = [
    r"\b(muito )?obrigad[oa]\b",
    r"\bde nada\b",
    r"\bfico (à|a) disposi[çc][ãa]o\b",
    r"\bqualquer coisa (é )?s[óo] (chamar|avisar)\b",
    r"\bat[ée] (logo|mais)\b",
    r"\bbom trabalho\b",
    r"\bexcelente trabalho\b",
]


def detectar_laco(mensagem: str, ctx: Contexto,
                  limiar_similaridade: float = 0.85) -> Deteccao | None:
    texto = _normalizar(mensagem)

    # Mensagem só de cortesia, sem nenhum trabalho: o laço educado do artigo.
    cortesias = sum(1 for p in _PADROES_CORTESIA if re.search(p, texto))
    if cortesias and not ctx.ferramentas_usadas and ctx.fatos_registrados == 0 \
            and len(mensagem.strip()) < 220:
        return Deteccao(
            tipo="laco_infinito",
            agente=ctx.agente,
            evidencia="mensagem apenas de cortesia, sem avanço na tarefa",
            correcao=(
                "Nada de agradecimentos nem despedidas. Ou execute o próximo "
                "passo da tarefa, ou declare que sua parte terminou."
            ),
        )

    # Repetição quase literal de uma fala anterior do mesmo agente.
    for anterior in ctx.mensagens_anteriores:
        if jaccard(mensagem, anterior) >= limiar_similaridade:
            return Deteccao(
                tipo="laco_infinito",
                agente=ctx.agente,
                evidencia="repetiu quase literalmente uma mensagem anterior",
                correcao=(
                    "Você repetiu o que já disse. Se não há como avançar com "
                    "suas ferramentas, diga isso explicitamente e pare."
                ),
            )
    return None


# --------------------------------------------------------------------------- #
# Avaliação conjunta
# --------------------------------------------------------------------------- #

DETECTORES = [
    detectar_inversao_papeis,
    detectar_repeticao_instrucao,
    detectar_resposta_evasiva,
    detectar_laco,
]


def avaliar(mensagem: str, ctx: Contexto) -> Deteccao | None:
    """
    Aplica os quatro detectores na ordem e devolve a primeira detecção.

    A ordem importa: inversão de papéis é a falha mais específica e a que mais
    muda o rumo da conversa, então é testada primeiro. A evasiva é a mais
    genérica e vem depois, senão engoliria as outras três.
    """
    for detector in DETECTORES:
        deteccao = detector(mensagem, ctx)
        if deteccao is not None:
            return deteccao
    return None


TIPOS = ["inversao_papeis", "repeticao_instrucao", "resposta_evasiva", "laco_infinito"]
