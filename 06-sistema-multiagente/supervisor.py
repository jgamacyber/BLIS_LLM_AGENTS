"""
O supervisor: escolhe quem fala, entrega o handoff e fecha a resposta.

É o gerente de conversa em grupo do AutoGen, com os três passos do artigo:
escolher dinamicamente o próximo a falar, coletar a resposta e transmitir ao
grupo. A escolha usa um prompt em estilo de interpretação de papéis, com o
catálogo dos especialistas e o quadro compartilhado à vista.

Duas diferenças em relação ao artigo, ambas deliberadas.

A primeira é que a transmissão não copia a fala de todos para todos. O que é
transmitido é o **quadro compartilhado**: fatos com autor e fonte. A fala fica
no histórico para auditoria. Copiar tudo para todos é o que faz o contexto de
um sistema multiagente explodir na terceira rodada.

A segunda é que cada resposta de especialista passa pelas salvaguardas do CAMEL
antes de ser aceita. Detectada uma das quatro falhas, a correção volta para o
mesmo agente, e isso conta contra o limite de respostas automáticas
consecutivas do AutoGen. Duas ideias de artigos diferentes se encaixando: o
CAMEL diz o que detectar, o AutoGen diz quantas vezes insistir antes de parar.

**Término** segue a taxonomia do AgentBench, com um acréscimo. O artigo
classifica por que o agente parou; aqui vale o mesmo, mais uma razão que só
existe em sistema multiagente: a conversa encerrada por salvaguarda.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import salvaguardas
from agentes import Agente, Resposta
from config import SETTINGS, Settings, get_client
from estado import EstadoCompartilhado


class Termino(str, Enum):
    """Razões de término, na taxonomia do AgentBench mais a salvaguarda."""

    COMPLETO = "complete"            # entregou a resposta final
    FORMATO_INVALIDO = "IF"          # supervisor não seguiu o formato
    ESPECIALISTA_INVALIDO = "IA"     # insistiu em chamar quem não existe
    LIMITE_RODADAS = "TLE"           # estourou as rodadas
    SALVAGUARDA = "SG"               # modos de falha do CAMEL, repetidos
    ERRO = "erro"                    # falha de infraestrutura

    def descricao(self) -> str:
        return {
            "complete": "entregou a resposta final",
            "IF": "o supervisor não seguiu o formato de decisão",
            "IA": "insistiu em acionar um especialista inexistente",
            "TLE": "estourou o limite de rodadas",
            "SG": "encerrada pelas salvaguardas do CAMEL",
            "erro": "falha de infraestrutura",
        }[self.value]


# --------------------------------------------------------------------------- #
# Prompt do supervisor
# --------------------------------------------------------------------------- #

PROMPT_SUPERVISOR = """Você é o supervisor de um time de agentes especialistas.
Você NÃO tem ferramentas: você não lê arquivos, não calcula e não consulta
documentos. Tudo o que você pode fazer é decidir qual especialista trabalha
agora e o que exatamente ele deve fazer, ou fechar a resposta final com o que
o time já apurou.

SEU TIME:
{especialistas}

SITUAÇÃO ATUAL:
{situacao}

FORMATO OBRIGATÓRIO. Produza EXATAMENTE um destes dois blocos:

Para acionar um especialista:
ANALISE: <o que ainda falta e por que este especialista é o certo>
HANDOFF: <nome exato do especialista> | <instrução específica e verificável>

Para encerrar:
ANALISE: <por que o time já tem o necessário>
RESPOSTA FINAL: <a resposta à tarefa, citando as fontes dos fatos usados>

REGRAS:
1. Um bloco por vez. Nunca acione dois especialistas na mesma mensagem.
2. A instrução do handoff precisa ser executável com as ferramentas daquele
   especialista. Não peça cálculo a quem só consulta documentos.
3. Não repita um handoff que já foi cumprido. Olhe os fatos já apurados.
4. Só dê a RESPOSTA FINAL com base nos fatos do quadro. Se faltar algo que
   nenhum especialista consegue obter, diga isso na resposta em vez de
   inventar.
5. Cite a fonte de cada número que aparecer na resposta final."""


_RE_ANALISE = re.compile(r"ANALISE\s*:\s*(.+?)(?=\n\s*(?:HANDOFF|RESPOSTA FINAL)\s*:|\Z)",
                         re.IGNORECASE | re.DOTALL)
_RE_HANDOFF = re.compile(r"HANDOFF\s*:\s*(.+?)(?=\n\s*(?:ANALISE|RESPOSTA FINAL)\s*:|\Z)",
                         re.IGNORECASE | re.DOTALL)
_RE_FINAL = re.compile(r"RESPOSTA\s+FINAL\s*:\s*(.+?)(?=\n\s*(?:ANALISE|HANDOFF)\s*:|\Z)",
                       re.IGNORECASE | re.DOTALL)


@dataclass
class Decisao:
    """O que o supervisor decidiu nesta rodada."""

    analise: str = ""
    especialista: str = ""
    instrucao: str = ""
    resposta_final: str = ""
    erro_formato: str = ""

    @property
    def encerra(self) -> bool:
        return bool(self.resposta_final)


def analisar_decisao(texto: str) -> Decisao:
    """Parser da saída do supervisor, tolerante às variações do modelo."""
    bruto = (texto or "").strip()
    if not bruto:
        return Decisao(erro_formato="saída vazia")

    decisao = Decisao()

    m = _RE_ANALISE.search(bruto)
    if m:
        decisao.analise = m.group(1).strip()

    m = _RE_FINAL.search(bruto)
    if m:
        decisao.resposta_final = m.group(1).strip()
        return decisao

    m = _RE_HANDOFF.search(bruto)
    if m:
        corpo = m.group(1).strip()
        if "|" in corpo:
            nome, instrucao = corpo.split("|", 1)
            decisao.especialista = nome.strip().strip("`*")
            decisao.instrucao = instrucao.strip()
        else:
            # Aceita "HANDOFF: nome" seguido da instrução na linha de baixo.
            linhas = [l.strip() for l in corpo.splitlines() if l.strip()]
            decisao.especialista = linhas[0].strip("`*") if linhas else ""
            decisao.instrucao = " ".join(linhas[1:])
        if not decisao.especialista:
            decisao.erro_formato = "handoff sem nome de especialista"
        elif not decisao.instrucao:
            decisao.erro_formato = "handoff sem instrução"
        return decisao

    decisao.erro_formato = "nenhum bloco HANDOFF ou RESPOSTA FINAL reconhecido"
    return decisao


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #


@dataclass
class Execucao:
    tarefa: str
    resposta: str = ""
    termino: Termino = Termino.ERRO
    detalhe_termino: str = ""
    rodadas: int = 0
    duracao: float = 0.0
    tokens_totais: int = 0
    estado: EstadoCompartilhado | None = None
    deteccoes: list[salvaguardas.Deteccao] = field(default_factory=list)

    @property
    def sucesso(self) -> bool:
        return self.termino == Termino.COMPLETO and bool(self.resposta)

    @property
    def especialistas_acionados(self) -> list[str]:
        return self.estado.especialistas_acionados() if self.estado else []

    def imprimir(self) -> None:
        print(f"\n{'=' * 74}")
        marca = "OK" if self.sucesso else "FALHOU"
        print(f"  [{marca}] {self.termino.value} — {self.termino.descricao()}")
        if self.detalhe_termino:
            print(f"  {self.detalhe_termino}")
        if self.resposta:
            print(f"\n  RESPOSTA FINAL:\n  {self.resposta}")
        print(f"\n  {self.rodadas} rodadas | {self.duracao:.2f}s | "
              f"{self.tokens_totais} tokens")
        if self.deteccoes:
            print(f"  salvaguardas disparadas: "
                  f"{', '.join(d.tipo for d in self.deteccoes)}")
        if self.estado:
            self.estado.imprimir()

    def como_dict(self) -> dict:
        return {
            "tarefa": self.tarefa,
            "resposta": self.resposta,
            "termino": self.termino.value,
            "detalhe_termino": self.detalhe_termino,
            "sucesso": self.sucesso,
            "rodadas": self.rodadas,
            "duracao": round(self.duracao, 3),
            "tokens_totais": self.tokens_totais,
            "especialistas_acionados": self.especialistas_acionados,
            "deteccoes": [
                {"tipo": d.tipo, "agente": d.agente, "evidencia": d.evidencia}
                for d in self.deteccoes
            ],
            "estado": self.estado.como_dict() if self.estado else {},
        }

    def salvar(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(self.como_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return caminho


class Supervisor:
    """Gerente da conversa em grupo: escolhe, coleta e transmite."""

    def __init__(
        self,
        especialistas: list[Agente],
        estado: EstadoCompartilhado,
        settings: Settings | None = None,
        verboso: bool = True,
        usar_salvaguardas: bool = True,
        nome: str = "supervisor",
    ) -> None:
        self.nome = nome
        self.especialistas = {a.nome: a for a in especialistas}
        self.estado = estado
        self.settings = settings or SETTINGS
        self.verboso = verboso
        self.usar_salvaguardas = usar_salvaguardas
        self.tokens = 0

    # ------------------------------ o laço --------------------------------- #

    def executar(self, tarefa: str, max_rodadas: int | None = None) -> Execucao:
        max_rodadas = max_rodadas or self.settings.max_rodadas
        inicio = time.perf_counter()

        self.estado.tarefa = tarefa
        execucao = Execucao(tarefa=tarefa, estado=self.estado)

        erros_formato = 0
        especialistas_invalidos = 0
        ultimo_especialista = ""
        repeticoes = 0

        for rodada in range(1, max_rodadas + 1):
            self.estado.rodada = rodada
            execucao.rodadas = rodada

            if self.verboso:
                print(f"\n  ── rodada {rodada} ──")

            try:
                bruto, tokens = self._decidir()
            except Exception as erro:  # noqa: BLE001
                execucao.termino = Termino.ERRO
                execucao.detalhe_termino = str(erro)
                break

            execucao.tokens_totais += tokens
            decisao = analisar_decisao(bruto)

            # --- Invalid Format ---
            if decisao.erro_formato:
                erros_formato += 1
                if self.verboso:
                    print(f"    [formato] {decisao.erro_formato}")
                if erros_formato >= 3:
                    execucao.termino = Termino.FORMATO_INVALIDO
                    execucao.detalhe_termino = (
                        f"{erros_formato} decisões seguidas fora do formato"
                    )
                    break
                self.estado.registrar_mensagem(
                    "sistema", self.nome,
                    "Sua saída não seguiu o formato. Produza 'ANALISE:' seguido "
                    "de 'HANDOFF: <especialista> | <instrução>' ou de "
                    "'RESPOSTA FINAL: <resposta>'.",
                    tipo="salvaguarda",
                )
                continue

            erros_formato = 0

            # --- encerramento ---
            if decisao.encerra:
                execucao.resposta = decisao.resposta_final
                execucao.termino = Termino.COMPLETO
                self.estado.registrar_mensagem(self.nome, "todos",
                                               decisao.resposta_final, tipo="final")
                self.estado.encerrar("resposta final entregue")
                if self.verboso:
                    print(f"    {self.nome}: RESPOSTA FINAL")
                break

            # --- Invalid Action: especialista inexistente ---
            agente = self.especialistas.get(decisao.especialista)
            if agente is None:
                especialistas_invalidos += 1
                disponiveis = ", ".join(self.especialistas)
                if self.verboso:
                    print(f"    [especialista inexistente] {decisao.especialista}")
                self.estado.registrar_mensagem(
                    "sistema", self.nome,
                    f"O especialista '{decisao.especialista}' não existe. "
                    f"Disponíveis: {disponiveis}.",
                    tipo="salvaguarda",
                )
                if especialistas_invalidos >= 3:
                    execucao.termino = Termino.ESPECIALISTA_INVALIDO
                    execucao.detalhe_termino = (
                        f"{especialistas_invalidos} handoffs seguidos para "
                        f"especialistas inexistentes"
                    )
                    break
                continue

            especialistas_invalidos = 0

            # --- repetição do mesmo especialista ---
            if decisao.especialista == ultimo_especialista:
                repeticoes += 1
                if repeticoes >= self.settings.max_repeticao_especialista:
                    execucao.termino = Termino.LIMITE_RODADAS
                    execucao.detalhe_termino = (
                        f"acionou '{decisao.especialista}' {repeticoes + 1} vezes "
                        f"seguidas sem fechar a resposta"
                    )
                    break
            else:
                repeticoes = 0
            ultimo_especialista = decisao.especialista

            # --- handoff ---
            self.estado.registrar_handoff(self.nome, agente.nome, decisao.instrucao)
            self.estado.registrar_mensagem(self.nome, agente.nome,
                                           decisao.instrucao, tipo="handoff")
            if self.verboso:
                print(f"    handoff -> {agente.nome}: {decisao.instrucao[:80]}")

            resposta = self._coletar(agente, decisao.instrucao, execucao)
            execucao.tokens_totais += resposta.tokens

            # --- transmissão ---
            self.estado.registrar_mensagem(agente.nome, self.nome,
                                           resposta.conteudo, tipo="devolucao")
            if self.verboso:
                ferramentas = ", ".join(resposta.ferramentas_usadas) or "nenhuma"
                print(f"    {agente.nome} [{resposta.origem}] "
                      f"({ferramentas}; {resposta.fatos_registrados} fatos): "
                      f"{resposta.conteudo[:110]}")

            if len(execucao.deteccoes) >= self.settings.max_salvaguardas:
                execucao.termino = Termino.SALVAGUARDA
                execucao.detalhe_termino = (
                    f"{len(execucao.deteccoes)} detecções de modo de falha do "
                    f"CAMEL: "
                    f"{', '.join(sorted({d.tipo for d in execucao.deteccoes}))}"
                )
                self.estado.encerrar("encerrada pelas salvaguardas")
                break

        else:
            execucao.termino = Termino.LIMITE_RODADAS
            execucao.detalhe_termino = f"não fechou em {max_rodadas} rodadas"

        execucao.duracao = time.perf_counter() - inicio
        if not self.estado.encerrado:
            self.estado.encerrar(execucao.termino.descricao())
        return execucao

    # ---------------------------- coleta e correção ------------------------ #

    def _coletar(self, agente: Agente, instrucao: str,
                 execucao: Execucao) -> Resposta:
        """
        Coleta a resposta do especialista, aplicando as salvaguardas.

        Detectada uma falha, a correção volta ao mesmo agente. O número de
        idas e voltas é limitado pelo teto de respostas automáticas consecutivas
        do AutoGen: insistir com um agente que não está cooperando é como os
        dois agentes do CAMEL ficarem se agradecendo, só que caro.
        """
        pedido = instrucao
        resposta = Resposta(agente=agente.nome)

        for tentativa in range(1, self.settings.max_auto_resposta + 1):
            agente.receber(pedido, self.nome)
            resposta = agente.gerar_resposta(pedido)

            if not self.usar_salvaguardas:
                return resposta

            contexto = salvaguardas.Contexto(
                agente=agente.nome,
                instrucao_recebida=instrucao,
                ferramentas_usadas=resposta.ferramentas_usadas,
                fatos_registrados=resposta.fatos_registrados,
                mensagens_anteriores=agente.enviadas[:-1],
            )
            deteccao = salvaguardas.avaliar(resposta.conteudo, contexto)

            agente.enviadas.append(resposta.conteudo)

            if deteccao is None:
                return resposta

            execucao.deteccoes.append(deteccao)
            self.estado.registrar_deteccao(deteccao.tipo, agente.nome,
                                           deteccao.evidencia)
            if self.verboso:
                print(f"    {deteccao.formatar()}")

            if tentativa >= self.settings.max_auto_resposta:
                break

            # A correção vira a nova instrução, com a original anexada.
            pedido = (
                f"{deteccao.correcao}\n\nINSTRUÇÃO ORIGINAL, que continua "
                f"valendo:\n{instrucao}"
            )

        return resposta

    # ------------------------------ decisão -------------------------------- #

    def _decidir(self) -> tuple[str, int]:
        client = get_client(self.settings)
        conteudo = PROMPT_SUPERVISOR.format(
            especialistas="\n".join(a.ficha() for a in self.especialistas.values()),
            situacao=self.estado.resumo(),
        )
        resposta = client.chat.completions.create(
            model=self.settings.model,
            messages=[{"role": "user", "content": conteudo}],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        texto = (resposta.choices[0].message.content or "").strip()
        uso = getattr(resposta, "usage", None)
        tokens = ((getattr(uso, "prompt_tokens", 0) or 0)
                  + (getattr(uso, "completion_tokens", 0) or 0))
        self.tokens += tokens
        return texto, tokens


# --------------------------------------------------------------------------- #
# Montagem do time
# --------------------------------------------------------------------------- #

PERFIS = [
    {
        "nome": "pesquisador",
        "papel": "pesquisador",
        "descricao": "consulta o corpus de artigos sobre agentes através do RAG "
                     "e responde o que os documentos dizem, sempre com a fonte. "
                     "Não lê arquivos operacionais e não faz contas.",
    },
    {
        "nome": "analista",
        "papel": "analista",
        "descricao": "lê os arquivos operacionais (pedidos, estoque, notas da "
                     "equipe), calcula e converte unidades. Não tem acesso ao "
                     "corpus de artigos.",
    },
    {
        "nome": "redator",
        "papel": "redator",
        "descricao": "consolida os fatos do quadro compartilhado em um texto "
                     "final. Não tem nenhuma ferramenta de coleta: só usa o que "
                     "os outros registraram.",
    },
]


def montar_time(estado: EstadoCompartilhado, settings: Settings | None = None,
                verboso: bool = False) -> list[Agente]:
    """Cria os três especialistas, cada um com o seu subconjunto de ferramentas."""
    from ferramentas import registro_para

    return [
        Agente(
            nome=perfil["nome"],
            papel=perfil["papel"],
            descricao=perfil["descricao"],
            ferramentas=registro_para(perfil["papel"], estado, perfil["nome"]),
            estado=estado,
            settings=settings,
            verboso=verboso,
        )
        for perfil in PERFIS
    ]


def montar_sistema(tarefa: str = "", settings: Settings | None = None,
                   verboso: bool = True,
                   usar_salvaguardas: bool = True) -> tuple[Supervisor, EstadoCompartilhado]:
    estado = EstadoCompartilhado(tarefa)
    time = montar_time(estado, settings, verboso=False)
    supervisor = Supervisor(time, estado, settings, verboso=verboso,
                            usar_salvaguardas=usar_salvaguardas)
    return supervisor, estado
