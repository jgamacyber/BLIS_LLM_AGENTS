"""
O loop ReAct: raciocinar e agir de forma entrelaçada.

Implementação de Yao et al. (2023), *ReAct: Synergizing Reasoning and Acting in
Language Models* (ICLR 2023).

A ideia formal do artigo, em uma linha: o espaço de ações é **aumentado** com a
linguagem.

    Â = A ∪ L

onde A são as ações que afetam o ambiente (as ferramentas) e L é o espaço da
linguagem. Uma ação â ∈ L é um **pensamento**: ela não afeta o ambiente e não
gera observação, apenas atualiza o contexto:

    c_{t+1} = (c_t, â_t)

Por que isso importa: o chain-of-thought puro raciocina sem se aterrissar em
nada externo, o que produz alucinação e propagação de erro. Agir sem raciocinar
produz sequências de ações sem plano. O ReAct alterna os dois, e o artigo mostra
ganhos absolutos de 34% no ALFWorld e 10% no WebShop sobre imitação e RL, com
apenas um ou dois exemplos no prompt.

O artigo lista os tipos de pensamento que aparecem nas trajetórias boas:
decompor a meta, injetar senso comum, extrair partes importantes da observação,
rastrear o progresso e tratar exceções ajustando o plano. O prompt abaixo pede
exatamente isso.

**Término:** a taxonomia vem do AgentBench (Liu et al., 2024), que classifica
por que um agente parou em cinco tipos. Distinguir "não seguiu o formato" de
"escolheu uma ação inexistente" de "estourou o limite" é o que transforma uma
falha em diagnóstico.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

from config import SETTINGS, Settings, get_client
from ferramentas import RegistroFerramentas, registro_padrao
from memoria import MemoriaHierarquica


class Termino(str, Enum):
    """
    Razões de término, na taxonomia do AgentBench (Liu et al., 2024).

    O artigo observa que IF e IA decorrem de baixa capacidade de seguir
    instruções, enquanto TLE indica fraqueza em interação multi-turno.
    """

    COMPLETO = "complete"            # terminou normalmente com uma resposta
    LIMITE_CONTEXTO = "CLE"          # Context Limit Exceeded
    FORMATO_INVALIDO = "IF"          # Invalid Format
    ACAO_INVALIDA = "IA"             # Invalid Action
    LIMITE_TAREFA = "TLE"            # Task Limit Exceeded
    ERRO = "erro"                    # falha de infraestrutura

    def descricao(self) -> str:
        return {
            "complete": "concluiu a tarefa",
            "CLE": "estourou o limite de contexto",
            "IF": "não seguiu o formato exigido",
            "IA": "escolheu uma ação inexistente",
            "TLE": "estourou o limite de passos ou entrou em laço",
            "erro": "falha de infraestrutura",
        }[self.value]


# --------------------------------------------------------------------------- #
# Estruturas do rastro
# --------------------------------------------------------------------------- #


@dataclass
class Passo:
    """Um ciclo Thought → Action → Observation."""

    numero: int
    pensamento: str = ""
    acao: str = ""
    entrada_acao: str = ""
    observacao: str = ""
    resposta_final: str = ""
    bruto: str = ""
    erro_formato: str = ""
    acao_valida: bool = True
    latencia: float = 0.0
    tokens_prompt: int = 0
    tokens_resposta: int = 0

    @property
    def tem_acao(self) -> bool:
        return bool(self.acao)

    @property
    def concluiu(self) -> bool:
        return bool(self.resposta_final)

    def imprimir(self, largura: int = 100) -> None:
        print(f"\n  ── passo {self.numero} ──")
        if self.pensamento:
            print(f"  Thought: {self.pensamento[:largura * 2]}")
        if self.acao:
            marca = "" if self.acao_valida else "  [ação inválida]"
            print(f"  Action:  {self.acao}[{self.entrada_acao[:70]}]{marca}")
        if self.observacao:
            previa = self.observacao.replace("\n", " ")[:largura * 2]
            print(f"  Obs:     {previa}")
        if self.resposta_final:
            print(f"  Finish:  {self.resposta_final[:largura * 2]}")
        if self.erro_formato:
            print(f"  [formato] {self.erro_formato}")


@dataclass
class Execucao:
    """O resultado completo de uma execução do agente."""

    tarefa: str
    passos: list[Passo] = field(default_factory=list)
    resposta: str = ""
    termino: Termino = Termino.ERRO
    detalhe_termino: str = ""
    duracao: float = 0.0
    tokens_totais: int = 0
    estatisticas_memoria: dict = field(default_factory=dict)

    @property
    def n_passos(self) -> int:
        return len(self.passos)

    @property
    def sucesso(self) -> bool:
        return self.termino == Termino.COMPLETO and bool(self.resposta)

    @property
    def ferramentas_usadas(self) -> list[str]:
        vistas: list[str] = []
        for p in self.passos:
            if p.acao and p.acao_valida and p.acao not in vistas:
                vistas.append(p.acao)
        return vistas

    def imprimir(self, detalhado: bool = True) -> None:
        print(f"\n{'=' * 74}")
        print(f"  Tarefa: {self.tarefa}")
        print("=" * 74)

        if detalhado:
            for passo in self.passos:
                passo.imprimir()

        print(f"\n  {'─' * 70}")
        marca = "OK" if self.sucesso else "FALHOU"
        print(f"  [{marca}] {self.termino.value} — {self.termino.descricao()}")
        if self.detalhe_termino:
            print(f"  {self.detalhe_termino}")
        if self.resposta:
            print(f"\n  Resposta: {self.resposta}")
        print(f"\n  {self.n_passos} passos | {self.duracao:.2f}s | "
              f"{self.tokens_totais} tokens | "
              f"ferramentas: {', '.join(self.ferramentas_usadas) or 'nenhuma'}")

    def como_dict(self) -> dict:
        return {
            "tarefa": self.tarefa,
            "resposta": self.resposta,
            "termino": self.termino.value,
            "detalhe_termino": self.detalhe_termino,
            "sucesso": self.sucesso,
            "n_passos": self.n_passos,
            "duracao": round(self.duracao, 3),
            "tokens_totais": self.tokens_totais,
            "ferramentas_usadas": self.ferramentas_usadas,
            "passos": [asdict(p) for p in self.passos],
            "memoria": self.estatisticas_memoria,
        }


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

PROMPT_SISTEMA = """Você é um agente que resolve tarefas alternando raciocínio e ação.

FERRAMENTAS DISPONÍVEIS:
{ferramentas}

FORMATO OBRIGATÓRIO. A cada passo, produza EXATAMENTE um destes dois blocos:

Para usar uma ferramenta:
Thought: <seu raciocínio sobre o que fazer agora e por quê>
Action: <nome exato da ferramenta>
Action Input: <a entrada para a ferramenta>

Para dar a resposta final:
Thought: <por que você já tem o necessário para responder>
Final Answer: <a resposta>

REGRAS:
1. Produza um único bloco por vez e PARE. O sistema vai executar a ação e te
   devolver uma linha começando com "Observation:".
2. Nunca invente uma observação. Espere a do sistema.
3. Use exatamente os nomes de ferramenta da lista. Não invente ferramentas.
4. Se uma ferramenta devolver ERRO, leia a mensagem e corrija a próxima ação em
   vez de repetir a mesma chamada.
5. Se a informação necessária não estiver disponível por nenhuma ferramenta,
   diga isso na Final Answer em vez de inventar.
6. Não repita uma ação que já deu o mesmo resultado.

No Thought, quando fizer sentido: decomponha a tarefa em partes, registre o que
já descobriu, avalie se a observação responde ao que você precisava e ajuste o
plano quando algo falhar."""

PROMPT_TAREFA = """TAREFA: {tarefa}

Comece pelo primeiro Thought."""


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

_RE_THOUGHT = re.compile(r"Thought\s*:\s*(.+?)(?=\n\s*(?:Action|Final Answer)\s*:|\Z)",
                         re.IGNORECASE | re.DOTALL)
_RE_ACTION = re.compile(r"Action\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE)
_RE_ACTION_INPUT = re.compile(
    r"Action\s*Input\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL)
_RE_FINAL = re.compile(r"Final\s*Answer\s*:\s*(.+?)(?=\n\s*(?:Thought|Action)\s*:|\Z)",
                       re.IGNORECASE | re.DOTALL)


def analisar_saida(texto: str) -> Passo:
    """
    Extrai Thought, Action, Action Input e Final Answer da saída do modelo.

    É tolerante de propósito: aceita `Action: calculadora[2+2]` (o formato
    compacto do artigo) além do formato em três linhas. Um parser rígido demais
    transformaria variações inofensivas em Invalid Format, e o objetivo é medir
    a capacidade do agente, não a rigidez do parser.
    """
    passo = Passo(numero=0, bruto=texto or "")
    texto = (texto or "").strip()

    if not texto:
        passo.erro_formato = "saída vazia"
        return passo

    # O modelo às vezes alucina a observação; corta tudo a partir dali.
    corte = re.search(r"\n\s*Observation\s*:", texto, re.IGNORECASE)
    if corte:
        texto = texto[: corte.start()]

    match = _RE_THOUGHT.search(texto)
    if match:
        passo.pensamento = match.group(1).strip()

    match = _RE_FINAL.search(texto)
    if match:
        passo.resposta_final = match.group(1).strip()
        return passo

    match = _RE_ACTION.search(texto)
    if match:
        acao = match.group(1).strip().strip("`\"'")

        # Formato compacto: nome[entrada]
        compacto = re.match(r"^([\w_]+)\s*\[(.*)\]$", acao, re.DOTALL)
        if compacto:
            passo.acao = compacto.group(1).strip()
            passo.entrada_acao = compacto.group(2).strip()
            return passo

        passo.acao = acao.split("(")[0].strip()

        entrada = _RE_ACTION_INPUT.search(texto)
        if entrada:
            passo.entrada_acao = entrada.group(1).strip().strip("`\"'")
        return passo

    passo.erro_formato = (
        "não encontrei 'Action:' nem 'Final Answer:' na saída"
    )
    return passo


# --------------------------------------------------------------------------- #
# O agente
# --------------------------------------------------------------------------- #


# Ações internas de memória. No vocabulário do CoALA são ações de recuperação
# e de aprendizado: não tocam o ambiente externo, apenas a memória do agente.
FUNCOES_MEMORIA = frozenset({
    "working_context_append",
    "working_context_replace",
    "archival_insert",
    "archival_search",
    "recall_search",
})


class AgenteReAct:
    """
    Agente que alterna pensamento e ação até resolver a tarefa.

    Ciclo de decisão, no vocabulário do CoALA: observação → planejamento
    (aqui, o Thought) → execução (a ação de grounding) → nova observação.
    """

    def __init__(
        self,
        ferramentas: RegistroFerramentas | None = None,
        settings: Settings | None = None,
        memoria: MemoriaHierarquica | None = None,
        verboso: bool = True,
        usar_memoria: bool = False,
    ) -> None:
        self.settings = settings or SETTINGS
        self.ferramentas = ferramentas or registro_padrao()
        self.verboso = verboso
        # Com isso ligado, o catálogo do prompt inclui as funções de memória e o
        # agente decide sozinho o que salvar e o que buscar, como no MemGPT.
        self.usar_memoria = usar_memoria

        catalogo = (
            catalogo_com_memoria(self.ferramentas)
            if usar_memoria
            else self.ferramentas.catalogo()
        )

        self.memoria = memoria or MemoriaHierarquica(
            janela_contexto=self.settings.janela_contexto,
            limiar_pressao=self.settings.limiar_pressao,
            fracao_flush=self.settings.fracao_flush,
            tamanho_working_context=self.settings.tamanho_working_context,
            pagina=self.settings.pagina_memoria,
            instrucoes_sistema=PROMPT_SISTEMA.format(ferramentas=catalogo),
        )

    def acao_conhecida(self, nome: str) -> bool:
        """A ação existe no catálogo que foi apresentado ao agente?"""
        if nome in self.ferramentas:
            return True
        return self.usar_memoria and nome in FUNCOES_MEMORIA

    # ------------------------------ execução ------------------------------- #

    def executar(self, tarefa: str, max_passos: int | None = None) -> Execucao:
        """Roda o loop até a resposta final ou até uma condição de término."""
        max_passos = max_passos or self.settings.max_passos
        inicio = time.perf_counter()

        execucao = Execucao(tarefa=tarefa)
        self.memoria.adicionar_mensagem("usuario", PROMPT_TAREFA.format(tarefa=tarefa))

        erros_formato = 0
        acoes_invalidas = 0
        acoes_recentes: list[str] = []

        for numero in range(1, max_passos + 1):
            # Limite de contexto: só acontece se a memória não conseguir liberar.
            if self.memoria.tokens_usados > self.memoria.janela_contexto * 1.5:
                execucao.termino = Termino.LIMITE_CONTEXTO
                execucao.detalhe_termino = (
                    f"contexto em {self.memoria.tokens_usados} tokens, acima do "
                    f"limite mesmo após o flush"
                )
                break

            try:
                bruto, tokens_p, tokens_r = self._chamar_modelo()
            except Exception as erro:  # noqa: BLE001
                execucao.termino = Termino.ERRO
                execucao.detalhe_termino = str(erro)
                break

            passo = analisar_saida(bruto)
            passo.numero = numero
            passo.tokens_prompt, passo.tokens_resposta = tokens_p, tokens_r
            execucao.tokens_totais += tokens_p + tokens_r

            # --- Invalid Format ---
            if passo.erro_formato:
                erros_formato += 1
                execucao.passos.append(passo)
                if self.verboso:
                    passo.imprimir()

                if erros_formato >= self.settings.max_erros_formato:
                    execucao.termino = Termino.FORMATO_INVALIDO
                    execucao.detalhe_termino = (
                        f"{erros_formato} saídas seguidas fora do formato exigido"
                    )
                    break

                self.memoria.adicionar_mensagem(
                    "sistema",
                    "Sua saída não seguiu o formato. Produza exatamente:\n"
                    "Thought: ...\nAction: <ferramenta>\nAction Input: ...\n"
                    "ou\nThought: ...\nFinal Answer: ...",
                    passo=numero,
                )
                continue

            erros_formato = 0

            # --- resposta final ---
            if passo.concluiu:
                execucao.passos.append(passo)
                execucao.resposta = passo.resposta_final
                execucao.termino = Termino.COMPLETO
                if self.verboso:
                    passo.imprimir()
                break

            # --- ação ---
            self.memoria.adicionar_mensagem(
                "agente",
                f"Thought: {passo.pensamento}\nAction: {passo.acao}\n"
                f"Action Input: {passo.entrada_acao}",
                passo=numero,
            )

            passo.acao_valida = self.acao_conhecida(passo.acao)

            resultado = self._executar_acao(passo, numero)
            passo.observacao = resultado
            execucao.passos.append(passo)

            if self.verboso:
                passo.imprimir()

            self.memoria.adicionar_mensagem(
                "ferramenta", f"Observation: {resultado}", passo=numero
            )

            # --- Invalid Action ---
            # Uma ação inexistente não encerra a execução na hora: a observação
            # de erro lista o catálogo e o agente costuma se corrigir no passo
            # seguinte. Insistir no erro é que vira IA, no sentido do AgentBench.
            if not passo.acao_valida:
                acoes_invalidas += 1
                if acoes_invalidas >= self.settings.max_erros_formato:
                    execucao.termino = Termino.ACAO_INVALIDA
                    execucao.detalhe_termino = (
                        f"{acoes_invalidas} ações seguidas fora do catálogo "
                        f"(a última foi '{passo.acao}')"
                    )
                    break
                continue

            acoes_invalidas = 0

            # --- detecção de laço ---
            assinatura = f"{passo.acao}|{passo.entrada_acao}"
            acoes_recentes.append(assinatura)
            repeticoes = acoes_recentes.count(assinatura)

            if repeticoes >= self.settings.max_repeticoes:
                execucao.termino = Termino.LIMITE_TAREFA
                execucao.detalhe_termino = (
                    f"repetiu a ação '{passo.acao}' com a mesma entrada "
                    f"{repeticoes} vezes"
                )
                break

        else:
            # O for terminou sem break: estourou os passos.
            execucao.termino = Termino.LIMITE_TAREFA
            execucao.detalhe_termino = (
                f"não concluiu em {max_passos} passos"
            )

        execucao.duracao = time.perf_counter() - inicio
        execucao.estatisticas_memoria = self.memoria.estatisticas()
        return execucao

    def _executar_acao(self, passo: Passo, numero: int) -> str:
        """Executa a ferramenta ou uma das funções internas de memória."""
        nome = passo.acao
        entrada = passo.entrada_acao

        # Ações internas de memória (as ações de recuperação e aprendizado do
        # CoALA). Só são despachadas quando o catálogo as ofereceu; fora disso
        # caem no registro e viram observação de ação inexistente, que é o
        # comportamento honesto: o agente não pode usar o que não lhe foi dado.
        if self.usar_memoria and nome in FUNCOES_MEMORIA:
            if nome == "working_context_append":
                return self.memoria.working_context_append(entrada, numero)
            if nome == "working_context_replace":
                if "=>" not in entrada:
                    return (
                        "erro: use 'trecho antigo => trecho novo' na entrada de "
                        "working_context_replace"
                    )
                antigo, novo = entrada.split("=>", 1)
                return self.memoria.working_context_replace(
                    antigo.strip(), novo.strip(), numero
                )
            if nome == "archival_insert":
                return self.memoria.archival_insert(entrada, passo=numero)
            if nome == "archival_search":
                return self.memoria.archival_search(entrada, passo=numero)
            if nome == "recall_search":
                return self.memoria.recall_search(entrada, passo=numero)

        resultado = self.ferramentas.executar(nome, entrada)
        return resultado.como_observacao()

    def _chamar_modelo(self) -> tuple[str, int, int]:
        """Uma chamada ao LLM com o main context montado."""
        client = get_client(self.settings)
        contexto = self.memoria.montar_contexto()

        resposta = client.chat.completions.create(
            model=self.settings.model,
            messages=[{"role": "user", "content": contexto}],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stop=["\nObservation:", "Observation:"],
        )

        texto = (resposta.choices[0].message.content or "").strip()
        uso = getattr(resposta, "usage", None)
        return (
            texto,
            getattr(uso, "prompt_tokens", 0) or 0,
            getattr(uso, "completion_tokens", 0) or 0,
        )

    def reiniciar(self) -> None:
        """Zera a memória, preservando as instruções do sistema."""
        instrucoes = self.memoria.instrucoes_sistema
        self.memoria = MemoriaHierarquica(
            janela_contexto=self.settings.janela_contexto,
            limiar_pressao=self.settings.limiar_pressao,
            fracao_flush=self.settings.fracao_flush,
            tamanho_working_context=self.settings.tamanho_working_context,
            pagina=self.settings.pagina_memoria,
            instrucoes_sistema=instrucoes,
        )


def catalogo_com_memoria(ferramentas: RegistroFerramentas) -> str:
    """
    Catálogo incluindo as funções de memória.

    Usado quando o agente roda no modo com memória explícita, em que ele mesmo
    decide o que salvar e o que recuperar, como no MemGPT.
    """
    return ferramentas.catalogo() + """
- working_context_append: salva um fato curto na memória de trabalho, que fica
    sempre visível. Use para informação que você vai precisar até o fim.
    exemplo: working_context_append | o pedido 4471 tem total de R$ 347,90
- working_context_replace: corrige um fato já salvo na memória de trabalho.
    A entrada é 'trecho antigo => trecho novo'.
    exemplo: working_context_replace | total R$ 295,50 => total R$ 318,00
- archival_insert: guarda um fato no arquivo de longo prazo, fora do contexto.
    exemplo: archival_insert | o corpus tem 8 documentos sobre agentes
- archival_search: busca fatos que você guardou no arquivo.
    exemplo: archival_search | total do pedido
- recall_search: busca no histórico completo da conversa, inclusive no que já
    saiu do contexto por falta de espaço.
    exemplo: recall_search | qual era o valor do frete"""
