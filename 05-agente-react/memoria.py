"""
Memória hierárquica, no modelo do MemGPT.

Implementação de Packer et al. (2024), *MemGPT: Towards LLMs as Operating
Systems*, organizada com o vocabulário de memórias do CoALA (Sumers et al.,
2024).

A analogia do artigo é com memória virtual de sistemas operacionais: a janela de
contexto é a RAM, o armazenamento externo é o disco, e o próprio agente decide o
que paginar entre os dois.

    MAIN CONTEXT (tokens do prompt, análogo à RAM)
      ├── instruções do sistema     read-only
      ├── working context           bloco de tamanho fixo, escrita por funções
      └── fila FIFO                 histórico rolante; índice 0 guarda o resumo

    EXTERNAL CONTEXT (análogo ao disco)
      ├── recall storage            todas as mensagens já trocadas
      └── archival storage          fatos que o agente decidiu guardar

Os dois gatilhos que fazem isso funcionar:

1. **Pressão de memória** (70% da janela): o sistema avisa o agente, que ainda
   tem contexto sobrando para salvar o que importa antes da perda.
2. **Flush** (100%): despeja metade da fila e gera um **resumo recursivo**,
   usando o resumo anterior mais as mensagens despejadas. O conteúdo despejado
   continua acessível via `recall_search`.

No vocabulário do CoALA: o working context é memória semântica (fatos sobre o
mundo e sobre a tarefa), o recall storage é memória episódica (o que aconteceu),
e as funções de leitura e escrita são as ações internas de *retrieval* e
*learning*.

A contagem de tokens aqui é uma **estimativa** (~4 caracteres por token), não um
tokenizador real. É suficiente para exercitar a mecânica de paginação, que é o
ponto do módulo, e mantém o arquivo sem dependências.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path


def estimar_tokens(texto: str) -> int:
    """Estimativa grosseira: ~4 caracteres por token."""
    return max(1, len(texto or "") // 4)


def _sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokenizar(texto: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _sem_acentos((texto or "").lower())))


# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #


@dataclass
class Mensagem:
    """Uma entrada da fila FIFO ou do recall storage."""

    papel: str          # "usuario" | "agente" | "sistema" | "ferramenta"
    conteudo: str
    passo: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def tokens(self) -> int:
        return estimar_tokens(self.conteudo)

    def formatar(self) -> str:
        return f"[{self.papel}] {self.conteudo}"


@dataclass
class Fato:
    """Uma entrada do archival storage."""

    conteudo: str
    rotulo: str = ""
    passo: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EventoMemoria:
    """Registro de uma operação de memória, para auditoria."""

    tipo: str           # "pressao" | "flush" | "escrita" | "busca"
    detalhe: str
    passo: int = 0
    tokens_antes: int = 0
    tokens_depois: int = 0


# --------------------------------------------------------------------------- #
# Memória hierárquica
# --------------------------------------------------------------------------- #


class MemoriaHierarquica:
    """
    Gerencia main context e external context, com paginação automática.

    O agente interage com ela por funções (as ações internas do CoALA), e o
    gerenciador de fila cuida do overflow sem intervenção.
    """

    def __init__(
        self,
        janela_contexto: int = 4000,
        limiar_pressao: float = 0.7,
        fracao_flush: float = 0.5,
        tamanho_working_context: int = 1200,
        pagina: int = 3,
        instrucoes_sistema: str = "",
    ) -> None:
        self.janela_contexto = janela_contexto
        self.limiar_pressao = limiar_pressao
        self.fracao_flush = fracao_flush
        self.tamanho_working_context = tamanho_working_context
        self.pagina = pagina

        # --- main context ---
        self.instrucoes_sistema = instrucoes_sistema   # read-only
        self.working_context: list[str] = []           # escrita por funções
        self.fila: list[Mensagem] = []                 # histórico rolante
        self.resumo_recursivo = ""                     # índice 0 da fila

        # --- external context ---
        self.recall: list[Mensagem] = []               # tudo que já passou
        self.archival: list[Fato] = []                 # fatos guardados

        # --- auditoria ---
        self.eventos: list[EventoMemoria] = []
        self.alertas_pendentes: list[str] = []
        # O aviso de pressão é disparado na *subida* do limiar, não a cada
        # mensagem acima dele: repetir o alerta gasta justamente o contexto que
        # ele está tentando proteger.
        self._avisou_pressao = False

    # ---------------------- contabilidade de contexto ---------------------- #

    @property
    def tokens_instrucoes(self) -> int:
        return estimar_tokens(self.instrucoes_sistema)

    @property
    def tokens_working_context(self) -> int:
        return estimar_tokens("\n".join(self.working_context))

    @property
    def tokens_resumo(self) -> int:
        return estimar_tokens(self.resumo_recursivo)

    @property
    def tokens_fila(self) -> int:
        return sum(m.tokens for m in self.fila)

    @property
    def tokens_usados(self) -> int:
        return (self.tokens_instrucoes + self.tokens_working_context
                + self.tokens_resumo + self.tokens_fila)

    @property
    def ocupacao(self) -> float:
        return self.tokens_usados / self.janela_contexto if self.janela_contexto else 0.0

    @property
    def sob_pressao(self) -> bool:
        return self.ocupacao >= self.limiar_pressao

    @property
    def estourou(self) -> bool:
        return self.ocupacao >= 1.0

    # ------------------------------ escrita -------------------------------- #

    def adicionar_mensagem(self, papel: str, conteudo: str, passo: int = 0) -> None:
        """
        Acrescenta uma mensagem à fila e ao recall storage.

        Depois de cada escrita, verifica pressão e overflow, exatamente como o
        queue manager do artigo.
        """
        mensagem = Mensagem(papel=papel, conteudo=conteudo, passo=passo)
        self.fila.append(mensagem)
        self.recall.append(mensagem)
        self._verificar_pressao(passo)

    def _verificar_pressao(self, passo: int) -> None:
        """Dispara aviso em 70% e flush em 100% da janela."""
        if self.estourou:
            self._flush(passo)
            # Depois do flush a ocupação cai; o próximo cruzamento do limiar
            # merece um aviso novo.
            self._avisou_pressao = self.sob_pressao
            return

        if not self.sob_pressao:
            self._avisou_pressao = False
            return

        # Aqui está sob pressão. Só avisa se ainda não avisou nesta subida.
        if self._avisou_pressao:
            return

        self._avisou_pressao = True
        self.alertas_pendentes.append(
            f"ALERTA DO SISTEMA: a memória está em {self.ocupacao:.0%} da "
            f"capacidade. Use working_context_append para salvar fatos "
            f"importantes ou archival_insert para guardar informação que "
            f"você vai precisar depois. O histórico mais antigo será "
            f"resumido e sairá do contexto."
        )
        self.eventos.append(EventoMemoria(
            tipo="pressao",
            detalhe=f"ocupação em {self.ocupacao:.0%}",
            passo=passo,
            tokens_antes=self.tokens_usados,
            tokens_depois=self.tokens_usados,
        ))

    def _flush(self, passo: int) -> None:
        """
        Despeja a fração mais antiga da fila e atualiza o resumo recursivo.

        O resumo novo é construído a partir do resumo anterior mais as
        mensagens despejadas — daí "recursivo". As mensagens saem do contexto
        mas continuam no recall storage, acessíveis por busca.
        """
        if not self.fila:
            return

        antes = self.tokens_usados
        alvo = int(self.janela_contexto * self.fracao_flush)

        despejadas: list[Mensagem] = []
        liberado = 0
        # Nunca despeja a última mensagem: é o turno atual.
        while self.fila and liberado < alvo and len(self.fila) > 1:
            mensagem = self.fila.pop(0)
            despejadas.append(mensagem)
            liberado += mensagem.tokens

        if despejadas:
            self.resumo_recursivo = self._resumir(despejadas)

        self.eventos.append(EventoMemoria(
            tipo="flush",
            detalhe=f"{len(despejadas)} mensagens despejadas, "
                    f"{liberado} tokens liberados",
            passo=passo,
            tokens_antes=antes,
            tokens_depois=self.tokens_usados,
        ))

    def _resumir(self, despejadas: list[Mensagem]) -> str:
        """
        Resumo recursivo extrativo, sem chamar o LLM.

        O artigo usa o próprio modelo para resumir. Aqui a versão é extrativa e
        determinística, porque assim o comportamento da memória é testável e
        reproduzível sem custo. A função `resumir_com_llm` abaixo faz a versão
        do artigo, para quem quiser comparar.
        """
        partes = []
        if self.resumo_recursivo:
            partes.append(self.resumo_recursivo)

        por_papel: dict[str, int] = {}
        trechos: list[str] = []
        for m in despejadas:
            por_papel[m.papel] = por_papel.get(m.papel, 0) + 1
            primeira_linha = m.conteudo.strip().splitlines()[0] if m.conteudo.strip() else ""
            if primeira_linha:
                trechos.append(f"{m.papel}: {primeira_linha[:110]}")

        contagem = ", ".join(f"{n} de {p}" for p, n in sorted(por_papel.items()))
        cabecalho = f"[resumo de {len(despejadas)} mensagens anteriores ({contagem})]"

        # Mantém as primeiras e as últimas: início e fim carregam mais sinal.
        if len(trechos) > 6:
            selecionados = trechos[:3] + ["..."] + trechos[-2:]
        else:
            selecionados = trechos

        partes.append(cabecalho + "\n" + "\n".join(selecionados))
        resumo = "\n\n".join(partes)

        # O resumo é recursivo, então cresce a cada flush: sem teto ele
        # reintroduz o problema que veio resolver. Teto de um quarto da janela,
        # convertido para caracteres pela mesma razão de 4 chars por token que
        # estimar_tokens usa.
        limite = int(self.janela_contexto * 0.25) * 4
        if len(resumo) <= limite:
            return resumo

        # Corta por linhas inteiras, das mais antigas para as mais novas: o
        # começo do resumo é o histórico mais distante. Cortar por caractere
        # deixaria a primeira linha pela metade.
        linhas = resumo.splitlines()
        mantidas: list[str] = []
        total = 0
        for linha in reversed(linhas):
            if total + len(linha) + 1 > limite:
                break
            mantidas.insert(0, linha)
            total += len(linha) + 1

        if not mantidas:
            mantidas = [linhas[-1][:limite]]

        return "[...trecho mais antigo do resumo descartado...]\n" + "\n".join(mantidas)

    # ------------------- funções de memória do agente ---------------------- #

    def working_context_append(self, conteudo: str, passo: int = 0) -> str:
        """
        Acrescenta um fato ao working context.

        Bloco de tamanho fixo: quando enche, a entrada mais antiga sai. É o
        comportamento do artigo, e obriga o agente a ser seletivo.
        """
        conteudo = (conteudo or "").strip()
        if not conteudo:
            return "ERRO: conteúdo vazio."
        if conteudo in self.working_context:
            return "Esse fato já está no working context."

        self.working_context.append(conteudo)

        removidos = 0
        while (estimar_tokens("\n".join(self.working_context)) * 4
               > self.tamanho_working_context and len(self.working_context) > 1):
            self.working_context.pop(0)
            removidos += 1

        self.eventos.append(EventoMemoria(
            tipo="escrita", detalhe=f"working_context += {conteudo[:60]}", passo=passo
        ))

        if removidos:
            return (f"Fato salvo no working context. "
                    f"{removidos} entrada(s) antiga(s) saíram por falta de espaço.")
        return "Fato salvo no working context."

    def working_context_replace(self, antigo: str, novo: str, passo: int = 0) -> str:
        """Substitui um fato do working context. Usado quando algo muda."""
        antigo, novo = (antigo or "").strip(), (novo or "").strip()
        for i, item in enumerate(self.working_context):
            if antigo.lower() in item.lower():
                self.working_context[i] = novo
                self.eventos.append(EventoMemoria(
                    tipo="escrita", detalhe=f"substituiu '{antigo[:40]}'", passo=passo
                ))
                return "Fato substituído no working context."
        return f"ERRO: não encontrei '{antigo[:40]}' no working context."

    def archival_insert(self, conteudo: str, rotulo: str = "", passo: int = 0) -> str:
        """Guarda um fato no archival storage (fora do contexto, sem limite)."""
        conteudo = (conteudo or "").strip()
        if not conteudo:
            return "ERRO: conteúdo vazio."
        self.archival.append(Fato(conteudo=conteudo, rotulo=rotulo, passo=passo))
        self.eventos.append(EventoMemoria(
            tipo="escrita", detalhe=f"archival += {conteudo[:60]}", passo=passo
        ))
        return f"Guardado no archival storage ({len(self.archival)} fatos no total)."

    # ------------------------------ leitura -------------------------------- #

    def _buscar(self, itens: list, consulta: str, texto_de) -> list:
        """Busca por sobreposição de termos, ordenada por relevância."""
        termos = _tokenizar(consulta)
        if not termos:
            return []

        pontuados = []
        for item in itens:
            palavras = _tokenizar(texto_de(item))
            comuns = len(termos & palavras)
            if comuns:
                pontuados.append((comuns / len(termos), item))

        pontuados.sort(key=lambda par: par[0], reverse=True)
        return [item for _, item in pontuados]

    def archival_search(self, consulta: str, pagina: int = 0, passo: int = 0) -> str:
        """Busca no archival storage, com paginação."""
        achados = self._buscar(self.archival, consulta, lambda f: f"{f.rotulo} {f.conteudo}")
        self.eventos.append(EventoMemoria(
            tipo="busca", detalhe=f"archival: '{consulta[:40]}' -> {len(achados)}",
            passo=passo,
        ))

        if not achados:
            return f"Nada no archival storage sobre '{consulta}'."
        return self._formatar_pagina(
            [f.conteudo for f in achados], consulta, pagina, "archival storage"
        )

    def recall_search(self, consulta: str, pagina: int = 0, passo: int = 0) -> str:
        """
        Busca no histórico completo de mensagens, inclusive as que saíram do
        contexto. É como o agente recupera o que foi despejado pelo flush.
        """
        achados = self._buscar(self.recall, consulta, lambda m: m.conteudo)
        self.eventos.append(EventoMemoria(
            tipo="busca", detalhe=f"recall: '{consulta[:40]}' -> {len(achados)}",
            passo=passo,
        ))

        if not achados:
            return f"Nada no histórico sobre '{consulta}'."
        return self._formatar_pagina(
            [f"(passo {m.passo}) {m.formatar()}" for m in achados],
            consulta, pagina, "histórico",
        )

    def _formatar_pagina(self, itens: list[str], consulta: str, pagina: int,
                         origem: str) -> str:
        """
        Pagina os resultados.

        O artigo é explícito quanto a isso: "os mecanismos de recuperação são
        projetados cientes das restrições de contexto e implementam paginação
        para evitar que as chamadas de recuperação estourem a janela".
        """
        total_paginas = max(1, (len(itens) + self.pagina - 1) // self.pagina)
        pagina = max(0, min(pagina, total_paginas - 1))
        inicio = pagina * self.pagina
        selecionados = itens[inicio : inicio + self.pagina]

        cabecalho = (
            f"Resultados de '{consulta}' no {origem} "
            f"(página {pagina + 1}/{total_paginas}, {len(itens)} no total):"
        )
        corpo = "\n".join(f"  {i}. {t[:300]}" for i, t in enumerate(selecionados, start=inicio + 1))

        rodape = ""
        if total_paginas > 1 and pagina + 1 < total_paginas:
            rodape = f"\n  [há mais {len(itens) - inicio - len(selecionados)} resultado(s); peça a página {pagina + 2}]"

        return f"{cabecalho}\n{corpo}{rodape}"

    # --------------------------- montagem do prompt ------------------------ #

    def montar_contexto(self) -> str:
        """
        Monta o main context como uma única string, na ordem do artigo:
        instruções, working context, resumo e fila.
        """
        partes = []

        if self.instrucoes_sistema:
            partes.append(self.instrucoes_sistema)

        if self.working_context:
            fatos = "\n".join(f"- {f}" for f in self.working_context)
            partes.append(f"### MEMÓRIA DE TRABALHO (fatos que você salvou)\n{fatos}")

        if self.resumo_recursivo:
            partes.append(f"### RESUMO DO HISTÓRICO ANTERIOR\n{self.resumo_recursivo}")

        if self.alertas_pendentes:
            partes.append("\n".join(self.alertas_pendentes))
            self.alertas_pendentes = []

        if self.fila:
            historico = "\n".join(m.formatar() for m in self.fila)
            partes.append(f"### HISTÓRICO RECENTE\n{historico}")

        return "\n\n".join(partes)

    # ------------------------------- infos --------------------------------- #

    def estatisticas(self) -> dict:
        return {
            "tokens_usados": self.tokens_usados,
            "janela": self.janela_contexto,
            "ocupacao": round(self.ocupacao, 3),
            "mensagens_na_fila": len(self.fila),
            "mensagens_no_recall": len(self.recall),
            "fatos_working_context": len(self.working_context),
            "fatos_archival": len(self.archival),
            "tem_resumo": bool(self.resumo_recursivo),
            "flushes": sum(1 for e in self.eventos if e.tipo == "flush"),
            "alertas_pressao": sum(1 for e in self.eventos if e.tipo == "pressao"),
        }

    def imprimir_estado(self) -> None:
        st = self.estatisticas()
        barra_cheia = int(st["ocupacao"] * 30)
        barra = "#" * min(30, barra_cheia) + "." * max(0, 30 - barra_cheia)

        print(f"\n  Memória  [{barra}] {st['ocupacao']:.0%} "
              f"({st['tokens_usados']}/{st['janela']} tokens)")
        print(f"    main context:     {st['mensagens_na_fila']} msgs na fila, "
              f"{st['fatos_working_context']} fatos no working context")
        print(f"    external context: {st['mensagens_no_recall']} msgs no recall, "
              f"{st['fatos_archival']} fatos no archival")
        print(f"    eventos:          {st['alertas_pressao']} alertas de pressão, "
              f"{st['flushes']} flushes")

    def salvar(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps({
            "working_context": self.working_context,
            "resumo_recursivo": self.resumo_recursivo,
            "fila": [asdict(m) for m in self.fila],
            "recall": [asdict(m) for m in self.recall],
            "archival": [asdict(f) for f in self.archival],
            "eventos": [asdict(e) for e in self.eventos],
            "estatisticas": self.estatisticas(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return caminho


# --------------------------------------------------------------------------- #
# Resumo com LLM (a versão do artigo)
# --------------------------------------------------------------------------- #

PROMPT_RESUMO = """Resuma a conversa abaixo preservando fatos, números, nomes e \
decisões importantes. O resumo será a única lembrança que restará destas \
mensagens, então não omita nada que possa ser necessário depois.

{resumo_anterior}CONVERSA:
{mensagens}

Responda apenas com o resumo, em no máximo {max_palavras} palavras."""


def resumir_com_llm(
    memoria: MemoriaHierarquica,
    despejadas: list[Mensagem],
    settings=None,
    max_palavras: int = 150,
) -> str:
    """
    Resumo recursivo usando o LLM, como no artigo.

    Alternativa ao resumo extrativo: produz um texto melhor, mas custa uma
    chamada por flush e deixa de ser determinístico. Em caso de falha, cai de
    volta no resumo extrativo, para que a memória nunca quebre por erro de rede.
    """
    from config import SETTINGS, get_client

    settings = settings or SETTINGS
    mensagens = "\n".join(m.formatar() for m in despejadas)
    anterior = (
        f"RESUMO ANTERIOR (incorpore-o):\n{memoria.resumo_recursivo}\n\n"
        if memoria.resumo_recursivo else ""
    )

    try:
        client = get_client(settings)
        resposta = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": PROMPT_RESUMO.format(
                resumo_anterior=anterior, mensagens=mensagens[:8000],
                max_palavras=max_palavras)}],
            temperature=0.0,
            max_tokens=max_palavras * 3,
        )
        texto = (resposta.choices[0].message.content or "").strip()
        if texto:
            return texto
    except Exception as erro:  # noqa: BLE001
        print(f"  [memória] resumo com LLM falhou ({erro}); usando o extrativo")

    return memoria._resumir(despejadas)
