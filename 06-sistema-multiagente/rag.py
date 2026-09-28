"""
RAG como ferramenta de um especialista.

Este não é um módulo de RAG completo: a versão com embeddings, busca híbrida,
reordenação e HyDE está no repositório blis-rag, dos módulos 03 e 04. Aqui o
RAG aparece no papel que o enunciado do módulo 06 pede, o de **ferramenta** de
um agente especialista, e por isso a implementação é a mínima que sustenta esse
papel com honestidade:

1. **Chunking por seção**, respeitando os títulos do Markdown. Um chunk que
   corta no meio de uma seção devolve contexto mutilado, e o agente responde
   com metade da regra.
2. **BM25 do zero**, com a fórmula clássica de Robertson e Sparck Jones. Sem
   dependência externa e sem custo por consulta.
3. **Citação obrigatória na saída.** Todo trecho volta rotulado com documento e
   seção, para que o fato registrado no estado compartilhado tenha procedência.

O que esta versão NÃO faz, e é honesto dizer: não usa embeddings, então não
acha um trecho que responde à pergunta com outras palavras. Para o corpus deste
módulo, que é pequeno e usa vocabulário próximo ao das perguntas, o BM25 basta.
Em um corpus real, não bastaria.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DIR_CORPUS = Path("data/corpus")


# --------------------------------------------------------------------------- #
# Tokenização
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "por", "para", "com", "sem", "que", "e", "ou",
    "se", "ao", "aos", "as", "e", "ser", "sao", "eh", "the", "of", "and",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def _sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    bruto = _TOKEN.findall(_sem_acentos((texto or "").lower()))
    return [t for t in bruto if t not in _STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """Um pedaço recuperável, com endereço suficiente para virar citação."""

    doc: str
    titulo: str
    secao: str
    texto: str
    indice: int = 0

    @property
    def fonte(self) -> str:
        """O rótulo que vai parar no fato registrado no estado compartilhado."""
        return f"{self.doc}#{self.secao}" if self.secao else self.doc


def dividir_por_secao(texto: str, doc: str, tamanho: int = 900,
                      sobreposicao: int = 150) -> list[Chunk]:
    """
    Divide o documento nos títulos de nível 2 e, se a seção for longa demais,
    em janelas com sobreposição.

    A sobreposição existe porque uma regra pode cair exatamente na emenda entre
    duas janelas. Sem ela, nenhum dos dois chunks contém a regra inteira.
    """
    linhas = texto.splitlines()
    titulo = ""
    secao_atual = ""
    buffer: list[str] = []
    blocos: list[tuple[str, str]] = []

    for linha in linhas:
        if linha.startswith("# "):
            titulo = linha[2:].strip()
            continue
        if linha.startswith("## "):
            if buffer:
                blocos.append((secao_atual, "\n".join(buffer).strip()))
                buffer = []
            secao_atual = linha[3:].strip()
            continue
        buffer.append(linha)

    if buffer:
        blocos.append((secao_atual, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    for secao, corpo in blocos:
        if not corpo:
            continue
        if len(corpo) <= tamanho:
            janelas = [corpo]
        else:
            janelas = []
            passo = max(1, tamanho - sobreposicao)
            for inicio in range(0, len(corpo), passo):
                janela = corpo[inicio:inicio + tamanho]
                if janela.strip():
                    janelas.append(janela.strip())
                if inicio + tamanho >= len(corpo):
                    break

        for janela in janelas:
            chunks.append(Chunk(doc=doc, titulo=titulo, secao=secao,
                                texto=janela, indice=len(chunks)))

    return chunks


# --------------------------------------------------------------------------- #
# Índice BM25
# --------------------------------------------------------------------------- #


class RAG:
    """
    Índice BM25 sobre os chunks do corpus.

        score(q, d) = Σ_t IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1-b+b·|d|/avgdl))

    O termo `b` normaliza pelo tamanho do documento: sem ele, chunk longo ganha
    sempre, só por conter mais palavras.
    """

    def __init__(self, diretorio: str | Path = DIR_CORPUS, tamanho: int = 900,
                 sobreposicao: int = 150, k1: float = 1.5, b: float = 0.75) -> None:
        self.diretorio = Path(diretorio)
        self.tamanho = tamanho
        self.sobreposicao = sobreposicao
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.freq: list[Counter] = []
        self.comprimentos: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self._carregar()

    def _carregar(self) -> None:
        if not self.diretorio.exists():
            return

        for caminho in sorted(self.diretorio.glob("*.md")):
            texto = caminho.read_text(encoding="utf-8")
            self.chunks.extend(
                dividir_por_secao(texto, caminho.stem, self.tamanho,
                                  self.sobreposicao)
            )

        if not self.chunks:
            return

        for chunk in self.chunks:
            tokens = tokenizar(f"{chunk.titulo} {chunk.secao} {chunk.texto}")
            self.freq.append(Counter(tokens))
            self.comprimentos.append(len(tokens))

        self.avgdl = sum(self.comprimentos) / len(self.comprimentos) or 1.0

        df: Counter = Counter()
        for contagem in self.freq:
            df.update(contagem.keys())

        n = len(self.chunks)
        self.idf = {
            termo: math.log((n - k + 0.5) / (k + 0.5) + 1.0)
            for termo, k in df.items()
        }

    def documentos(self) -> list[str]:
        vistos: list[str] = []
        for chunk in self.chunks:
            if chunk.doc not in vistos:
                vistos.append(chunk.doc)
        return vistos

    def titulo_de(self, doc: str) -> str:
        for chunk in self.chunks:
            if chunk.doc == doc:
                return chunk.titulo
        return doc

    def consultar(self, pergunta: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        tokens = tokenizar(pergunta)
        if not tokens or not self.chunks:
            return []

        pontuados: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            contagem = self.freq[i]
            norma = self.k1 * (1 - self.b + self.b * self.comprimentos[i] / self.avgdl)
            score = 0.0
            for termo in tokens:
                f = contagem.get(termo, 0)
                if f:
                    score += self.idf.get(termo, 0.0) * (f * (self.k1 + 1)) / (f + norma)
            if score > 0:
                pontuados.append((chunk, score))

        pontuados.sort(key=lambda par: par[1], reverse=True)
        return pontuados[:top_k]

    def formatar(self, resultados: list[tuple[Chunk, float]],
                 tamanho_trecho: int = 700) -> str:
        """
        Formata para o agente, sempre com a fonte à vista.

        O rótulo [fonte: doc#secao] não é enfeite: é o que o especialista copia
        para o campo de fonte ao registrar o fato no estado compartilhado.
        """
        if not resultados:
            return ("Nenhum trecho encontrado no corpus para essa consulta. "
                    "Não invente a resposta: diga que o corpus não cobre o assunto.")

        partes = []
        for i, (chunk, score) in enumerate(resultados, start=1):
            texto = chunk.texto.strip()
            if len(texto) > tamanho_trecho:
                texto = texto[:tamanho_trecho].rsplit(" ", 1)[0] + "..."
            partes.append(
                f"[{i}] {chunk.titulo} — seção '{chunk.secao}' "
                f"(score {score:.2f})\n[fonte: {chunk.fonte}]\n{texto}"
            )
        return "\n\n".join(partes)

    def __len__(self) -> int:
        return len(self.chunks)


_RAG: RAG | None = None


def obter_rag(diretorio: str | Path = DIR_CORPUS, **kwargs) -> RAG:
    """Índice carregado uma vez e reaproveitado entre as chamadas."""
    global _RAG
    if _RAG is None or str(_RAG.diretorio) != str(diretorio):
        _RAG = RAG(diretorio, **kwargs)
    return _RAG
