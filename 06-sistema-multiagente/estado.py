"""
Estado compartilhado: o quadro onde os agentes escrevem.

Por que isso existe. Em um sistema multiagente, a tentação é passar tudo pela
conversa: o especialista responde, o supervisor lê, repassa ao próximo, e assim
por diante. Isso tem dois defeitos. O contexto cresce a cada turno, porque todo
mundo carrega a fala de todo mundo. E a informação se degrada, porque cada
repasse é uma paráfrase da paráfrase anterior.

A alternativa é separar o canal de coordenação do canal de dados. A conversa
carrega decisões ("agora é com você, faça X"). O estado compartilhado carrega
os fatos, com procedência. Quem precisa de um número lê o número, não a
lembrança que outro agente tem dele.

**Procedência é o ponto.** Todo fato registrado guarda quem escreveu e de onde
veio. Sem isso, a resposta final é uma afirmação sem rastro, e não há como
auditar de onde saiu cada número. É a mesma exigência que o RAG faz do
gerador: cite a fonte ou admita que não sabe.

O vocabulário do CoALA ajuda a situar: este arquivo é a memória de trabalho
compartilhada do sistema, e o histórico de mensagens é a memória episódica.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Fato:
    """Uma informação apurada por um agente, com procedência."""

    conteudo: str
    autor: str
    fonte: str = ""          # arquivo, documento do corpus, cálculo
    rodada: int = 0
    momento: float = field(default_factory=time.time)

    def formatar(self) -> str:
        origem = f" [fonte: {self.fonte}]" if self.fonte else ""
        return f"- {self.conteudo}{origem}  ({self.autor})"


@dataclass
class Mensagem:
    """Uma fala no canal de coordenação."""

    remetente: str
    destinatario: str
    conteudo: str
    rodada: int = 0
    tipo: str = "fala"       # fala, handoff, devolucao, salvaguarda, final

    def formatar(self) -> str:
        return f"[r{self.rodada}] {self.remetente} -> {self.destinatario}: {self.conteudo}"


@dataclass
class Artefato:
    """Um resultado nomeado que outro agente vai consumir inteiro."""

    nome: str
    conteudo: str
    autor: str
    rodada: int = 0


class EstadoCompartilhado:
    """
    Quadro compartilhado: fatos, artefatos, histórico e registro de handoffs.

    Todos os agentes leem daqui. Só escrevem através dos métodos, para que toda
    escrita fique com autor e rodada registrados.
    """

    def __init__(self, tarefa: str = "") -> None:
        self.tarefa = tarefa
        self.fatos: list[Fato] = []
        self.artefatos: dict[str, Artefato] = {}
        self.historico: list[Mensagem] = []
        self.handoffs: list[tuple[str, str, str]] = []   # (de, para, instrução)
        self.deteccoes: list[dict] = []
        self.rodada = 0
        self.encerrado = False
        self.motivo = ""

    # ------------------------------ escrita ------------------------------- #

    def registrar_fato(self, conteudo: str, autor: str, fonte: str = "") -> str:
        """Registra um fato apurado. Devolve a confirmação que o agente vê."""
        conteudo = (conteudo or "").strip()
        if not conteudo:
            return "erro: fato vazio."

        # Evita que o mesmo fato seja gravado de novo a cada rodada.
        for existente in self.fatos:
            if existente.conteudo.lower() == conteudo.lower():
                return f"fato já registrado por {existente.autor}; nada a fazer."

        self.fatos.append(Fato(conteudo=conteudo, autor=autor, fonte=fonte,
                               rodada=self.rodada))
        return f"fato registrado ({len(self.fatos)} no total)."

    def escrever_artefato(self, nome: str, conteudo: str, autor: str) -> str:
        nome = (nome or "").strip()
        if not nome:
            return "erro: artefato sem nome."
        self.artefatos[nome] = Artefato(nome=nome, conteudo=conteudo, autor=autor,
                                        rodada=self.rodada)
        return f"artefato '{nome}' salvo."

    def registrar_mensagem(self, remetente: str, destinatario: str, conteudo: str,
                           tipo: str = "fala") -> Mensagem:
        mensagem = Mensagem(remetente=remetente, destinatario=destinatario,
                            conteudo=conteudo, rodada=self.rodada, tipo=tipo)
        self.historico.append(mensagem)
        return mensagem

    def registrar_handoff(self, de: str, para: str, instrucao: str) -> None:
        self.handoffs.append((de, para, instrucao))

    def registrar_deteccao(self, tipo: str, agente: str, evidencia: str) -> None:
        self.deteccoes.append({
            "tipo": tipo, "agente": agente, "evidencia": evidencia,
            "rodada": self.rodada,
        })

    def encerrar(self, motivo: str) -> None:
        self.encerrado = True
        self.motivo = motivo

    # ------------------------------ leitura ------------------------------- #

    def ler_artefato(self, nome: str) -> str:
        artefato = self.artefatos.get((nome or "").strip())
        if artefato is None:
            disponiveis = ", ".join(self.artefatos) or "nenhum"
            return f"erro: artefato '{nome}' não existe. Disponíveis: {disponiveis}"
        return artefato.conteudo

    def buscar_fatos(self, consulta: str = "") -> list[Fato]:
        if not consulta.strip():
            return list(self.fatos)
        termos = set(consulta.lower().split())
        return [f for f in self.fatos
                if termos & set(f.conteudo.lower().split())]

    def fatos_formatados(self) -> str:
        if not self.fatos:
            return "(nenhum fato apurado ainda)"
        return "\n".join(f.formatar() for f in self.fatos)

    def especialistas_acionados(self) -> list[str]:
        vistos: list[str] = []
        for _de, para, _instrucao in self.handoffs:
            if para not in vistos:
                vistos.append(para)
        return vistos

    def resumo(self, ultimas: int = 4) -> str:
        """
        O que o supervisor vê antes de decidir a próxima fala.

        Só as últimas mensagens mais o quadro de fatos inteiro: é a separação
        entre coordenação (recente, volátil) e dados (acumulados, com fonte).
        """
        partes = [f"TAREFA: {self.tarefa}", "", "FATOS APURADOS ATÉ AGORA:",
                  self.fatos_formatados()]

        if self.artefatos:
            nomes = ", ".join(self.artefatos)
            partes += ["", f"ARTEFATOS DISPONÍVEIS: {nomes}"]

        if self.historico:
            recentes = self.historico[-ultimas:]
            partes += ["", "ÚLTIMAS MENSAGENS:"]
            partes += [m.formatar() for m in recentes]

        if self.especialistas_acionados():
            partes += ["", "ESPECIALISTAS JÁ ACIONADOS: "
                       + ", ".join(self.especialistas_acionados())]

        return "\n".join(partes)

    # ------------------------------ auditoria ----------------------------- #

    def estatisticas(self) -> dict:
        return {
            "rodadas": self.rodada,
            "mensagens": len(self.historico),
            "fatos": len(self.fatos),
            "artefatos": len(self.artefatos),
            "handoffs": len(self.handoffs),
            "especialistas_acionados": self.especialistas_acionados(),
            "deteccoes": len(self.deteccoes),
            "fatos_com_fonte": sum(1 for f in self.fatos if f.fonte),
        }

    def imprimir(self) -> None:
        st = self.estatisticas()
        print(f"\n  Estado compartilhado")
        print(f"    {st['rodadas']} rodadas, {st['mensagens']} mensagens, "
              f"{st['handoffs']} handoffs")
        print(f"    {st['fatos']} fatos ({st['fatos_com_fonte']} com fonte), "
              f"{st['artefatos']} artefatos")
        print(f"    especialistas: {', '.join(st['especialistas_acionados']) or 'nenhum'}")
        if st["deteccoes"]:
            print(f"    salvaguardas disparadas: {st['deteccoes']}")

    def como_dict(self) -> dict:
        return {
            "tarefa": self.tarefa,
            "encerrado": self.encerrado,
            "motivo": self.motivo,
            "estatisticas": self.estatisticas(),
            "fatos": [asdict(f) for f in self.fatos],
            "artefatos": [asdict(a) for a in self.artefatos.values()],
            "handoffs": [{"de": d, "para": p, "instrucao": i}
                         for d, p, i in self.handoffs],
            "deteccoes": self.deteccoes,
            "historico": [asdict(m) for m in self.historico],
        }

    def salvar(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(self.como_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return caminho
