"""
Configuração central do módulo 06.

Tudo vem de variáveis de ambiente, com padrões que funcionam sem nenhum .env
para os comandos offline. Os comandos que chamam o modelo exigem a chave e
avisam de forma clara quando ela falta.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    # --- Credenciais e modelo ---
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv("MODEL", "openai/gpt-4o-mini")
    )

    # --- Geração ---
    # Temperatura 0: quem decide roteamento e handoff precisa ser reproduzível.
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "700")))

    # --- Conversa em grupo (AutoGen) ---
    # Rodadas do gerente de conversa: escolher quem fala, coletar, transmitir.
    max_rodadas: int = field(default_factory=lambda: int(os.getenv("MAX_RODADAS", "8")))
    # Respostas automáticas consecutivas de um mesmo agente. No artigo é o que
    # impede dois agentes de conversarem para sempre sem intervenção.
    max_auto_resposta: int = field(
        default_factory=lambda: int(os.getenv("MAX_AUTO_RESPOSTA", "3"))
    )
    # Passos de ferramenta que um especialista pode dar antes de responder.
    max_passos_ferramenta: int = field(
        default_factory=lambda: int(os.getenv("MAX_PASSOS_FERRAMENTA", "4"))
    )
    # Quantas vezes o supervisor pode chamar o mesmo especialista seguidamente.
    max_repeticao_especialista: int = field(
        default_factory=lambda: int(os.getenv("MAX_REPETICAO_ESPECIALISTA", "2"))
    )

    # --- Salvaguardas (CAMEL) ---
    # Quantas detecções de modo de falha antes de encerrar a conversa.
    max_salvaguardas: int = field(
        default_factory=lambda: int(os.getenv("MAX_SALVAGUARDAS", "3"))
    )

    # --- RAG como ferramenta ---
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "3")))
    rag_tamanho_chunk: int = field(
        default_factory=lambda: int(os.getenv("RAG_TAMANHO_CHUNK", "900"))
    )
    rag_sobreposicao: int = field(
        default_factory=lambda: int(os.getenv("RAG_SOBREPOSICAO", "150"))
    )

    @property
    def tem_chave(self) -> bool:
        return bool(self.api_key)


SETTINGS = Settings()


def get_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or SETTINGS
    if not settings.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha a chave. O RAG, o estado compartilhado, as salvaguardas e "
            "os testes rodam sem chave: veja `python main.py --help`."
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/jgamacyber",
            "X-Title": "BLIS-Sistema-Multiagente",
        },
    )


def resumo_config(settings: Settings | None = None) -> str:
    s = settings or SETTINGS
    return (
        f"modelo={s.model} | temp={s.temperature} | rodadas={s.max_rodadas} "
        f"| auto_resposta={s.max_auto_resposta} | api_key="
        f"{'OK' if s.tem_chave else 'AUSENTE'}"
    )
