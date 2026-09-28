"""
Configuração central do módulo 05 — Agente ReAct.
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
    # Temperatura 0 por padrão: um agente que decide ações deve ser o mais
    # determinístico possível, para que um erro seja reproduzível e depurável.
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "600")))

    # --- Loop ReAct ---
    # Limite de passos. Estourar vira TLE (Task Limit Exceeded) na taxonomia
    # do AgentBench, que é um resultado informativo, não um crash.
    max_passos: int = field(default_factory=lambda: int(os.getenv("MAX_PASSOS", "8")))
    # Quantas saídas malformadas seguidas antes de desistir (Invalid Format).
    max_erros_formato: int = field(
        default_factory=lambda: int(os.getenv("MAX_ERROS_FORMATO", "3"))
    )
    # Detecção de laço: quantas ações idênticas repetidas encerram a execução.
    max_repeticoes: int = field(
        default_factory=lambda: int(os.getenv("MAX_REPETICOES", "3"))
    )

    # --- Memória (MemGPT) ---
    # Orçamento da "janela de contexto" simulada, em tokens estimados.
    janela_contexto: int = field(
        default_factory=lambda: int(os.getenv("JANELA_CONTEXTO", "4000"))
    )
    # 70% da janela dispara o aviso de pressão de memória.
    limiar_pressao: float = field(
        default_factory=lambda: float(os.getenv("LIMIAR_PRESSAO", "0.7"))
    )
    # 100% dispara o flush, que despeja uma fração da fila.
    fracao_flush: float = field(
        default_factory=lambda: float(os.getenv("FRACAO_FLUSH", "0.5"))
    )
    # Tamanho máximo do working context, em caracteres.
    tamanho_working_context: int = field(
        default_factory=lambda: int(os.getenv("TAMANHO_WORKING_CONTEXT", "1200"))
    )
    # Resultados por página nas buscas de memória (evita estourar o contexto).
    pagina_memoria: int = field(
        default_factory=lambda: int(os.getenv("PAGINA_MEMORIA", "3"))
    )

    @property
    def tem_chave(self) -> bool:
        return bool(self.api_key)

    @property
    def tokens_pressao(self) -> int:
        return int(self.janela_contexto * self.limiar_pressao)


SETTINGS = Settings()


def get_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or SETTINGS
    if not settings.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha a chave. As ferramentas, a memória e os testes rodam sem "
            "chave: use `python main.py ferramentas` e `python main.py memoria`."
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/jgamacyber",
            "X-Title": "BLIS-Agente-ReAct",
        },
    )


def resumo_config(settings: Settings | None = None) -> str:
    s = settings or SETTINGS
    return (
        f"modelo={s.model} | temp={s.temperature} | max_passos={s.max_passos} "
        f"| janela={s.janela_contexto} tok | api_key="
        f"{'OK' if s.tem_chave else 'AUSENTE'}"
    )
