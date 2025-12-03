from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from core.memory import MemoryEngine
from core.models import MemoryScope, MemoryType


class RJobError(RuntimeError):
    """Errore di esecuzione dello script R."""


def _find_script_path(script_name: str) -> Path:
    """
    Restituisce il path assoluto dello script R da eseguire.
    Assume che gli script R siano nella stessa cartella di questo file.
    """
    here = Path(__file__).resolve().parent  # .../r_agents
    script_path = here / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script R non trovato: {script_path}")
    return script_path


def run_r_job(
    script_name: str,
    job: Dict[str, Any],
    memory: MemoryEngine,
    memory_key: str,
    *,
    scope: MemoryScope = MemoryScope.PROJECT,
    type_: MemoryType = MemoryType.PROCEDURAL,
) -> Tuple[Dict[str, Any], str]:
    """
    Esegue uno script R passando un job JSON come argomento.

    - script_name: es. 'eda_generic.R'
    - job: dizionario Python -> verrà convertito in JSON e passato a R
    - memory: MemoryEngine per salvare l'output grezzo
    - memory_key: key logica con cui salvare il risultato in memoria
    - scope/type_: dove salvare il risultato nel DB delle memorie

    Ritorna (data_parsed, raw_stdout) dove:
      - data_parsed è il JSON parsato dallo stdout dello script R
      - raw_stdout è la stringa grezza (per debug)
    """
    script_path = _find_script_path(script_name)

    # JSON del job che R riceve come unico argomento
    job_json = json.dumps(job, ensure_ascii=False)

    cmd = ["Rscript", str(script_path), job_json]

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        # includiamo lo stderr per avere indizi di errore in R
        raise RJobError(
            f"Script R '{script_name}' terminato con codice {proc.returncode}.\n"
            f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )

    # Proviamo a parsare lo stdout come JSON
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RJobError(
            f"Impossibile parsare lo stdout di '{script_name}' come JSON: {exc}\n"
            f"STDOUT grezzo:\n{stdout}\n\nSTDERR:\n{stderr}"
        ) from exc

    # Salviamo comunque lo stdout grezzo in memoria (per debug/riuso)
    try:
        metadata = {
            "script_name": script_name,
            "job": job,
        }
        memory.store_item(
            scope=scope,
            type_=type_,
            key=memory_key,
            content=stdout,
            metadata=metadata,
        )
    except Exception:
        # non vogliamo che un errore di persistenza spezzi l'agent
        pass

    return data, stdout
