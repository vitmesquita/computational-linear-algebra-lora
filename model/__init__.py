"""Shim leve para expor o pacote em ``src/model`` a partir da raiz do repositorio."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC_MODEL = _HERE.parent / "src" / "model"

__path__ = [str(_HERE)]
if _SRC_MODEL.exists():
    __path__.append(str(_SRC_MODEL))
