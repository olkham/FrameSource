"""Structured device descriptor returned by ``discover()`` implementations.

Historically each capture type's :meth:`discover` returned a list of plain
``dict`` objects with type-specific keys. :class:`DeviceInfo` replaces those
dicts with a typed :mod:`dataclasses` dataclass while remaining fully
dict-compatible (indexing, ``get``, ``in``, ``keys``/``items``) so existing
consumers that treated discovery results as dictionaries keep working
unchanged.

The historical webcam dicts used an ``'id'`` key; that is preserved as an
alias for :attr:`DeviceInfo.device_id`, and any extra type-specific keys are
kept flat via :attr:`DeviceInfo.metadata` so ``dev['id']``, ``dev['name']``,
``dev['index']`` and ``dev['backend_name']`` all still resolve.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DeviceInfo:
    """A discovered capture device.

    Instances behave like the plain dicts previously returned by
    ``discover()``: they support ``dev[key]``, ``dev.get(key)``,
    ``key in dev``, ``dev.keys()`` and ``dev.items()``. The ``'id'`` key is an
    alias for :attr:`device_id`, and :attr:`metadata` entries are exposed as
    flat top-level keys.

    Attributes:
        device_id: String identifier used as the ``source`` when creating a
            capture. Exposed under the alias ``'id'`` for dict-compatibility.
        index: OS/enumeration index, if applicable.
        name: Human-readable device name.
        driver: Backend/driver that produced this entry (e.g. ``'opencv'`` or
            ``'pyaudio'``).
        id_stable: True when ``device_id`` is stable across reboots/replug
            (e.g. a serial number); False for OS index-based identifiers.
        metadata: Free-form extra fields, kept flat for dict-compatibility.
    """

    device_id: str
    index: Optional[int] = None
    name: str = ""
    driver: str = ""
    id_stable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def _flat(self) -> dict[str, Any]:
        """Return the flat, dict-compatible view (``'id'`` alias + metadata)."""
        flat: dict[str, Any] = {
            "id": self.device_id,
            "index": self.index,
            "name": self.name,
            "driver": self.driver,
            "id_stable": self.id_stable,
        }
        flat.update(self.metadata)
        return flat

    def __getitem__(self, key: str) -> Any:
        if key in ("id", "device_id"):
            return self.device_id
        if key in ("index", "name", "driver", "id_stable"):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``get`` with alias/metadata fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        if key in ("id", "device_id", "index", "name", "driver", "id_stable"):
            return True
        return key in self.metadata

    def keys(self):
        """Return the flat top-level keys (``'id'`` alias + metadata keys)."""
        return self._flat().keys()

    def values(self):
        """Return the flat top-level values."""
        return self._flat().values()

    def items(self):
        """Return the flat ``(key, value)`` pairs."""
        return self._flat().items()

    def as_dict(self) -> dict[str, Any]:
        """Return a faithful dict of the dataclass fields.

        Unlike the flat dict-compat view, this uses the canonical
        ``'device_id'`` key and keeps ``metadata`` nested, so the result
        round-trips: ``DeviceInfo(**info.as_dict())`` reconstructs an
        equivalent instance.
        """
        return {
            "device_id": self.device_id,
            "index": self.index,
            "name": self.name,
            "driver": self.driver,
            "id_stable": self.id_stable,
            "metadata": dict(self.metadata),
        }
