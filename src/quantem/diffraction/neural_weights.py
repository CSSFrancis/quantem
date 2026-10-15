"""Where the neural spot finder's trained weights come from.

The weights are not part of quantem. They are downloaded on first use from
em-database and carry their own license (non-commercial), which is printed the first
time any of them is fetched. Every name that ties quantem to a published file lives
in this module, so a renamed entry or a new detector is a one-line change here.

Each network has a ``_Torch`` entry holding the PyTorch checkpoint quantem loads (the
entries without the suffix hold ONNX exports for other runtimes), pinned to
:data:`WEIGHTS_VERSION` so a newer upload cannot change detection under a caller.
"""

from __future__ import annotations

from pathlib import Path

#: The dated em-database version every entry is pinned to.
WEIGHTS_VERSION = "261015"

#: em-database entries of the center networks: DE CenterNet, the refine step, and
#: DE FriedelNet, the Friedel-partner step.
CENTER_REFINER_ENTRY = "DE_CenterNet_Torch"
FRIEDEL_PARTNER_ENTRY = "DE_FriedelNet_Torch"

#: The em-database entry of each DE SpotNet detector, by ``model=`` name. ``spotnet``
#: is trained across disk sizes, so it tolerates residual error in the disk-size
#: estimate; ``dynamical`` adds a disk-coverage head that centers each disk on its
#: aperture rather than on where the intensity sits inside it, which matters for
#: dynamically filled disks; ``production`` is slightly better on disks of the trained
#: size and weaker off it; ``base16`` is the earlier edge-trained detector, kept as a
#: fallback.
DETECTOR_ENTRIES = {
    "spotnet": "DE_SpotNet_Torch",
    "dynamical": "DE_SpotNet_Dynamical_Torch",
    "production": "DE_SpotNet_Production_Torch",
    "base16": "DE_SpotNet_Base16_Torch",
}
DEFAULT_DETECTOR = "spotnet"

#: The replay buffer teaching mixes in (see :mod:`~quantem.diffraction.neural_adapt`).
#: Not published yet: until it is, teaching takes a local path or runs without it.
REPLAY_ENTRY = "DE_SpotNet_Replay"

_LICENSE_SHOWN = False


def detector_path(model: str | Path | None, version: str | None = None) -> Path:
    """Local path of a detector checkpoint, downloading it if needed.

    Parameters
    ----------
    model : str or Path, optional
        A name in :data:`DETECTOR_ENTRIES`, or a path to a local checkpoint. ``None``
        is :data:`DEFAULT_DETECTOR`.
    version : str, optional
        em-database version to fetch instead of :data:`WEIGHTS_VERSION`.

    Returns
    -------
    Path
        The checkpoint file.
    """
    if model is None:
        model = DEFAULT_DETECTOR
    if str(model) in DETECTOR_ENTRIES:
        return entry_path(DETECTOR_ENTRIES[str(model)], version)
    path = Path(model)
    if not path.is_file():
        raise ValueError(
            f"model must be one of {sorted(DETECTOR_ENTRIES)} or a checkpoint path, got {model!r}."
        )
    return path


def entry_path(entry_name: str, version: str | None = None) -> Path:
    """Download one em-database weights entry and return its local path.

    Parameters
    ----------
    entry_name : str
        The em-database entry, e.g. :data:`CENTER_REFINER_ENTRY`.
    version : str, optional
        Dated version to fetch; ``None`` is :data:`WEIGHTS_VERSION`.

    Returns
    -------
    Path
        The downloaded file, checked against the entry's checksum.
    """
    try:
        from emdatabase import data
    except ImportError as error:
        raise ImportError(
            "The neural spot finder downloads its weights with em-database "
            "(pip install em-database); pass a checkpoint path instead to use local files."
        ) from error
    entry_class = getattr(data, entry_name, None)
    if entry_class is None:
        raise RuntimeError(
            f"This em-database release has no {entry_name!r} entry; upgrade em-database, "
            "or pass a checkpoint path (model=, refine_centers=, friedel_partner=) to use "
            "local files."
        )
    entry = entry_class()
    _show_license(entry.metadata.license)
    return Path(
        entry.download(version=version or WEIGHTS_VERSION, background=False, progressbar=False)
    )


def _show_license(license_id: str | None) -> None:
    global _LICENSE_SHOWN
    if _LICENSE_SHOWN:
        return
    print(
        f"Direct Electron network weights are licensed {license_id}: free for research "
        "and other non-commercial use, including in-house use at a company; they may not "
        "be redistributed in, or used to provide, a product or service that is sold."
    )
    _LICENSE_SHOWN = True
