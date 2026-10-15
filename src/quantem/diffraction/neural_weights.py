"""Where the neural spot finder's trained weights come from.

The weights are not part of quantem. They are downloaded on first use from
em-database and carry their own license (non-commercial), which is printed the first
time any of them is fetched. Every name that ties quantem to a published file lives
in this module, so a renamed entry or a new detector is a one-line change here.

Each em-database entry serves one checkpoint, so every detector that can be
downloaded has an entry of its own.
"""

from __future__ import annotations

from pathlib import Path

#: em-database entries of the center networks: the refine step (F5) and the
#: Friedel-partner step (P4).
CENTER_REFINER_ENTRY = "SpotUNetCenterRefiner"
FRIEDEL_PARTNER_ENTRY = "SpotUNetFriedelPartner"

#: The em-database entry of each detector, by name; ``None`` for one that is not
#: published and can only be used from a local checkpoint. ``anyscale8`` is trained
#: across disk sizes, so it tolerates residual error in the disk-size estimate;
#: ``dynamical-mask8`` adds a disk-coverage head that centers each disk on its aperture
#: rather than on where the intensity sits inside it, which matters for dynamically
#: filled disks; ``production`` is slightly better on disks of the trained size and
#: weaker off it.
DETECTOR_ENTRIES = {
    "anyscale8": "SpotUNetDiffraction",
    "dynamical-mask8": "SpotUNetDynamical",
    "production": "SpotUNetProduction",
    "base16": None,
}
DEFAULT_DETECTOR = "anyscale8"

_LICENSE_SHOWN = False


def detector_path(model: str | Path | None, version: str | None = None) -> Path:
    """Local path of a detector checkpoint, downloading it if needed.

    Parameters
    ----------
    model : str or Path, optional
        A name in :data:`DETECTOR_ENTRIES`, or a path to a local checkpoint. ``None``
        is :data:`DEFAULT_DETECTOR`.
    version : str, optional
        Dated em-database version to pin.

    Returns
    -------
    Path
        The checkpoint file.
    """
    if model is None:
        model = DEFAULT_DETECTOR
    if str(model) in DETECTOR_ENTRIES:
        entry_name = DETECTOR_ENTRIES[str(model)]
        if entry_name is None:
            raise ValueError(
                f"Detector {model!r} is not published on em-database; pass its checkpoint path."
            )
        return entry_path(entry_name, version)
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
        Dated version to pin; ``None`` fetches the latest.

    Returns
    -------
    Path
        The downloaded file.
    """
    global _LICENSE_SHOWN
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
            f"This em-database release has no {entry_name!r} entry yet; upgrade "
            "em-database, or pass a checkpoint path (model=, refine_centers=, "
            "friedel_partner=) to use local files."
        )
    entry = entry_class()
    if not _LICENSE_SHOWN:
        print(
            f"SpotUNet weights from em-database are licensed {entry.metadata.license}: "
            "free for research and other non-commercial use, including in-house use at "
            "a company; they may not be redistributed in, or used to provide, a product "
            "or service that is sold. See the em-database entry for the full terms."
        )
        _LICENSE_SHOWN = True
    return Path(entry.download(version=version, background=False, progressbar=False))
