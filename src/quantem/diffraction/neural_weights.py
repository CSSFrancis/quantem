"""Where the neural spot finder's trained weights come from.

The weights are not part of quantem. They are downloaded on first use from
em-database and carry their own licence (non-commercial), which is printed the first
time any of them is fetched. Every name that ties quantem to a published file lives
in this module, so a renamed entry or a new detector is a one-line change here.

Each em-database entry serves one file, so only a detector whose file is the
``SpotUNetDiffraction`` entry's can be downloaded; any other detector needs its own
entry, or a local checkpoint passed by path.
"""

from __future__ import annotations

from pathlib import Path

#: em-database entries: the detector, the refine step's centre network (F5) and the
#: Friedel-partner centre network (P4).
DETECTOR_ENTRY = "SpotUNetDiffraction"
CENTRE_REFINER_ENTRY = "SpotUNetCentreRefiner"
FRIEDEL_PARTNER_ENTRY = "SpotUNetFriedelPartner"

#: Detector checkpoints by name. ``anyscale8`` is trained across disk sizes, so it
#: tolerates residual error in the disk-size estimate; ``dynamical-mask8`` adds a
#: disk-coverage head that centres each disk on its aperture rather than on where the
#: intensity sits inside it, which matters for dynamically filled disks.
DETECTOR_FILES = {
    "anyscale8": "spotunet_anyscale8_peaked_sharpened.pt",
    "dynamical-mask8": "spotunet_dynamical_mask8.pt",
    "production": "spotunet_production.pt",
    "base16": "spotunet_base16.pt",
}
DEFAULT_DETECTOR = "anyscale8"

_LICENSE_SHOWN = False


def detector_path(model: str | Path | None, version: str | None = None) -> Path:
    """Local path of a detector checkpoint, downloading it if needed.

    Parameters
    ----------
    model : str or Path, optional
        A name in :data:`DETECTOR_FILES`, or a path to a local checkpoint. ``None``
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
    if str(model) not in DETECTOR_FILES:
        path = Path(model)
        if not path.is_file():
            raise ValueError(
                f"model must be one of {sorted(DETECTOR_FILES)} or a checkpoint path, "
                f"got {model!r}."
            )
        return path
    return entry_path(DETECTOR_ENTRY, DETECTOR_FILES[str(model)], version)


def entry_path(entry_name: str, file_name: str | None = None, version: str | None = None) -> Path:
    """Download one em-database weights entry and return its local path.

    Parameters
    ----------
    entry_name : str
        The em-database entry, e.g. :data:`CENTRE_REFINER_ENTRY`.
    file_name : str, optional
        The file the caller needs; checked against the file the entry serves.
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
            "em-database, or pass a checkpoint path (model=, refine_centres=, "
            "friedel_partner=) to use local files."
        )
    entry = entry_class()
    if file_name is not None and entry.metadata.file != file_name:
        raise RuntimeError(
            f"em-database entry {entry_name!r} serves {entry.metadata.file!r}, not "
            f"{file_name!r}; pass that checkpoint's path instead."
        )
    if not _LICENSE_SHOWN:
        print(
            f"SpotUNet weights from em-database are licensed {entry.metadata.license}: "
            "free for research and other non-commercial use, including in-house use at "
            "a company; they may not be redistributed in, or used to provide, a product "
            "or service that is sold. See the em-database entry for the full terms."
        )
        _LICENSE_SHOWN = True
    return Path(entry.download(version=version, background=False, progressbar=False))
