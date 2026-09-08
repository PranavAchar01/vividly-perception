"""Dataset registry: licence buckets, physical separation, manifest enforcement.

One YAML in configs/datasets/ == one dataset. Adding a future source (avatar renders,
pilot-derived data, a new public set) is one YAML entry plus one converter module and
nothing else.

Enforcement, weakest to strongest:
  1. typed metadata validated at load time (the cross-field invariants in _validate());
  2. physical trees: TRAIN_OK under data/train_ok/, TRAIN_PROTOTYPE under
     data/train_prototype/, EVAL_ONLY under data/eval_only/;
  3. build_train_manifest() refuses non-TRAIN_OK and re-resolves paths with symlinks
     followed, so data/train_ok/x -> data/eval_only/y is caught; the prototype weight
     class has to be asked for by name (weight_class="prototype") before a
     TRAIN_PROTOTYPE entry enters a train manifest, and the manifest records which class
     it is;
  4. LEGAL_HOLD entries have no local root at all and are never fetched.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "datasets"
SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Bucket(StrEnum):
    TRAIN_OK = "TRAIN_OK"  # may train shippable (commercial) weights
    # A research release whose attached licence, the download-page grant, permits
    # commercial use may train prototype weights. Those weights and the numbers they
    # produce are internal demonstration material: under POLICY.md, section 3, "Prototype
    # corpus admission", they are never deployed, distributed or quoted externally without
    # counsel sign-off on the conflict note in the entry, and where they are quoted the
    # attribution the grant names travels with them. A TRAIN_OK corpus may never derive
    # from this bucket. Physical tree: data/train_prototype/.
    TRAIN_PROTOTYPE = "TRAIN_PROTOTYPE"
    EVAL_ONLY = "EVAL_ONLY"  # measures only; never trains anything (POLICY.md, rule 6)
    LEGAL_HOLD = "LEGAL_HOLD"  # never fetched, never resolved


#: The two weight classes a training manifest can be built for. "shippable" is the
#: default and admits TRAIN_OK only; "prototype" additionally admits TRAIN_PROTOTYPE and is
#: what a training configuration declaring `weight_class: prototype` corresponds to. The
#: standing rule is POLICY.md, rule 6, evaluation-only data never trains.
WEIGHT_CLASSES = ("shippable", "prototype")


class Status(StrEnum):
    PLANNED = "planned"  # registered, not requested
    REQUESTED = "requested"  # human action submitted, waiting
    AVAILABLE = "available"  # bytes on disk, verified
    CONVERTED = "converted"  # bytes on disk and converted to COCO or MOT
    BLOCKED = "blocked"  # cannot proceed (counsel, dead host, gate)
    DROPPED = "dropped"  # decided against; kept as a record


class CommercialUse(StrEnum):
    PERMITTED = "permitted"
    PROHIBITED = "prohibited"
    CONFLICTING = "conflicting"  # primary sources disagree; conservative reading governs
    UNREVIEWED = "unreviewed"


# ---------------------------------------------------------------- exceptions
class RegistryError(RuntimeError): ...


class RegistryValidationError(RegistryError): ...


class EvalOnlyInTrainManifest(RegistryError): ...


class LegalHoldViolation(RegistryError): ...


class DatasetNotAvailable(RegistryError): ...


class PathEscapesBucket(RegistryError): ...


class CounselGateOpen(RegistryError): ...


# ---------------------------------------------------------------- value types
@dataclass(frozen=True, slots=True)
class FileRecord:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TermsAcceptance:
    """The role that accepted a dataset's terms on the project's behalf, when, and how.

    POLICY.md, section 3, "Registration before bytes", requires the record captured at the
    moment a person registers, not reconstructed later from memory. `evidence_path` is
    repo-relative and optional: a saved confirmation email or a screenshot lives outside
    the repo more often than not.
    """

    accepted_by: str
    accepted_utc: str
    method: str
    evidence_path: str | None


@dataclass(frozen=True, slots=True)
class PrototypeGrant:
    """The recorded grant that lets a TRAIN_PROTOTYPE entry train prototype weights.

    Required on every TRAIN_PROTOTYPE entry. `attribution` is the exact credit line
    external material carries; `ship_policy` is the sentence that stops a prototype
    weight from being mistaken for a shippable one.
    """

    decision: str  # lowercase slug naming the grant, for example "fall-prototype-sets"
    decided_utc: str  # quoted ISO 8601 date, validated like counsel_cleared_utc
    decided_by: str
    attribution: str
    ship_policy: str


@dataclass(frozen=True, slots=True)
class EventGrant:
    """The recorded grant that admits a dataset of acted medical events.

    POLICY.md, rule 4 bans acted medical events by default; this grant is the recorded
    exception, and `role` is its whole point. 'train_interim' admits the entry to interim
    prototype event-model training (TRAIN_PROTOTYPE bucket, with all the prototype
    machinery on top); 'test_only' admits it as an engineering test set that trains
    nothing (EVAL_ONLY bucket). A dataset that trains loses its test role permanently, so
    one entry carries exactly one role.
    """

    decision: str  # lowercase slug naming the grant, for example "interim-event-models"
    decided_utc: str  # quoted ISO 8601 date, validated like counsel_cleared_utc
    decided_by: str
    role: str  # 'train_interim' | 'test_only'


EVENT_GRANT_ROLES = ("train_interim", "test_only")


@dataclass(frozen=True, slots=True)
class Licensing:
    license: str
    license_url: str
    license_verbatim: str
    license_conflict_note: str | None
    commercial_use: CommercialUse
    attribution_required: bool
    redistribute: bool
    counsel_questions: tuple[str, ...]
    counsel_cleared_utc: str | None
    terms_acceptance: TermsAcceptance | None = None
    prototype_grant: PrototypeGrant | None = None
    event_grant: EventGrant | None = None


@dataclass(frozen=True, slots=True)
class Provenance:
    collector: str
    imagery_source: str
    contains_real_people: bool
    simulated_medical_events: bool
    citation: str
    #: True clinical footage of real medical events (for example EMU seizure
    #: recordings). A separate axis from simulated_medical_events: real events are not
    #: acted, but they carry more weight rather than less, so they pass through the same
    #: event_grant gate. Defaults False for every ordinary set.
    real_medical_events: bool = False


@dataclass(frozen=True, slots=True)
class Acquisition:
    method: str
    urls: tuple[str, ...]
    mirrors: tuple[str, ...]
    human_required: bool
    human_steps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Layout:
    root: str | None  # repo-relative
    images: tuple[tuple[str, str], ...]  # (split, relpath); tuples keep the entry hashable
    annotations: tuple[tuple[str, str], ...]

    def image_rel(self, split: str) -> str:
        return dict(self.images)[split]

    def ann_rel(self, split: str) -> str:
        return dict(self.annotations)[split]


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    display_name: str
    task: tuple[str, ...]
    bucket: Bucket
    status: Status
    licensing: Licensing
    provenance: Provenance
    acquisition: Acquisition
    layout: Layout
    ann_format: str
    converter: str | None
    files: tuple[FileRecord, ...]
    hashes_unavailable_reason: str | None
    counts: tuple[tuple[str, int], ...]
    derived_from: tuple[str, ...]
    notes: str
    source_yaml: str

    @property
    def trainable(self) -> bool:
        """May train shippable weights (TRAIN_OK only)."""
        return self.bucket is Bucket.TRAIN_OK

    @property
    def prototype_trainable(self) -> bool:
        """May train prototype weights (TRAIN_OK or TRAIN_PROTOTYPE)."""
        return self.bucket in (Bucket.TRAIN_OK, Bucket.TRAIN_PROTOTYPE)


# ---------------------------------------------------------------- manifests
@dataclass(frozen=True, slots=True)
class ManifestEntry:
    name: str
    bucket: str
    split: str
    data_root: str  # repo-relative, as written in the YAML
    ann_file: str  # repo-relative
    resolved_root: str  # absolute, symlinks followed; this is the audit record
    license: str
    ann_format: str


@dataclass(frozen=True, slots=True)
class Manifest:
    kind: str  # "train" | "eval"
    split: str
    created_utc: str
    schema_version: int
    entries: tuple[ManifestEntry, ...]
    # "shippable" unless a train manifest was built with weight_class="prototype".
    # Eval manifests carry "shippable" too: the field describes what may be trained from
    # the manifest, and an eval manifest trains nothing.
    weight_class: str = "shippable"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "split": self.split,
            "created_utc": self.created_utc,
            "schema_version": self.schema_version,
            "weight_class": self.weight_class,
            "content_sha256": self.content_sha256,
            "entries": [asdict(e) for e in self.entries],
        }

    @property
    def content_sha256(self) -> str:
        """Hash of the entries only: stable across runs, independent of created_utc.

        The hash covers the manifest entries and nothing else, so two builds over the same
        datasets and splits produce the same value and two manifests are comparable by
        content rather than by build time. Checkpoint provenance is a separate record:
        POLICY.md, rule 8 names where it lives for each kind of checkpoint.
        """
        payload = [asdict(e) for e in self.entries]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------- parsing
def _req(d: Mapping, key: str, where: str):
    if key not in d:
        raise RegistryValidationError(f"{where}: missing required key '{key}'")
    return d[key]


_CLEARED_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?Z?)?$")

#: A grant decision is named by a lowercase slug that says what was granted, for example
#: "fall-prototype-sets". An opaque serial number is not accepted: the value is read by a
#: person auditing the registry, so it has to carry its own meaning.
_DECISION_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")


def _decision_slug(value: object, where: str, field: str) -> str:
    """Validate a grant decision name: a lowercase slug of 3 to 41 characters.

    Letters, digits and hyphens only, and the first character is a letter or a digit.
    """
    text = str(value)
    if not _DECISION_SLUG.match(text):
        raise RegistryValidationError(
            f"{where}: {field} must be a lowercase slug matching "
            f"{_DECISION_SLUG.pattern} that names what was granted, for example "
            f'"fall-prototype-sets", got {text!r}'
        )
    return text


def _counsel_cleared(value: object, where: str) -> str | None:
    """Validate `licensing.counsel_cleared_utc`: None, or a quoted ISO 8601 UTC string.

    This field opens the counsel gate, which makes it a legal record, not a comment.
    Truthy junk ("TBD", "pending", `true`) must not open the gate, and a bare YAML date
    parses as `datetime.date`, which breaks the declared `str | None` type downstream,
    so the value has to be quoted in the YAML.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not _CLEARED_SHAPE.match(value):
        raise RegistryValidationError(
            f"{where}: counsel_cleared_utc must be null or a quoted ISO 8601 UTC date "
            f'string like "2026-08-08", got {value!r}. A truthy non-date must never '
            "open the counsel gate."
        )
    return value


def _pairs(d: Mapping | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in (d or {}).items()))


def _prototype_grant(raw: object, where: str) -> PrototypeGrant | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(
            f"{where}: licensing.prototype_grant must be a mapping with decision, "
            "decided_utc, decided_by, attribution, ship_policy (or null)"
        )
    decided = raw.get("decided_utc")
    if not isinstance(decided, str) or not _CLEARED_SHAPE.match(decided):
        raise RegistryValidationError(
            f"{where}: prototype_grant.decided_utc must be a quoted ISO 8601 UTC date "
            f'string like "2026-08-22", got {decided!r}'
        )
    grant = PrototypeGrant(
        decision=_decision_slug(_req(raw, "decision", where), where, "prototype_grant.decision"),
        decided_utc=decided,
        decided_by=str(_req(raw, "decided_by", where)),
        attribution=str(_req(raw, "attribution", where)),
        ship_policy=str(_req(raw, "ship_policy", where)),
    )
    for field_name in ("decision", "decided_by", "attribution", "ship_policy"):
        if not getattr(grant, field_name).strip():
            raise RegistryValidationError(
                f"{where}: prototype_grant.{field_name} must be non-empty; the grant is a "
                "record of what was granted and by which role, not a flag"
            )
    return grant


def _event_grant(raw: object, where: str) -> EventGrant | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(
            f"{where}: licensing.event_grant must be a mapping with decision, "
            "decided_utc, decided_by, role (or null)"
        )
    decided = raw.get("decided_utc")
    if not isinstance(decided, str) or not _CLEARED_SHAPE.match(decided):
        raise RegistryValidationError(
            f"{where}: event_grant.decided_utc must be a quoted ISO 8601 UTC date "
            f'string like "2026-08-26", got {decided!r}'
        )
    grant = EventGrant(
        decision=_decision_slug(_req(raw, "decision", where), where, "event_grant.decision"),
        decided_utc=decided,
        decided_by=str(_req(raw, "decided_by", where)),
        role=str(_req(raw, "role", where)),
    )
    for field_name in ("decision", "decided_by"):
        if not getattr(grant, field_name).strip():
            raise RegistryValidationError(
                f"{where}: event_grant.{field_name} must be non-empty; the grant is a "
                "record of what was granted and by which role, not a flag"
            )
    if grant.role not in EVENT_GRANT_ROLES:
        raise RegistryValidationError(
            f"{where}: event_grant.role must be one of {list(EVENT_GRANT_ROLES)}, "
            f"got {grant.role!r}"
        )
    return grant


def _terms_acceptance(raw: object, where: str) -> TermsAcceptance | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(
            f"{where}: licensing.terms_acceptance must be a mapping with "
            "accepted_by, accepted_utc, method, evidence_path (or null)"
        )
    evidence = raw.get("evidence_path")
    return TermsAcceptance(
        accepted_by=str(_req(raw, "accepted_by", where)),
        accepted_utc=str(_req(raw, "accepted_utc", where)),
        method=str(_req(raw, "method", where)),
        evidence_path=str(evidence) if evidence else None,
    )


def _parse(path: Path) -> Dataset:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    where = str(path)
    version = int(raw.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise RegistryValidationError(f"{where}: schema_version {version} != {SCHEMA_VERSION}")
    lic, prov, acq, lay = (
        _req(raw, k, where) for k in ("licensing", "provenance", "acquisition", "layout")
    )
    return Dataset(
        name=str(_req(raw, "name", where)),
        display_name=str(raw.get("display_name", raw["name"])),
        task=tuple(raw.get("task", ())),
        bucket=Bucket(_req(raw, "bucket", where)),
        status=Status(_req(raw, "status", where)),
        licensing=Licensing(
            license=str(_req(lic, "license", where)),
            license_url=str(lic.get("license_url", "")),
            license_verbatim=str(lic.get("license_verbatim", "")),
            license_conflict_note=lic.get("license_conflict_note"),
            commercial_use=CommercialUse(_req(lic, "commercial_use", where)),
            attribution_required=bool(lic.get("attribution_required", True)),
            redistribute=bool(lic.get("redistribute", False)),
            counsel_questions=tuple(lic.get("counsel_questions", ()) or ()),
            counsel_cleared_utc=_counsel_cleared(lic.get("counsel_cleared_utc"), where),
            terms_acceptance=_terms_acceptance(lic.get("terms_acceptance"), where),
            prototype_grant=_prototype_grant(lic.get("prototype_grant"), where),
            event_grant=_event_grant(lic.get("event_grant"), where),
        ),
        provenance=Provenance(
            collector=str(_req(prov, "collector", where)),
            imagery_source=str(_req(prov, "imagery_source", where)),
            contains_real_people=bool(_req(prov, "contains_real_people", where)),
            simulated_medical_events=bool(_req(prov, "simulated_medical_events", where)),
            citation=str(prov.get("citation", "")),
            real_medical_events=bool(prov.get("real_medical_events", False)),
        ),
        acquisition=Acquisition(
            method=str(_req(acq, "method", where)),
            urls=tuple(acq.get("urls", ()) or ()),
            mirrors=tuple(acq.get("mirrors", ()) or ()),
            human_required=bool(_req(acq, "human_required", where)),
            human_steps=tuple(acq.get("human_steps", ()) or ()),
        ),
        layout=Layout(
            root=lay.get("root"),
            images=_pairs(lay.get("images")),
            annotations=_pairs(lay.get("annotations")),
        ),
        ann_format=str(raw.get("ann_format", "")),
        converter=raw.get("converter"),
        files=tuple(
            FileRecord(str(f["name"]), int(f["size_bytes"]), str(f["sha256"]))
            for f in (raw.get("files") or [])
        ),
        hashes_unavailable_reason=raw.get("hashes_unavailable_reason"),
        counts=tuple(sorted((str(k), int(v)) for k, v in (raw.get("counts") or {}).items())),
        derived_from=tuple(raw.get("derived_from", ()) or ()),
        notes=str(raw.get("notes", "")),
        source_yaml=str(path),
    )


def _validate(ds: Dataset, stem: str) -> None:
    w = ds.source_yaml
    if ds.name != stem or not _NAME_RE.match(ds.name):
        raise RegistryValidationError(
            f"{w}: name '{ds.name}' must match filename stem and {_NAME_RE.pattern}"
        )

    # POLICY.md, rule 4: acted and simulated medical events are banned from training and
    # evaluation sets by default. Under the recorded carve-out, an acted-event dataset
    # may enter only with an explicit event_grant, and only into the bucket its role
    # names: 'train_interim' into TRAIN_PROTOTYPE (interim prototype event models, never
    # shipped without counsel), 'test_only' into EVAL_ONLY (engineering test sets that
    # train nothing). A dataset that trains loses its test role, so roles are exclusive.
    # Real clinical event footage (EMU seizure recordings) rides the same gate. Real
    # events are not the acted events the rule bans, but they are the most sensitive data
    # in the repository, so admitting one without an explicit recorded grant would be
    # worse rather than better.
    if ds.provenance.simulated_medical_events or ds.provenance.real_medical_events:
        egrant = ds.licensing.event_grant
        if egrant is None:
            raise RegistryValidationError(
                f"{w}: a dataset containing medical events (acted or real) requires "
                "licensing.event_grant. POLICY.md, rule 4 bans acted medical events and "
                "extends the gate to real clinical footage; only a recorded grant admits "
                "an exception"
            )
        expected = Bucket.TRAIN_PROTOTYPE if egrant.role == "train_interim" else Bucket.EVAL_ONLY
        if ds.bucket is not expected:
            raise RegistryValidationError(
                f"{w}: event_grant.role '{egrant.role}' requires bucket {expected.value}, "
                f"got {ds.bucket.value}; the train and test roles are exclusive"
            )
    elif ds.licensing.event_grant is not None:
        raise RegistryValidationError(
            f"{w}: event_grant on an entry without simulated_medical_events or "
            "real_medical_events; the grant exists only for event data"
        )

    # The project never records its own footage, so this schema carries no
    # consent-policy machinery for self-captured material. Pilot-derived data, if it ever
    # exists, arrives as a new entry type designed with counsel at that time.

    # A shippable-weight corpus needs a commercial grant and no open counsel question, and
    # carries no prototype grant. The demotion this invariant is written against is
    # recorded in docs/COUNSEL_QUESTIONS.md under CQ-3-mpii-commercial-training.
    if ds.bucket is Bucket.TRAIN_OK:
        if ds.licensing.commercial_use is not CommercialUse.PERMITTED:
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_OK requires commercial_use == 'permitted', "
                f"got '{ds.licensing.commercial_use.value}'"
            )
        if ds.licensing.counsel_questions:
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_OK requires no open counsel questions, got "
                f"{list(ds.licensing.counsel_questions)}"
            )
        if ds.licensing.prototype_grant is not None:
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_OK carries no prototype_grant; the grant marks data "
                "that may train prototype weights only, and TRAIN_OK needs no such "
                "carve-out"
            )

    # A TRAIN_PROTOTYPE entry is a recorded grant, not a relaxed TRAIN_OK. It must
    # (a) carry the grant that names what was granted and how results are credited,
    # (b) rest on a primary-source grant that is at least arguably commercial (a flatly
    # prohibited licence cannot be prototype-trained under any grant), (c) keep
    # attribution on, and (d) have the mechanical counsel gate open, or the manifest
    # layer would refuse it anyway and the bucket would be decoration.
    if ds.bucket is Bucket.TRAIN_PROTOTYPE:
        if ds.licensing.prototype_grant is None:
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_PROTOTYPE requires licensing.prototype_grant, the "
                "recorded grant that admits this entry to prototype training"
            )
        if ds.licensing.commercial_use not in (
            CommercialUse.PERMITTED,
            CommercialUse.CONFLICTING,
        ):
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_PROTOTYPE requires commercial_use 'permitted' or "
                f"'conflicting' (a primary source must grant it), got "
                f"'{ds.licensing.commercial_use.value}'"
            )
        if not ds.licensing.attribution_required:
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_PROTOTYPE requires attribution_required: true; "
                "crediting the source in every external artefact is a term of the grant, "
                "not a courtesy"
            )
        if ds.licensing.counsel_questions and not ds.licensing.counsel_cleared_utc:
            raise RegistryValidationError(
                f"{w}: bucket TRAIN_PROTOTYPE with open counsel questions "
                f"{list(ds.licensing.counsel_questions)} requires counsel_cleared_utc (the "
                "recorded mechanical clearance); otherwise no manifest could ever include it"
            )

    # Conflicting primary sources must be written down, not silently resolved.
    if (
        ds.licensing.commercial_use is CommercialUse.CONFLICTING
        and not ds.licensing.license_conflict_note
    ):
        raise RegistryValidationError(
            f"{w}: commercial_use 'conflicting' requires license_conflict_note"
        )

    # Bucket determines the physical tree.
    expected = {
        Bucket.TRAIN_OK: "data/train_ok/",
        Bucket.TRAIN_PROTOTYPE: "data/train_prototype/",
        Bucket.EVAL_ONLY: "data/eval_only/",
    }
    if ds.bucket is Bucket.LEGAL_HOLD:
        if ds.layout.root is not None:
            raise RegistryValidationError(
                f"{w}: LEGAL_HOLD datasets must have layout.root: null (never fetched)"
            )
    elif ds.status is Status.DROPPED:
        pass  # dropped entries are records, not data
    else:
        root = ds.layout.root
        # Normalise before the prefix test: a raw startswith accepts
        # 'data/train_ok/../eval_only/x', which reads as the training tree and resolves to
        # the evaluation tree. An entry that declares no images has no other load-time
        # guard, so the check has to hold here.
        if not root or not posixpath.normpath(root).startswith(expected[ds.bucket]):
            raise RegistryValidationError(
                f"{w}: bucket {ds.bucket.value} requires layout.root under "
                f"'{expected[ds.bucket]}', got '{root}'"
            )
        # layout.images entries are joined to root and read directly by the dataset
        # readers and the eval runner, so a '..' segment there escapes the licence bucket
        # exactly as a bad root would: image paths may not escape the bucket (POLICY.md
        # rule 6). Hold every image path to the same prefix invariant as root, lexically
        # normalised so '../../eval_only/...' cannot slip past the string check.
        # image_dir()'s resolve()-based gate catches the symlink variant at read time; this
        # closes the load-time hole.
        prefix = expected[ds.bucket]
        for split, rel in ds.layout.images:
            joined = posixpath.normpath(posixpath.join(root, rel))
            if not (joined + "/").startswith(prefix):
                raise RegistryValidationError(
                    f"{w}: layout.images['{split}'] '{rel}' joins to '{joined}', outside "
                    f"'{prefix}'. Image paths may not escape the licence bucket."
                )

    # POLICY.md, section 3, "Registration before bytes": a gated set is acquired because a
    # person in a named role accepted a publisher's terms. Record the role, the time and
    # the method at the moment of registration, not afterwards from memory. Only the
    # statuses that mean the bytes are on disk demand the record: planned and requested are
    # pre-acquisition, and dropped and blocked sets were never acquired at all. No
    # registered entry currently satisfies both arms, so nothing is held back by it today.
    if (
        ds.acquisition.human_required
        and ds.status in (Status.AVAILABLE, Status.CONVERTED)
        and ds.licensing.terms_acceptance is None
    ):
        raise RegistryValidationError(
            f"{w}: acquisition.human_required is true and status is '{ds.status.value}', so "
            "licensing.terms_acceptance {accepted_by, accepted_utc, method, evidence_path} "
            "is required; the acceptance record is part of the entry, not a memory"
        )

    for f in ds.files:
        if not _SHA256_RE.match(f.sha256):
            raise RegistryValidationError(f"{w}: file '{f.name}' has a malformed sha256")
    if (
        ds.status in (Status.AVAILABLE, Status.CONVERTED)
        and not ds.files
        and not ds.hashes_unavailable_reason
    ):
        raise RegistryValidationError(
            f"{w}: status '{ds.status.value}' requires file hashes or hashes_unavailable_reason"
        )


# ---------------------------------------------------------------- registry
class Registry:
    """Loaded, validated set of dataset entries rooted at a repo."""

    def __init__(self, entries: Mapping[str, Dataset], *, repo_root: Path) -> None:
        self._entries = dict(entries)
        self.repo_root = Path(repo_root).resolve()
        self.train_ok_root = (self.repo_root / "data" / "train_ok").resolve()
        self.train_prototype_root = (self.repo_root / "data" / "train_prototype").resolve()
        self.eval_only_root = (self.repo_root / "data" / "eval_only").resolve()
        self._validate_cross_entry()

    @classmethod
    def from_dir(cls, config_dir: Path = CONFIG_DIR, *, repo_root: Path = REPO_ROOT) -> Registry:
        entries: dict[str, Dataset] = {}
        for path in sorted(Path(config_dir).glob("*.yaml")):
            if path.name.startswith("_"):
                continue  # _template.yaml is documentation
            ds = _parse(path)
            _validate(ds, path.stem)
            entries[ds.name] = ds
        if not entries:
            raise RegistryValidationError(f"no dataset entries found under {config_dir}")
        return cls(entries, repo_root=repo_root)

    def _validate_cross_entry(self) -> None:
        for ds in self._entries.values():
            for parent in ds.derived_from:
                if parent not in self._entries:
                    raise RegistryValidationError(
                        f"{ds.source_yaml}: derived_from '{parent}' is not registered"
                    )
                if (
                    ds.bucket is Bucket.TRAIN_OK
                    and self._entries[parent].bucket is not Bucket.TRAIN_OK
                ):
                    raise RegistryValidationError(
                        f"{ds.source_yaml}: TRAIN_OK dataset derives from non-TRAIN_OK '{parent}'; "
                        "a derived corpus cannot launder a licence"
                    )
                if ds.bucket is Bucket.TRAIN_PROTOTYPE and self._entries[parent].bucket not in (
                    Bucket.TRAIN_OK,
                    Bucket.TRAIN_PROTOTYPE,
                ):
                    raise RegistryValidationError(
                        f"{ds.source_yaml}: TRAIN_PROTOTYPE dataset derives from "
                        f"{self._entries[parent].bucket.value} '{parent}'; the prototype grant "
                        "admits research releases with a commercial grant, not EVAL_ONLY or "
                        "LEGAL_HOLD data"
                    )

    # -- access -----------------------------------------------------------
    def __getitem__(self, name: str) -> Dataset:
        try:
            return self._entries[name]
        except KeyError:
            raise RegistryError(
                f"unknown dataset '{name}'; registered: {sorted(self._entries)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def names(self, bucket: Bucket | None = None) -> list[str]:
        return sorted(n for n, d in self._entries.items() if bucket is None or d.bucket is bucket)

    def bucket_root(self, bucket: Bucket) -> Path:
        if bucket is Bucket.TRAIN_OK:
            return self.train_ok_root
        if bucket is Bucket.TRAIN_PROTOTYPE:
            return self.train_prototype_root
        if bucket is Bucket.EVAL_ONLY:
            return self.eval_only_root
        raise LegalHoldViolation("LEGAL_HOLD datasets have no physical root")

    # -- path resolution --------------------------------------------------
    def _resolve_within(self, path: Path, root: Path, ds: Dataset) -> Path:
        """resolve() follows symlinks, so data/train_ok/x -> data/eval_only/y is caught."""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise PathEscapesBucket(
                f"dataset '{ds.name}': path {path} resolves to {resolved}, which is outside "
                f"{root}. A symlink out of the bucket tree is a licence violation, not a shortcut."
            ) from None
        return resolved

    def dataset_root(self, name: str, *, require_exists: bool = True) -> Path:
        ds = self[name]
        if ds.bucket is Bucket.LEGAL_HOLD:
            raise LegalHoldViolation(
                f"dataset '{name}' is LEGAL_HOLD: {ds.notes[:160]} "
                f"(open counsel items: {list(ds.licensing.counsel_questions)})"
            )
        if ds.status in (Status.PLANNED, Status.REQUESTED, Status.BLOCKED, Status.DROPPED):
            raise DatasetNotAvailable(
                f"dataset '{name}' has status '{ds.status.value}'; not on disk"
            )
        root = self._resolve_within(
            self.repo_root / ds.layout.root, self.bucket_root(ds.bucket), ds
        )
        if require_exists and not root.is_dir():
            raise DatasetNotAvailable(
                f"dataset '{name}' root {root} does not exist; "
                f"run scripts/data/download_dataset.py --name {name}"
            )
        return root

    def annotation_file(self, name: str, split: str, *, require_exists: bool = True) -> Path:
        """The split's annotation path, resolved and confined to the licence bucket.

        Named for the simple case, but layout.annotations is a directory as often as a
        file: the single-file formats point at one file, while the tree-shaped ones point
        at a directory holding one record file per video or per session. The existence
        check therefore accepts either, the way image_dir() accepts a directory, and
        raises only when nothing is on disk at that path. What a tree has to contain is
        the converter's business, not the registry's.
        """
        ds = self[name]
        root = self.dataset_root(name, require_exists=require_exists)
        rel = ds.layout.ann_rel(split)
        ann = self._resolve_within(root / rel, self.bucket_root(ds.bucket), ds)
        if require_exists and not ann.exists():
            raise DatasetNotAvailable(f"dataset '{name}' split '{split}': {ann} is missing")
        return ann

    def image_dir(self, name: str, split: str, *, require_exists: bool = True) -> Path:
        """The split's image directory, resolved and confined to the licence bucket.

        Mirrors annotation_file() exactly: dataset_root() confines layout.root, then the
        joined layout.images[split] path is re-resolved through _resolve_within, so a symlink
        or a '..' in layout.images that points out of the bucket raises PathEscapesBucket
        rather than being read. Callers must use this instead of joining
        dataset_root() / layout.image_rel() by hand: that inline join skipped the resolve()
        check and read imagery from outside the licence bucket, which POLICY.md, rule 6
        forbids.
        """
        ds = self[name]
        root = self.dataset_root(name, require_exists=require_exists)
        rel = ds.layout.image_rel(split)
        images = self._resolve_within(root / rel, self.bucket_root(ds.bucket), ds)
        if require_exists and not images.is_dir():
            raise DatasetNotAvailable(
                f"dataset '{name}' split '{split}': image dir {images} is missing"
            )
        return images

    # -- manifests --------------------------------------------------------
    def _manifest(
        self,
        kind: str,
        names: Sequence[str],
        split: str,
        allowed: set[Bucket],
        require_exists: bool,
        weight_class: str = "shippable",
    ) -> Manifest:
        entries: list[ManifestEntry] = []
        for name in sorted(set(names)):
            ds = self[name]
            if (
                ds.licensing.counsel_questions
                and not ds.licensing.counsel_cleared_utc
                and ds.bucket is not Bucket.LEGAL_HOLD
            ):
                # LEGAL_HOLD has its own stronger refusal below. Everything else with an
                # open counsel question is refused mechanically until
                # licensing.counsel_cleared_utc is set. The refusal is pinned by
                # tests/test_registry.py::test_open_counsel_questions_block_eval_manifest.
                raise CounselGateOpen(
                    f"'{name}' has open counsel questions "
                    f"{list(ds.licensing.counsel_questions)} and no counsel_cleared_utc. "
                    "No manifest may include it until counsel answers are recorded in "
                    "its configs/datasets/ entry."
                )
            if ds.bucket not in allowed:
                if ds.bucket is Bucket.LEGAL_HOLD:
                    raise LegalHoldViolation(
                        f"'{name}' is LEGAL_HOLD and cannot enter any manifest "
                        f"(counsel: {list(ds.licensing.counsel_questions)})"
                    )
                if ds.bucket is Bucket.TRAIN_PROTOTYPE:
                    raise EvalOnlyInTrainManifest(
                        f"'{name}' is TRAIN_PROTOTYPE and cannot enter a {kind} manifest for "
                        f"weight class '{weight_class}'. The recorded grant admits it to "
                        "prototype weights only: build_train_manifest(..., "
                        'weight_class="prototype"), and the resulting checkpoint is never '
                        "deployed commercially without counsel sign-off."
                    )
                raise EvalOnlyInTrainManifest(
                    f"'{name}' is {ds.bucket.value} and cannot enter a {kind} manifest. "
                    f"Licence: {ds.licensing.license}. "
                    "POLICY.md, rule 6: EVAL_ONLY datasets never enter a training manifest."
                )
            root = self.dataset_root(name, require_exists=require_exists)
            # Side effect kept deliberately: annotation_file() re-resolves the path
            # and raises PathEscapesBucket on a laundering symlink.
            self.annotation_file(name, split, require_exists=require_exists)
            entries.append(
                ManifestEntry(
                    name=ds.name,
                    bucket=ds.bucket.value,
                    split=split,
                    data_root=ds.layout.root,
                    ann_file=str(Path(ds.layout.root) / ds.layout.ann_rel(split)),
                    resolved_root=str(root),
                    license=ds.licensing.license,
                    ann_format=ds.ann_format,
                )
            )
        return Manifest(
            kind=kind,
            split=split,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            schema_version=SCHEMA_VERSION,
            entries=tuple(entries),
            weight_class=weight_class,
        )

    def build_train_manifest(
        self,
        names: Sequence[str],
        *,
        split: str = "train",
        require_exists: bool = True,
        weight_class: str = "shippable",
    ) -> Manifest:
        """Build a training manifest.

        weight_class="shippable" (the default) raises on any non-TRAIN_OK name.
        weight_class="prototype" additionally admits TRAIN_PROTOTYPE entries, and the
        manifest records the class so the checkpoint's provenance inherits it. EVAL_ONLY
        and LEGAL_HOLD are refused under both.
        """
        if weight_class not in WEIGHT_CLASSES:
            raise RegistryError(
                f"weight_class must be one of {WEIGHT_CLASSES}, got {weight_class!r}"
            )
        allowed = {Bucket.TRAIN_OK}
        if weight_class == "prototype":
            allowed.add(Bucket.TRAIN_PROTOTYPE)
        return self._manifest("train", names, split, allowed, require_exists, weight_class)

    def build_eval_manifest(
        self, names: Sequence[str], *, split: str = "val", require_exists: bool = True
    ) -> Manifest:
        """Eval may read trains, prototype-train and eval-only sets, never a LEGAL_HOLD one."""
        return self._manifest(
            "eval",
            names,
            split,
            {Bucket.TRAIN_OK, Bucket.TRAIN_PROTOTYPE, Bucket.EVAL_ONLY},
            require_exists,
        )


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    return Registry.from_dir(CONFIG_DIR, repo_root=REPO_ROOT)


def build_train_manifest(
    names: Sequence[str],
    *,
    split: str = "train",
    require_exists: bool = True,
    weight_class: str = "shippable",
) -> Manifest:
    return default_registry().build_train_manifest(
        names, split=split, require_exists=require_exists, weight_class=weight_class
    )


def build_eval_manifest(
    names: Sequence[str], *, split: str = "val", require_exists: bool = True
) -> Manifest:
    return default_registry().build_eval_manifest(names, split=split, require_exists=require_exists)
