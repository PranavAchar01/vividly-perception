"""Registry enforcement tests.

Two layers. The mechanism tests build throwaway entries under `tmp_path` and prove what the
loader and the manifest builder refuse; they keep working whichever datasets happen to be
registered. The shipped-state tests read `configs/datasets/` and pin what the registry
currently accepts, so a later edit cannot quietly move a bucket, drop a grant or open a gate.

Two rules run through the whole file and must survive any rewrite of it: an EVAL_ONLY entry
can never build a training manifest, and a TRAIN_PROTOTYPE entry that lists counsel questions
must carry a clearance date.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from vividly.data import registry as R


# --------------------------------------------------------------- helpers
def _entry(**over) -> str:
    base = {
        "name": "toy",
        "bucket": "TRAIN_OK",
        "status": "available",
        "root": "data/train_ok/toy",
        "images": "{train: images, val: images}",
        "commercial_use": "permitted",
        "counsel": "[]",
        "conflict": "null",
        "sim_events": "false",
        "human_required": "false",
        "cleared": "null",
        "grant": "null",
        "event_grant": "null",
        "attribution": "false",
        "annotations": "{train: annotations/train.json, val: annotations/train.json}",
    }
    base.update(over)
    return textwrap.dedent(f"""\
        schema_version: 1
        name: {base["name"]}
        display_name: toy
        task: [detection]
        bucket: {base["bucket"]}
        status: {base["status"]}
        licensing:
          license: toy licence
          license_url: https://example.invalid
          license_verbatim: toy
          license_conflict_note: {base["conflict"]}
          commercial_use: {base["commercial_use"]}
          attribution_required: {base["attribution"]}
          redistribute: false
          counsel_questions: {base["counsel"]}
          counsel_cleared_utc: {base["cleared"]}
          prototype_grant: {base["grant"]}
          event_grant: {base["event_grant"]}
        provenance:
          collector: test
          imagery_source: test
          contains_real_people: false
          simulated_medical_events: {base["sim_events"]}
          citation: test
        acquisition:
          method: http
          human_required: {base["human_required"]}
          urls: []
          human_steps: []
        layout:
          root: {base["root"]}
          images: {base["images"]}
          annotations: {base["annotations"]}
        ann_format: coco_keypoints
        converter: null
        files: []
        hashes_unavailable_reason: built by the test
        counts: {{}}
        notes: toy
    """)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Minimal repo: configs/datasets + both bucket trees, each with a real ann file."""
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    for bucket, name in (
        ("train_ok", "toy"),
        ("train_prototype", "proto"),
        ("eval_only", "restricted"),
    ):
        d = tmp_path / "data" / bucket / name
        (d / "annotations").mkdir(parents=True)
        (d / "images").mkdir()
        (d / "annotations" / "train.json").write_text("{}")
    return tmp_path


def _write(repo: Path, filename: str, text: str) -> None:
    (repo / "configs" / "datasets" / filename).write_text(text)


def _reg(repo: Path) -> R.Registry:
    return R.Registry.from_dir(repo / "configs" / "datasets", repo_root=repo)


# --------------------------------------------------------------- the bucket lock
def test_eval_only_entry_cannot_build_a_train_manifest(repo: Path) -> None:
    """The first rule of the registry: evaluation-only data never enters a train manifest."""
    _write(repo, "toy.yaml", _entry())
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
        ),
    )
    reg = _reg(repo)
    with pytest.raises(R.EvalOnlyInTrainManifest, match="cannot enter a train manifest"):
        reg.build_train_manifest(["restricted"])
    # ...and it is still refused when hidden among legitimate names.
    with pytest.raises(R.EvalOnlyInTrainManifest):
        reg.build_train_manifest(["toy", "restricted"])


def test_train_ok_entry_builds_a_train_manifest(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry())
    m = _reg(repo).build_train_manifest(["toy"])
    assert [e.name for e in m.entries] == ["toy"]
    assert m.entries[0].ann_file == "data/train_ok/toy/annotations/train.json"
    assert len(m.content_sha256) == 64


def test_eval_manifest_accepts_both_buckets(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry())
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
        ),
    )
    m = _reg(repo).build_eval_manifest(["toy", "restricted"])
    assert {e.bucket for e in m.entries} == {"TRAIN_OK", "EVAL_ONLY"}


# --------------------------------------------------------------- the symlink escape
def test_symlink_out_of_train_ok_is_rejected(repo: Path) -> None:
    """A TRAIN_OK entry whose root string is legal but whose inode is in eval_only.

    Load-time validation, a string prefix check, passes here on purpose: this test exists to
    prove the resolve()-based check at manifest-build time is what actually stops the leak.
    """
    (repo / "data" / "train_ok" / "laundered").symlink_to(
        repo / "data" / "eval_only" / "restricted", target_is_directory=True
    )
    _write(repo, "toy.yaml", _entry(root="data/train_ok/laundered"))
    reg = _reg(repo)  # validation passes: string is under train_ok
    with pytest.raises(R.PathEscapesBucket, match="outside"):
        reg.dataset_root("toy")
    with pytest.raises(R.PathEscapesBucket):
        reg.build_train_manifest(["toy"])


def test_symlinked_ann_file_out_of_train_ok_is_rejected(repo: Path) -> None:
    """Same escape one level down: legitimate root, laundered annotation file."""
    src = repo / "data" / "eval_only" / "restricted" / "annotations" / "train.json"
    (repo / "data" / "train_ok" / "toy" / "annotations" / "train.json").unlink()
    (repo / "data" / "train_ok" / "toy" / "annotations" / "train.json").symlink_to(src)
    _write(repo, "toy.yaml", _entry())
    with pytest.raises(R.PathEscapesBucket):
        _reg(repo).build_train_manifest(["toy"])


def test_symlinked_images_dir_out_of_train_ok_is_rejected(repo: Path) -> None:
    """The images directory is the third escape route. A TRAIN_OK entry whose layout.images
    points through a symlink into eval_only passes load-time validation, because the string is
    a plain relative name, and must be caught by image_dir()'s resolve() gate so that no
    crop extractor and no eval runner ever reads pixels out of the bucket."""
    src = repo / "data" / "eval_only" / "restricted" / "images"
    (repo / "data" / "train_ok" / "toy" / "images").rmdir()
    (repo / "data" / "train_ok" / "toy" / "images").symlink_to(src, target_is_directory=True)
    _write(repo, "toy.yaml", _entry())
    reg = _reg(repo)  # load passes: 'images' is a plain relative name
    with pytest.raises(R.PathEscapesBucket, match="outside"):
        reg.image_dir("toy", "train")


def test_dotdot_images_path_out_of_train_ok_is_rejected_at_load(repo: Path) -> None:
    """The load-time half: a '..' in layout.images escapes the bucket lexically, with no
    symlink required, so it is refused when the registry loads, under the same prefix
    invariant layout.root has always been held to."""
    _write(
        repo,
        "toy.yaml",
        _entry(images="{train: ../../eval_only/restricted/images, val: images}"),
    )
    with pytest.raises(R.RegistryValidationError, match=r"layout\.images"):
        _reg(repo)


# --------------------------------------------------------------- annotation path shape
def test_annotation_path_may_be_a_directory(repo: Path) -> None:
    """layout.annotations is a directory for most shipped entries rather than a file:
    le2i_fdd, caucafall, gmdcsa24 and ur_fall point at a directory of per-video record
    files, ajile12 at sessions, physicalai_smartspaces at the split directory and
    wu_sahzu_emu_trainsplit at the dataset root itself. The existence check accepts a
    directory the way image_dir() does, so none of them is diagnosed as missing bytes."""
    _write(repo, "toy.yaml", _entry(annotations="{train: annotations, val: annotations}"))
    assert _reg(repo).annotation_file("toy", "train").is_dir()


def test_annotation_path_that_is_not_on_disk_is_still_missing(repo: Path) -> None:
    """The other half of the same check: accepting a tree must not accept an absent path."""
    _write(repo, "toy.yaml", _entry(annotations="{train: absent, val: absent}"))
    with pytest.raises(R.DatasetNotAvailable, match="is missing"):
        _reg(repo).annotation_file("toy", "train")


# --------------------------------------------------------------- load-time invariants
@pytest.mark.parametrize("commercial_use", ["prohibited", "conflicting", "unreviewed"])
def test_train_ok_requires_permitted_commercial_use(repo: Path, commercial_use: str) -> None:
    conflict = "a note" if commercial_use == "conflicting" else "null"
    _write(repo, "toy.yaml", _entry(commercial_use=commercial_use, conflict=conflict))
    with pytest.raises(R.RegistryValidationError, match="commercial_use == 'permitted'"):
        _reg(repo)


def test_train_ok_requires_no_open_counsel_questions(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry(counsel="[CQ-9]"))
    with pytest.raises(R.RegistryValidationError, match="counsel"):
        _reg(repo)


def test_conflicting_licence_requires_a_note(repo: Path) -> None:
    _write(
        repo,
        "restricted.yaml",
        _entry(
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            name="restricted",
            commercial_use="conflicting",
        ),
    )
    with pytest.raises(R.RegistryValidationError, match="license_conflict_note"):
        _reg(repo)


def test_bucket_must_match_physical_tree(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry(bucket="EVAL_ONLY", commercial_use="prohibited"))
    with pytest.raises(R.RegistryValidationError, match=r"requires layout\.root under"):
        _reg(repo)


_TEST_ONLY_GRANT = (
    "{decision: interim-event-models, decided_utc: '2026-08-26', decided_by: owner, "
    "role: test_only}"
)
_TRAIN_INTERIM_GRANT = (
    "{decision: interim-event-models, decided_utc: '2026-08-26', decided_by: owner, "
    "role: train_interim}"
)


def test_simulated_medical_events_banned_without_event_grant(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry(sim_events="true"))
    with pytest.raises(R.RegistryValidationError, match=r"requires licensing\.event_grant"):
        _reg(repo)


def test_event_grant_test_only_admits_eval_only_and_nothing_else(repo: Path) -> None:
    # wrong bucket: test_only into TRAIN_OK is refused
    _write(repo, "toy.yaml", _entry(sim_events="true", event_grant=_TEST_ONLY_GRANT))
    with pytest.raises(R.RegistryValidationError, match="requires bucket EVAL_ONLY"):
        _reg(repo)
    # right bucket loads, and stays untrainable
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            sim_events="true",
            event_grant=_TEST_ONLY_GRANT,
        ),
    )
    (repo / "configs" / "datasets" / "toy.yaml").unlink()
    ds = _reg(repo)["restricted"]
    assert ds.licensing.event_grant is not None
    assert ds.licensing.event_grant.role == "test_only"
    assert not ds.prototype_trainable


def test_event_grant_train_interim_requires_train_prototype(repo: Path) -> None:
    prototype_grant = (
        "{decision: fall-prototype-sets, decided_utc: '2026-08-22', decided_by: owner, "
        "attribution: 'Toy, cite me', ship_policy: 'prototype weights only'}"
    )
    # wrong bucket refused
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            sim_events="true",
            event_grant=_TRAIN_INTERIM_GRANT,
        ),
    )
    with pytest.raises(R.RegistryValidationError, match="requires bucket TRAIN_PROTOTYPE"):
        _reg(repo)
    (repo / "configs" / "datasets" / "restricted.yaml").unlink()
    # right bucket plus the full prototype machinery loads and is prototype-trainable
    _write(
        repo,
        "proto.yaml",
        _entry(
            name="proto",
            bucket="TRAIN_PROTOTYPE",
            root="data/train_prototype/proto",
            commercial_use="conflicting",
            conflict="'non-commercial scope against internal prototype training, question open'",
            counsel="[CQ-22]",
            cleared="'2026-08-26'",
            sim_events="true",
            grant=prototype_grant,
            event_grant=_TRAIN_INTERIM_GRANT,
            attribution="true",
        ),
    )
    ds = _reg(repo)["proto"]
    assert ds.licensing.event_grant is not None
    assert ds.licensing.event_grant.role == "train_interim"
    assert ds.prototype_trainable and not ds.trainable


def test_event_grant_without_sim_events_refused(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry(event_grant=_TEST_ONLY_GRANT))
    with pytest.raises(R.RegistryValidationError, match="without simulated_medical_events"):
        _reg(repo)


def test_event_grant_bad_role_refused(repo: Path) -> None:
    bad = (
        "{decision: interim-event-models, decided_utc: '2026-08-26', decided_by: owner, role: both}"
    )
    _write(repo, "toy.yaml", _entry(sim_events="true", event_grant=bad))
    with pytest.raises(R.RegistryValidationError, match=r"event_grant\.role"):
        _reg(repo)


# --------------------------------------------------------------- grant decision names
#: A grant decision is a lowercase slug that says what was granted, so the value is readable
#: by anyone auditing the registry rather than an opaque serial number that has to be looked
#: up elsewhere. The shape is enforced at load time on both grants.
_BAD_DECISION_NAMES = [
    "REF9",  # an opaque serial number, the form this schema exists to refuse
    "Fall-Prototype-Sets",  # upper case
    "fall prototype sets",  # spaces
    "fall_prototype_sets",  # underscores
    "-fall",  # leading hyphen
    "ab",  # shorter than three characters
    "",  # empty
]


@pytest.mark.parametrize("bad", _BAD_DECISION_NAMES)
def test_event_grant_decision_must_be_a_lowercase_slug(repo: Path, bad: str) -> None:
    grant = _TEST_ONLY_GRANT.replace("interim-event-models", bad)
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            sim_events="true",
            event_grant=grant,
        ),
    )
    with pytest.raises(R.RegistryValidationError, match=r"event_grant\.decision"):
        _reg(repo)


def test_every_shipped_grant_decision_is_a_slug() -> None:
    """The shipped registry carries grant names, not serial numbers, in both grant fields."""
    reg = R.default_registry()
    for name in reg.names():
        lic = reg[name].licensing
        for field, grant in (
            ("prototype_grant", lic.prototype_grant),
            ("event_grant", lic.event_grant),
        ):
            if grant is None:
                continue
            assert R._DECISION_SLUG.match(grant.decision), (
                f"{name}: {field}.decision is {grant.decision!r}, which is not a lowercase "
                "slug naming what was granted"
            )


def test_legal_hold_has_no_root_and_cannot_be_used(repo: Path) -> None:
    """No shipped entry sits in LEGAL_HOLD today, so the bucket is pinned on a scratch entry:
    the refusal has to keep working for the next entry that lands in it."""
    _write(
        repo,
        "held.yaml",
        _entry(
            name="held",
            bucket="LEGAL_HOLD",
            root="null",
            status="blocked",
            commercial_use="prohibited",
        ),
    )
    reg = _reg(repo)
    assert reg["held"].layout.root is None
    with pytest.raises(R.LegalHoldViolation):
        reg.dataset_root("held")
    with pytest.raises(R.LegalHoldViolation):
        reg.build_eval_manifest(["held"])


def test_derived_train_ok_cannot_have_a_restricted_parent(repo: Path) -> None:
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
        ),
    )
    derived = _entry(name="derived", root="data/train_ok/toy") + "derived_from: [restricted]\n"
    _write(repo, "derived.yaml", derived)
    with pytest.raises(R.RegistryValidationError, match="cannot launder a licence"):
        _reg(repo)


def test_manifest_hash_is_order_independent_and_time_independent(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry())
    _write(repo, "toy2.yaml", _entry(name="toy2", root="data/train_ok/toy"))
    reg = _reg(repo)
    a = reg.build_train_manifest(["toy", "toy2"])
    b = reg.build_train_manifest(["toy2", "toy"])
    assert a.content_sha256 == b.content_sha256


def test_unknown_dataset_names_the_alternatives(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry())
    with pytest.raises(R.RegistryError, match="registered: \\['toy'\\]"):
        _reg(repo)["nope"]


# --------------------------------------------------------------- the shipped entries
#: The registry as it ships: twelve loadable entries plus the schema template, which
#: from_dir() skips because its name begins with an underscore. Named, not derived, so that
#: adding or removing an entry is a deliberate edit to this list and not a silent drift.
SHIPPED_ENTRIES = [
    "ajile12",
    "caucafall",
    "coco",
    "gmdcsa24",
    "le2i_fdd",
    "mevid",
    "multicam_fall",
    "mvor",
    "physicalai_smartspaces",
    "ur_fall",
    "wu_sahzu_emu",
    "wu_sahzu_emu_trainsplit",
]


def test_shipped_registry_loads() -> None:
    assert sorted(R.default_registry().names()) == SHIPPED_ENTRIES


def test_shipped_registry_covers_every_file_in_the_config_directory() -> None:
    """The loader skips underscore-prefixed files, which is how the template stays out of the
    registry. Every other YAML in the directory has to be a loaded entry, so a file that fails
    to parse cannot hide behind a rename."""
    on_disk = sorted(p.stem for p in R.CONFIG_DIR.glob("*.yaml"))
    assert on_disk == ["_template", *SHIPPED_ENTRIES]


#: Named, not derived, on purpose: this list is the regression lock. Deriving it from the
#: registry would make it agree with whatever a future edit does to the buckets, which is
#: the failure it exists to catch. `test_the_clearances_stay_inside_the_train_lock` checks
#: that every counsel-cleared entry is still inside it. "Never trainable" means never
#: trainable for shippable weights: the prototype grants admit six of these to prototype
#: training (PROTOTYPE_TRAINABLE below), and the default train manifest still refuses all of
#: them.
NEVER_TRAINABLE = [
    "le2i_fdd",  # prototype event training only; the shippable manifest still refuses it
    "ur_fall",  # test only: never trains in any weight class
    "multicam_fall",  # test only: never trains in any weight class
    "wu_sahzu_emu",  # test only: real clinical footage, never trains via this entry
    "wu_sahzu_emu_trainsplit",  # prototype event training only
    "gmdcsa24",  # prototype event training only
    "caucafall",  # prototype event training only
    "ajile12",  # prototype negative-diet source only
    "mevid",  # prototype re-identification corpus only
    "mvor",  # exam only (CQ-24): never trains in any weight class
]


@pytest.mark.parametrize("name", NEVER_TRAINABLE)
def test_shipped_restricted_sets_are_never_trainable(name: str) -> None:
    """Regression lock: nobody re-promotes any of these into the shippable class by accident."""
    reg = R.default_registry()
    assert reg[name].bucket is not R.Bucket.TRAIN_OK
    # Every entry in this list is counsel-cleared or carries no question, so the refusal that
    # fires is the bucket lock, which is the property this test exists to prove. A
    # CounselGateOpen here means an entry lost its clearance date.
    with pytest.raises((R.EvalOnlyInTrainManifest, R.LegalHoldViolation)):
        reg.build_train_manifest([name], require_exists=False)


def test_shipped_train_ok_is_exactly_the_two_permitted_sets() -> None:
    """Only a dataset whose licence permits commercial use reaches the shippable class."""
    reg = R.default_registry()
    assert reg.names(R.Bucket.TRAIN_OK) == ["coco", "physicalai_smartspaces"]
    for name in reg.names(R.Bucket.TRAIN_OK):
        lic = reg[name].licensing
        assert lic.commercial_use is R.CommercialUse.PERMITTED
        assert not lic.counsel_questions
        assert lic.prototype_grant is None and lic.event_grant is None


def test_shipped_train_ok_manifest_builds() -> None:
    m = R.build_train_manifest(["coco"], require_exists=False)
    assert m.entries[0].ann_file == "data/train_ok/coco/annotations/person_keypoints_train2017.json"


@pytest.mark.skipif(
    not (R.REPO_ROOT / "data" / "train_ok" / "coco" / "train2017").is_dir(),
    reason="COCO not present on this machine",
)
def test_shipped_coco_exists_on_disk() -> None:
    m = R.build_train_manifest(["coco"])  # require_exists=True
    assert Path(m.entries[0].resolved_root).is_dir()


def test_every_conflicting_shipped_entry_records_the_conflict() -> None:
    """A conflicting licence with no note is a licence nobody wrote down. The mechanism is
    pinned on a scratch entry by test_conflicting_licence_requires_a_note; this is the
    shipped state, and it is what stops a note being emptied during an edit."""
    reg = R.default_registry()
    conflicting = [
        n for n in reg.names() if reg[n].licensing.commercial_use is R.CommercialUse.CONFLICTING
    ]
    assert conflicting, "no conflicting entry to check, has the filter drifted?"
    for name in conflicting:
        note = reg[name].licensing.license_conflict_note
        assert note and note.strip(), f"{name}: commercial_use conflicting with an empty note"


# --------------------------------------------------------------- counsel gate
def test_open_counsel_questions_block_eval_manifest(repo: Path) -> None:
    """The counsel gate is mechanical. An EVAL_ONLY dataset with open counsel questions
    cannot enter even an eval manifest until it is cleared."""
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
            counsel="[CQ-NC-EVAL]",
        ),
    )
    with pytest.raises(R.CounselGateOpen, match="CQ-NC-EVAL"):
        _reg(repo).build_eval_manifest(["restricted"])


def test_every_counsel_question_id_is_documented() -> None:
    """`counsel_questions` values are identifiers into the counsel register, and for a while
    that register did not exist, so every identifier was a dangling reference. This test makes
    the register the target it claims to be: an identifier nobody documented fails the suite.
    """
    doc = (R.REPO_ROOT / "docs" / "COUNSEL_QUESTIONS.md").read_text(encoding="utf-8")
    headings = [ln for ln in doc.splitlines() if ln.startswith("#")]
    reg = R.default_registry()
    missing = sorted(
        (name, cq)
        for name in reg.names()
        for cq in reg[name].licensing.counsel_questions
        if not any(cq in h for h in headings)
    )
    assert not missing, (
        f"counsel question ids with no section in docs/COUNSEL_QUESTIONS.md: {missing}. "
        "Add a heading containing the exact id; the register is what goes to counsel."
    )


# --------------------------------------------------------------- the re-identification sets
#: The two entries carrying the re-identification task. MEVID is the Phase A training corpus
#: and is prototype-only because its annotation layer ships with no licence; SmartSpaces is a
#: permissively licensed multi-camera corpus in the shippable class. Pinned as a set so a
#: later edit cannot add a third re-identification source without touching this list.
REID_SETS = ("mevid", "physicalai_smartspaces")


def test_reid_task_entries_are_exactly_the_pinned_pair() -> None:
    reg = R.default_registry()
    assert tuple(sorted(n for n in reg.names() if "reid" in reg[n].task)) == REID_SETS


def test_mevid_is_prototype_only_and_carries_its_grant() -> None:
    """The re-identification corpus can never be anything but prototype: the MEVA source video
    is CC BY 4.0, and the annotation layer that actually trains the embedder carries no licence
    at all, which the entry records as a conflict rather than resolving."""
    reg = R.default_registry()
    ds = reg["mevid"]
    assert ds.bucket is R.Bucket.TRAIN_PROTOTYPE
    assert ds.layout.root == "data/train_prototype/mevid"
    assert ds.licensing.commercial_use is R.CommercialUse.CONFLICTING
    assert ds.licensing.license_conflict_note
    grant = ds.licensing.prototype_grant
    assert grant is not None and grant.decision == "mevid-phase-a"
    assert "prototype" in grant.ship_policy.lower()
    assert ds.prototype_trainable and not ds.trainable
    with pytest.raises(R.EvalOnlyInTrainManifest):
        reg.build_train_manifest(["mevid"], require_exists=False)


def test_smartspaces_is_train_ok_and_needs_no_grant() -> None:
    ds = R.default_registry()["physicalai_smartspaces"]
    assert ds.bucket is R.Bucket.TRAIN_OK
    assert ds.licensing.commercial_use is R.CommercialUse.PERMITTED
    assert ds.licensing.prototype_grant is None
    assert ds.trainable


# ---------------------------------------------- the standing mechanical clearance
# The mechanical counsel refusal was lifted on a set of entries for prototype-stage use, and
# no counsel question was answered by it. The mechanism itself is pinned on scratch entries by
# test_open_counsel_questions_block_eval_manifest and
# test_cleared_counsel_questions_pass_eval_manifest, which is where it belongs: those two keep
# working whichever shipped entry happens to be gated. Everything below describes the shipped
# state and is derived from the registry, so the next clearance or revocation moves one YAML
# and no test.
def _counsel_cleared() -> tuple[str, ...]:
    reg = R.default_registry()
    return tuple(n for n in reg.names() if reg[n].licensing.counsel_cleared_utc)


COUNSEL_CLEARED = _counsel_cleared()


def test_a_clearance_never_empties_the_counsel_questions() -> None:
    """A cleared entry with no recorded questions would be a policy regression, not a tidy-up.

    The clearance covers the mechanical gate and leaves the questions open, so the clearance
    date and the question list have to coexist in every covered entry. An entry carrying
    `counsel_cleared_utc` and an empty `counsel_questions` reads, to anyone who finds it
    later, as counsel having answered. Nobody has answered anything.
    """
    reg = R.default_registry()
    assert COUNSEL_CLEARED, "a registry where no entry is cleared is a regression"
    for name in COUNSEL_CLEARED:
        ds = reg[name]
        assert ds.bucket in (R.Bucket.EVAL_ONLY, R.Bucket.TRAIN_PROTOTYPE), (
            f"'{name}' is cleared and is bucket {ds.bucket.value}. The clearance covers the "
            "gate, never a bucket: nothing it covers may become TRAIN_OK."
        )
        assert ds.licensing.counsel_questions, (
            f"'{name}' records counsel_cleared_utc but lists no counsel_questions. The "
            "clearance covers the mechanical gate, not the questions; deleting them rewrites "
            "the decision."
        )
        cleared = ds.licensing.counsel_cleared_utc
        assert isinstance(cleared, str) and re.match(r"^\d{4}-\d{2}-\d{2}", cleared), (
            f"'{name}' counsel_cleared_utc is {cleared!r}. The field is a legal record and "
            "the registry validates it to a quoted ISO 8601 UTC string; truthy junk like "
            "'TBD' must never open the gate."
        )


def test_any_future_uncleared_eval_only_entry_is_refused_by_the_eval_manifest() -> None:
    """Universal quantifier over the shipped registry, vacuous today, armed for the next entry.

    Every shipped EVAL_ONLY entry with open counsel questions and no clearance must be refused
    by build_eval_manifest. Today every such entry is cleared, so this loop asserts over an
    empty set; the day one lands with an open question and no clearance, this test is what
    proves the gate still refuses it without anyone remembering to write a test for it.
    """
    reg = R.default_registry()
    for name in reg.names():
        ds = reg[name]
        if ds.bucket is not R.Bucket.EVAL_ONLY:
            continue
        if not ds.licensing.counsel_questions or ds.licensing.counsel_cleared_utc:
            continue
        with pytest.raises(R.CounselGateOpen):
            reg.build_eval_manifest([name], require_exists=False)


@pytest.mark.parametrize("name", COUNSEL_CLEARED)
def test_a_cleared_entry_clears_the_counsel_gate_whatever_its_bytes_say(name: str) -> None:
    """After the clearance the eval manifest no longer refuses these entries on counsel grounds.

    Disk-aware, not status-aware: the YAML status is repository state and travels to every
    machine, while the bytes stay local, so continuous integration runs with status available
    or converted and an empty data/ tree. The only portable assertion is that whatever
    build_eval_manifest does, it is never a counsel refusal: a DatasetNotAvailable about
    missing bytes is the gate having opened, and a machine with the bytes must get a manifest.
    The cleared-and-present path is also pinned on a scratch entry by
    test_cleared_counsel_questions_pass_eval_manifest.
    """
    reg = R.default_registry()
    try:
        manifest = reg.build_eval_manifest([name], require_exists=True)
    except R.DatasetNotAvailable as exc:
        # No bytes on this machine: a still-planned entry, or a fresh clone, where the YAML
        # says available or converted but data/ is empty. The refusal is about disk state and
        # never about counsel, which is exactly the gate having opened: CounselGateOpen fires
        # before any root is resolved and would escape this except clause.
        assert "counsel" not in str(exc).lower()
    else:
        assert [e.name for e in manifest.entries] == [name]


def test_the_clearances_stay_inside_the_train_lock() -> None:
    """Bucket law is untouched by a clearance.

    The refusal itself is pinned by test_shipped_restricted_sets_are_never_trainable; this
    only checks that every cleared entry is inside that test's list, so opening the counsel
    gate can never quietly carry an entry out of the training-manifest lock.
    """
    assert set(COUNSEL_CLEARED) <= set(NEVER_TRAINABLE)


# --------------------------------------------------------------- attribution
def test_attribution_required_entries_are_credited() -> None:
    """`attribution_required` is parsed and used: it drives the credit list in README.md.
    Every live entry that requires attribution must carry a citation and be credited there.
    The test stays honest as the gated sets go `available`."""
    readme = (R.REPO_ROOT / "README.md").read_text(encoding="utf-8")
    reg = R.default_registry()
    checked: list[str] = []
    for name in reg.names():
        ds = reg[name]
        # Attribution is owed for data actually held. planned and requested are not on disk;
        # dropped and blocked never arrive.
        if not ds.licensing.attribution_required or ds.status not in (
            R.Status.AVAILABLE,
            R.Status.CONVERTED,
        ):
            continue
        checked.append(name)
        assert ds.provenance.citation, f"{name}: attribution_required with an empty citation"
        # Either the credited party (collector) or the citation's lead author has to appear
        # exactly. README.md credits COCO as "COCO Consortium", not as "Lin et al.".
        first_author = ds.provenance.citation.split(",")[0].split()[0]
        credits = {ds.provenance.collector, first_author}
        assert any(c and c in readme for c in credits), (
            f"{name}: attribution_required but README.md credits none of {sorted(credits)}"
        )
    assert checked, "no live attribution_required entry to check, has the filter drifted?"


# --------------------------------------------------------------- terms acceptance
_TERMS_BLOCK = (
    "  terms_acceptance:\n"
    "    accepted_by: owner\n"
    '    accepted_utc: "2026-08-07T00:00:00Z"\n'
    "    method: web form at https://example.invalid/request\n"
    "    evidence_path: null"
)


def _with_terms(text: str) -> str:
    return text.replace("counsel_cleared_utc: null", "counsel_cleared_utc: null\n" + _TERMS_BLOCK)


def test_human_required_available_entry_needs_a_terms_record(repo: Path) -> None:
    """A record that the terms were accepted on a given date is the artefact a diligence
    review asks for. Inert on today's tree, and it fires the moment a human-gated entry lands
    on disk."""
    _write(repo, "toy.yaml", _entry(human_required="true"))  # status: available
    with pytest.raises(R.RegistryValidationError, match="terms_acceptance"):
        _reg(repo)


@pytest.mark.parametrize("status", ["planned", "requested", "dropped", "blocked"])
def test_terms_record_not_required_before_acquisition(repo: Path, status: str) -> None:
    _write(repo, "toy.yaml", _entry(human_required="true", status=status))
    assert _reg(repo)["toy"].licensing.terms_acceptance is None


def test_terms_record_is_parsed_when_present(repo: Path) -> None:
    _write(repo, "toy.yaml", _with_terms(_entry(human_required="true")))
    ta = _reg(repo)["toy"].licensing.terms_acceptance
    assert ta is not None
    assert (ta.accepted_by, ta.accepted_utc, ta.evidence_path) == (
        "owner",
        "2026-08-07T00:00:00Z",
        None,
    )


def test_partial_terms_record_is_rejected(repo: Path) -> None:
    partial = "\n".join(_TERMS_BLOCK.splitlines()[:2])  # accepted_by only, no accepted_utc
    text = _entry(human_required="true").replace(
        "counsel_cleared_utc: null", "counsel_cleared_utc: null\n" + partial
    )
    _write(repo, "toy.yaml", text)
    with pytest.raises(R.RegistryValidationError, match="accepted_utc"):
        _reg(repo)


# --------------------------------------------------------------- the template
def test_registry_template_entry(tmp_path: Path) -> None:
    """configs/datasets/_template.yaml is documentation that from_dir() skips, so nothing
    caught it drifting out of the schema. Fill every <placeholder> and it must parse and
    validate like any real entry.
    """
    raw = (R.CONFIG_DIR / "_template.yaml").read_text(encoding="utf-8")
    filled = re.sub(r"<[^>]*>", "placeholder", raw)
    path = tmp_path / "placeholder.yaml"
    path.write_text(filled, encoding="utf-8")

    ds = R._parse(path)
    R._validate(ds, path.stem)

    assert ds.name == "placeholder"
    assert ds.bucket is R.Bucket.TRAIN_OK and ds.status is R.Status.PLANNED
    assert ds.layout.root == "data/train_ok/placeholder"
    assert ds.licensing.terms_acceptance is None  # documented as null; required later
    assert "_template" not in R.default_registry().names()


def test_cleared_counsel_questions_pass_eval_manifest(repo: Path) -> None:
    text = _entry(
        name="restricted",
        bucket="EVAL_ONLY",
        root="data/eval_only/restricted",
        commercial_use="prohibited",
        counsel="[CQ-NC-EVAL]",
    ).replace("counsel_cleared_utc: null", 'counsel_cleared_utc: "2026-08-07T00:00:00Z"')
    _write(repo, "restricted.yaml", text)
    m = _reg(repo).build_eval_manifest(["restricted"])
    assert [e.name for e in m.entries] == ["restricted"]


# --------------------------------------------------------------- the prototype weight class
# A prototype grant admits an entry to TRAIN_PROTOTYPE: it may train prototype weights, asked
# for by name with weight_class="prototype", its held-out numbers may be shown externally with
# attribution, and nothing about shipping changes. An event grant with role train_interim
# admits acted or clinical event footage to interim event-verifier training on top of the same
# machinery. The mechanism is pinned on scratch entries first, the shipped state second.
PROTOTYPE_TRAINABLE = [
    "ajile12",  # epilepsy-monitoring-context negative diet for the seizure lane, pose only
    "caucafall",  # interim fall set
    "gmdcsa24",  # interim fall set, positive-side angle and subject diversity
    "le2i_fdd",  # interim event-training grant, the first acted-fall set admitted
    "mevid",  # re-identification Phase A training corpus
    "wu_sahzu_emu_trainsplit",  # train-side epilepsy-monitoring patients, real events
]

_GRANT = (
    "{decision: fall-prototype-sets, decided_utc: '2026-08-22', decided_by: owner, "
    "attribution: 'Toy (Toy Lab), cite me', ship_policy: 'prototype weights only'}"
)


def _proto_entry(**over) -> str:
    base = {
        "name": "proto",
        "bucket": "TRAIN_PROTOTYPE",
        "root": "data/train_prototype/proto",
        "commercial_use": "conflicting",
        "conflict": "'download page grants, dataset page forbids'",
        "counsel": "[CQ-22-acted-fall-dataset-licenses]",
        "cleared": "'2026-08-08'",
        "grant": _GRANT,
        "attribution": "true",
    }
    base.update(over)
    return _entry(**base)


def test_train_prototype_entry_loads_and_builds_a_prototype_manifest(repo: Path) -> None:
    _write(repo, "proto.yaml", _proto_entry())
    reg = _reg(repo)
    ds = reg["proto"]
    assert ds.bucket is R.Bucket.TRAIN_PROTOTYPE
    assert ds.prototype_trainable and not ds.trainable
    assert ds.licensing.prototype_grant is not None
    assert ds.licensing.prototype_grant.decision == "fall-prototype-sets"
    m = reg.build_train_manifest(["proto"], weight_class="prototype")
    assert m.weight_class == "prototype"
    assert m.to_dict()["weight_class"] == "prototype"
    assert m.entries[0].bucket == "TRAIN_PROTOTYPE"
    assert m.entries[0].resolved_root == str(repo / "data" / "train_prototype" / "proto")


def test_default_train_manifest_refuses_train_prototype(repo: Path) -> None:
    """The shippable class is the default, and a prototype entry never enters it by accident."""
    _write(repo, "proto.yaml", _proto_entry())
    reg = _reg(repo)
    with pytest.raises(R.EvalOnlyInTrainManifest, match="weight class 'shippable'"):
        reg.build_train_manifest(["proto"])
    with pytest.raises(R.EvalOnlyInTrainManifest, match="prototype"):
        reg.build_train_manifest(["proto"], weight_class="shippable")
    # A manifest of shippable entries is unchanged by the new class.
    _write(repo, "toy.yaml", _entry())
    m = _reg(repo).build_train_manifest(["toy"])
    assert m.weight_class == "shippable"


def test_prototype_weight_class_still_refuses_eval_only_and_legal_hold(repo: Path) -> None:
    """The prototype class widened the train manifest by exactly one bucket. The bucket lock
    is untouched."""
    _write(repo, "proto.yaml", _proto_entry())
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
        ),
    )
    _write(
        repo,
        "held.yaml",
        _entry(
            name="held",
            bucket="LEGAL_HOLD",
            root="null",
            status="blocked",
            commercial_use="prohibited",
        ),
    )
    reg = _reg(repo)
    with pytest.raises(R.EvalOnlyInTrainManifest, match="EVAL_ONLY"):
        reg.build_train_manifest(["proto", "restricted"], weight_class="prototype")
    with pytest.raises(R.LegalHoldViolation):
        reg.build_train_manifest(["proto", "held"], weight_class="prototype")


def test_unknown_weight_class_is_refused(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry())
    with pytest.raises(R.RegistryError, match="weight_class"):
        _reg(repo).build_train_manifest(["toy"], weight_class="shipped")


def test_eval_manifest_admits_train_prototype(repo: Path) -> None:
    _write(repo, "proto.yaml", _proto_entry())
    m = _reg(repo).build_eval_manifest(["proto"], split="val")
    assert [e.bucket for e in m.entries] == ["TRAIN_PROTOTYPE"]
    assert m.weight_class == "shippable"  # an eval manifest trains nothing


def test_train_prototype_requires_its_grant(repo: Path) -> None:
    _write(repo, "proto.yaml", _proto_entry(grant="null"))
    with pytest.raises(R.RegistryValidationError, match="prototype_grant"):
        _reg(repo)


@pytest.mark.parametrize("bad", _BAD_DECISION_NAMES)
def test_prototype_grant_decision_must_be_a_lowercase_slug(repo: Path, bad: str) -> None:
    grant = _GRANT.replace("fall-prototype-sets", bad)
    _write(repo, "proto.yaml", _proto_entry(grant=grant))
    with pytest.raises(R.RegistryValidationError, match=r"prototype_grant\.decision"):
        _reg(repo)


def test_train_prototype_grant_fields_are_validated(repo: Path) -> None:
    bad_date = _GRANT.replace("'2026-08-22'", "soon")
    _write(repo, "proto.yaml", _proto_entry(grant=bad_date))
    with pytest.raises(R.RegistryValidationError, match="decided_utc"):
        _reg(repo)
    empty_attr = _GRANT.replace("'Toy (Toy Lab), cite me'", "'  '")
    _write(repo, "proto.yaml", _proto_entry(grant=empty_attr))
    with pytest.raises(R.RegistryValidationError, match="attribution must be non-empty"):
        _reg(repo)


def test_train_prototype_requires_a_commercial_grant_in_some_primary_source(repo: Path) -> None:
    """A flatly prohibited licence cannot be prototype-trained, whatever anyone decides."""
    _write(repo, "proto.yaml", _proto_entry(commercial_use="prohibited", conflict="null"))
    with pytest.raises(R.RegistryValidationError, match="'permitted' or 'conflicting'"):
        _reg(repo)


def test_train_prototype_requires_attribution(repo: Path) -> None:
    _write(repo, "proto.yaml", _proto_entry(attribution="false"))
    with pytest.raises(R.RegistryValidationError, match="attribution_required: true"):
        _reg(repo)


def test_train_prototype_with_open_questions_requires_the_clearance_date(repo: Path) -> None:
    """The second rule that must survive any rewrite of this file: a TRAIN_PROTOTYPE entry
    that lists counsel questions has to carry counsel_cleared_utc."""
    _write(repo, "proto.yaml", _proto_entry(cleared="null"))
    with pytest.raises(R.RegistryValidationError, match="counsel_cleared_utc"):
        _reg(repo)


def test_train_prototype_must_live_under_its_own_tree(repo: Path) -> None:
    _write(repo, "proto.yaml", _proto_entry(root="data/train_ok/toy"))
    with pytest.raises(R.RegistryValidationError, match=r"requires layout\.root under"):
        _reg(repo)


def test_train_ok_carries_no_grant(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry(grant=_GRANT))
    with pytest.raises(R.RegistryValidationError, match="carries no prototype_grant"):
        _reg(repo)


def test_train_ok_cannot_derive_from_train_prototype(repo: Path) -> None:
    """Rendering or transforming prototype imagery does not launder it into a shippable corpus."""
    _write(repo, "proto.yaml", _proto_entry())
    derived = _entry(name="derived", root="data/train_ok/toy") + "derived_from: [proto]\n"
    _write(repo, "derived.yaml", derived)
    with pytest.raises(R.RegistryValidationError, match="cannot launder a licence"):
        _reg(repo)


def test_train_prototype_cannot_derive_from_eval_only(repo: Path) -> None:
    _write(
        repo,
        "restricted.yaml",
        _entry(
            name="restricted",
            bucket="EVAL_ONLY",
            root="data/eval_only/restricted",
            commercial_use="prohibited",
        ),
    )
    _write(repo, "proto.yaml", _proto_entry() + "derived_from: [restricted]\n")
    with pytest.raises(R.RegistryValidationError, match="TRAIN_PROTOTYPE dataset derives from"):
        _reg(repo)


def test_train_prototype_may_derive_from_train_ok(repo: Path) -> None:
    _write(repo, "toy.yaml", _entry())
    _write(repo, "proto.yaml", _proto_entry() + "derived_from: [toy]\n")
    assert _reg(repo)["proto"].derived_from == ("toy",)


def test_symlink_from_train_ok_into_train_prototype_is_rejected(repo: Path) -> None:
    """The shippable tree may not reach the prototype tree through a symlink either."""
    _write(repo, "proto.yaml", _proto_entry())
    (repo / "data" / "train_ok" / "laundered").symlink_to(
        repo / "data" / "train_prototype" / "proto", target_is_directory=True
    )
    _write(repo, "toy.yaml", _entry(root="data/train_ok/laundered"))
    reg = _reg(repo)
    with pytest.raises(R.PathEscapesBucket):
        reg.build_train_manifest(["toy"])


# -- the shipped state
@pytest.mark.parametrize("name", PROTOTYPE_TRAINABLE)
def test_shipped_prototype_entries_are_prototype_trainable_and_nothing_more(name: str) -> None:
    """What the prototype grant changed for each admitted set, pinned so it cannot drift.

    The bucket, the tree, the grant, any open question and its clearance date all have to
    coexist: a prototype entry that lost its grant reads as a relaxed TRAIN_OK, and one that
    lost its counsel question reads as counsel having answered. Nobody has answered anything.
    """
    reg = R.default_registry()
    ds = reg[name]
    assert ds.bucket is R.Bucket.TRAIN_PROTOTYPE
    assert ds.layout.root == f"data/train_prototype/{name}"
    grant = ds.licensing.prototype_grant
    assert grant is not None
    assert R._DECISION_SLUG.match(grant.decision)
    assert re.match(r"^\d{4}-\d{2}-\d{2}", grant.decided_utc)
    assert grant.attribution.strip()
    assert "prototype" in grant.ship_policy.lower()
    assert ds.licensing.attribution_required
    assert ds.licensing.commercial_use in (
        R.CommercialUse.PERMITTED,
        R.CommercialUse.CONFLICTING,
    )
    if ds.licensing.counsel_questions:
        assert ds.licensing.counsel_cleared_utc, (
            f"'{name}' lists counsel questions and no clearance date; the registry must "
            "refuse that combination on a prototype entry"
        )
    # Default (shippable) refuses; prototype admits; eval admits. The weight-class check
    # runs before any root is resolved, so the shippable refusal is portable even for an
    # entry whose bytes have not landed; the two admitting calls can still stop on disk
    # state, and a DatasetNotAvailable there is the class gate having opened.
    with pytest.raises(R.EvalOnlyInTrainManifest):
        reg.build_train_manifest([name], split="train", require_exists=False)
    eval_split = "val" if any(k == "val" for k, _ in ds.layout.annotations) else "test"
    try:
        m = reg.build_train_manifest(
            [name], split="train", require_exists=False, weight_class="prototype"
        )
    except R.DatasetNotAvailable as exc:
        assert ds.status is not R.Status.AVAILABLE
        assert "weight class" not in str(exc)
    else:
        assert m.weight_class == "prototype"
    try:
        reg.build_eval_manifest([name], split=eval_split, require_exists=False)
    except R.DatasetNotAvailable:
        assert ds.status is not R.Status.AVAILABLE


def test_prototype_bucket_covers_exactly_the_granted_sets() -> None:
    """Derived from the registry so the next grant or revocation moves one YAML and no test,
    but the current set is also pinned, because widening it is an owner decision."""
    reg = R.default_registry()
    assert reg.names(R.Bucket.TRAIN_PROTOTYPE) == sorted(PROTOTYPE_TRAINABLE)
    for name in reg.names(R.Bucket.TRAIN_PROTOTYPE):
        assert reg[name].licensing.prototype_grant is not None


def test_shipped_le2i_entry_is_interim_event_training_and_nothing_more() -> None:
    """What the interim event grant changed for le2i_fdd, pinned so it cannot drift either
    way: the acted-event grant, its exclusive train_interim role, the prototype grant, the
    open licence question and the mechanical clearance must all coexist."""
    ds = R.default_registry()["le2i_fdd"]
    assert ds.bucket is R.Bucket.TRAIN_PROTOTYPE
    assert ds.layout.root == "data/train_prototype/le2i_fdd"
    assert ds.provenance.simulated_medical_events
    egrant = ds.licensing.event_grant
    assert egrant is not None
    assert egrant.decision == "interim-event-models" and egrant.role == "train_interim"
    grant = ds.licensing.prototype_grant
    assert grant is not None
    assert grant.decision == "interim-event-models" and grant.decided_utc == "2026-08-26"
    assert "counsel" in grant.ship_policy.lower()
    assert "CQ-22-acted-fall-dataset-licenses" in ds.licensing.counsel_questions
    assert ds.licensing.counsel_cleared_utc
    assert ds.licensing.commercial_use is R.CommercialUse.CONFLICTING


def test_shipped_acted_event_test_sets_never_train() -> None:
    """ur_fall and multicam_fall carry test_only grants in EVAL_ONLY: the manifest layer must
    refuse them for training in every weight class."""
    reg = R.default_registry()
    for name in ("ur_fall", "multicam_fall"):
        ds = reg[name]
        assert ds.bucket is R.Bucket.EVAL_ONLY
        assert ds.provenance.simulated_medical_events
        assert ds.licensing.event_grant is not None
        assert ds.licensing.event_grant.role == "test_only"
        assert not ds.prototype_trainable and not ds.trainable


def test_shipped_clinical_footage_entries_carry_an_event_grant() -> None:
    """Real recorded medical events pass through the same gate as acted ones, and they carry
    more weight, not less. The clinical epilepsy-monitoring corpus is split into an exam-side
    entry that never trains and a train-side entry that trains prototype weights only."""
    reg = R.default_registry()
    clinical = [n for n in reg.names() if reg[n].provenance.real_medical_events]
    assert sorted(clinical) == ["ajile12", "wu_sahzu_emu", "wu_sahzu_emu_trainsplit"]
    for name in clinical:
        grant = reg[name].licensing.event_grant
        assert grant is not None, f"{name}: real medical events with no event_grant"
    assert reg["wu_sahzu_emu"].licensing.event_grant.role == "test_only"
    assert reg["wu_sahzu_emu_trainsplit"].licensing.event_grant.role == "train_interim"
    assert reg["wu_sahzu_emu"].bucket is R.Bucket.EVAL_ONLY
    assert reg["wu_sahzu_emu_trainsplit"].bucket is R.Bucket.TRAIN_PROTOTYPE
