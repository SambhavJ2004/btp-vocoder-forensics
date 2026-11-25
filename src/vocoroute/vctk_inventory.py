"""Index the original VCTK 0.92 ZIP and assign generator speakers deterministically."""

import hashlib
import re
import stat
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from zipfile import ZipFile, ZipInfo

SPEAKER = r"[ps][0-9]+"
ACQUISITION_REF = "sambhavjainsj/vocoroute-vctk092-acquisition"
ACQUISITION_VERSION = 2
AUDIO = re.compile(rf"wav48_silence_trimmed/({SPEAKER})/(\1_[0-9]+)_(mic[12])\.flac")
TEXT = re.compile(rf"txt/({SPEAKER})/(\1_[0-9]+)\.txt")
MAX_RECORDING_BYTES = 32 * 1024**2
MAX_TEXT_BYTES = 128 * 1024


@dataclass(frozen=True)
class Speaker:
    age: int
    gender: str
    accent: str
    region: str


@dataclass(frozen=True)
class Utterance:
    speaker: str
    audio: ZipInfo
    transcript_sha256: str
    normalized_text_sha256: str


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _speaker_metadata(raw: bytes) -> dict[str, Speaker]:
    lines = [line.strip() for line in _read_text(raw).splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("#")]
    if not lines or [field.upper() for field in lines[0].split()[:3]] != ["ID", "AGE", "GENDER"]:
        raise ValueError("unexpected VCTK speaker-info header")
    speakers = {}
    for line in lines[1:]:
        fields = line.split(maxsplit=4)
        if len(fields) < 4:
            raise ValueError(f"incomplete VCTK speaker row: {line!r}")
        sid, age, gender, accent = fields[:4]
        sid = f"p{sid}" if sid.isdecimal() else sid.casefold()
        if not re.fullmatch(SPEAKER, sid) or sid in speakers or not age.isdecimal():
            raise ValueError(f"invalid or duplicate VCTK speaker row: {line!r}")
        speakers[sid] = Speaker(int(age), gender, accent, fields[4] if len(fields) == 5 else "")
    if not speakers:
        raise ValueError("empty VCTK speaker metadata")
    return speakers


def _safe_members(zipped: ZipFile) -> dict[str, ZipInfo]:
    entries = {}
    seen = set()
    files = set()
    for info in zipped.infolist():
        name = info.orig_filename.removesuffix("/")
        parts = name.split("/")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (
            any(part in ("", ".", "..") for part in parts)
            or any(c in name for c in ("\\", ":"))
            or any(ord(c) < 32 for c in name)
            or mode not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG)
            or info.flag_bits & 1
            or name.casefold() in seen
        ):
            raise ValueError(f"unsafe or duplicate VCTK ZIP member: {info.orig_filename!r}")
        seen.add(name.casefold())
        if not info.is_dir():
            files.add(name.casefold())
            entries[name] = info
    if any(other in files for path in seen for other in _parents(path)):
        raise ValueError("VCTK ZIP contains a file used as a directory")
    return entries


def _parents(path: str) -> Iterator[str]:
    while "/" in path:
        path = path.rpartition("/")[0]
        yield path


def _bounded_read(zipped: ZipFile, entry: ZipInfo, maximum: int) -> bytes:
    if not 0 < entry.file_size <= maximum:
        raise ValueError(f"invalid VCTK member size: {entry.filename}")
    with zipped.open(entry) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != entry.file_size:
        raise ValueError(f"VCTK member size differs from ZIP index: {entry.filename}")
    return raw


@dataclass(frozen=True)
class Inventory:
    speakers: dict[str, Speaker]
    speaker_info: bytes
    utterances: dict[str, Utterance]
    report: dict


def inspect_archive(zipped: ZipFile, mic: str) -> Inventory:
    entries = _safe_members(zipped)
    metadata = [path for path in entries if path.split("/")[-1] == "speaker-info.txt"]
    if len(metadata) != 1 or mic not in ("mic1", "mic2"):
        raise ValueError("expected one speaker-info.txt and a single documented microphone")
    speaker_info = _bounded_read(zipped, entries[metadata[0]], MAX_TEXT_BYTES)
    speakers = _speaker_metadata(speaker_info)
    prefix = metadata[0].removesuffix("speaker-info.txt")
    audio = {"mic1": {}, "mic2": {}}
    transcripts = {}
    other = []
    for name, entry in entries.items():
        path = name.removeprefix(prefix)
        if name.startswith(prefix) and (match := AUDIO.fullmatch(path)):
            audio[match[3]][match[2]] = entry
        elif name.startswith(prefix) and (match := TEXT.fullmatch(path)):
            transcripts[match[2]] = entry
        elif path.startswith(("txt/", "wav48_silence_trimmed/")) or path.lower().endswith(
            (".flac", ".wav")
        ):
            other.append(name)

    if not audio[mic] or not transcripts:
        raise ValueError("VCTK ZIP contains no selected microphone audio or transcripts")
    issues = {
        "unrecognized_audio_or_text": sorted(other),
        "unlisted_speaker": [],
        "invalid_transcript": [],
        "invalid_recording_size": [],
    }
    utterances = {}
    for utterance in sorted(audio[mic].keys() & transcripts.keys()):
        sid = utterance.split("_")[0]
        recording = audio[mic][utterance]
        if sid not in speakers:
            issues["unlisted_speaker"].append(utterance)
            continue
        if not 0 < recording.file_size <= MAX_RECORDING_BYTES:
            issues["invalid_recording_size"].append(utterance)
            continue
        try:
            raw = _bounded_read(zipped, transcripts[utterance], MAX_TEXT_BYTES)
            normalized = unicodedata.normalize("NFKC", _read_text(raw)).casefold()
        except (UnicodeError, ValueError):
            issues["invalid_transcript"].append(utterance)
            continue
        tokens = " ".join(re.findall(r"[^\W_]+", normalized))
        if not tokens:
            issues["invalid_transcript"].append(utterance)
            continue
        utterances[utterance] = Utterance(sid, recording, sha256(raw), sha256(tokens.encode()))

    mic1, mic2 = audio["mic1"].keys(), audio["mic2"].keys()
    speaker_counts = {
        label: Counter(uid.split("_")[0] for uid in catalog)
        for label, catalog in (
            ("mic1", mic1),
            ("mic2", mic2),
            ("transcripts", transcripts),
            ("eligible_pairs", utterances),
        )
    }
    report = {
        "archive_root": prefix,
        "members": len(entries),
        "microphone": mic,
        "speaker_info_sha256": sha256(speaker_info),
        "selected_mic_without_transcript": sorted(audio[mic].keys() - transcripts.keys()),
        "transcript_without_recording": sorted(transcripts.keys() - mic1 - mic2),
        "transcript_without_selected_mic": sorted(transcripts.keys() - audio[mic].keys()),
        "paired_microphones": len(mic1 & mic2),
        "speakers_without_metadata": sorted(
            {uid.split("_")[0] for uid in mic1 | mic2 | transcripts.keys()} - speakers.keys()
        ),
        "issues": issues,
        "per_speaker": {
            sid: {label: counts[sid] for label, counts in speaker_counts.items()}
            for sid in sorted(speakers)
        },
    }
    return Inventory(speakers, speaker_info, utterances, report)


@dataclass(frozen=True)
class SplitProtocol:
    seed: int = 20261125
    microphone: str = "mic2"
    base_speakers: int = 60
    dev_speakers: int = 10
    eval_speakers: int = 10
    router_fit_speakers: int = 10
    router_diag_speakers: int = 5
    pilot_train_speakers: int = 20
    pilot_dev_speakers: int = 4
    expanded_per_speaker: int = 200
    pilot_per_speaker: int = 50
    eval_per_speaker: int = 100
    overfit_utterances: int = 8

    def __post_init__(self) -> None:
        counts = (
            self.base_speakers,
            self.dev_speakers,
            self.eval_speakers,
            self.router_fit_speakers,
            self.router_diag_speakers,
            self.pilot_train_speakers,
            self.pilot_dev_speakers,
            self.expanded_per_speaker,
            self.pilot_per_speaker,
            self.eval_per_speaker,
            self.overfit_utterances,
        )
        if (
            any(type(n) is not int or n <= 0 for n in counts)
            or self.microphone not in ("mic1", "mic2")
            or self.pilot_train_speakers > self.base_speakers
            or self.pilot_dev_speakers > self.dev_speakers
            or self.overfit_utterances > self.pilot_train_speakers
            or max(self.pilot_per_speaker, self.eval_per_speaker) > self.expanded_per_speaker
        ):
            raise ValueError("invalid nested corpus quotas or microphone")


def ranked_ids(seed: int, label: str, values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (sha256(f"{seed}:{label}:{value}".encode()), value))


def _stratified(speakers: dict[str, Speaker], seed: int, label: str) -> list[str]:
    strata = defaultdict(list)
    for sid, speaker in speakers.items():
        strata[speaker.gender.casefold(), speaker.accent.casefold()].append(sid)
    spread = []
    for bucket in strata.values():
        for index, sid in enumerate(ranked_ids(seed, label, bucket)):
            spread.append((Fraction(2 * index + 1, 2 * len(bucket)), sid))
    tie = {sid: i for i, sid in enumerate(ranked_ids(seed, label, speakers))}
    return [sid for _, sid in sorted(spread, key=lambda item: (item[0], tie[item[1]]))]


def allocate_speakers(inventory: Inventory, protocol: SplitProtocol) -> dict[str, list[str]]:
    """Spread accent/gender strata across role quotas without looking at model output."""
    counts = Counter(item.speaker for item in inventory.utterances.values())
    eligible = {
        sid: speaker
        for sid, speaker in inventory.speakers.items()
        if counts[sid] >= protocol.expanded_per_speaker
    }
    quotas = (
        ("eval", protocol.eval_speakers),
        ("dev", protocol.dev_speakers),
        ("router_fit", protocol.router_fit_speakers),
        ("router_diagnostics", protocol.router_diag_speakers),
        ("base_expert_fit", protocol.base_speakers),
    )
    required = sum(count for _, count in quotas)
    if len(eligible) < required:
        raise ValueError(f"need {required} structurally eligible speakers, found {len(eligible)}")
    chosen = {}
    for role, count in quotas:
        chosen[role] = _stratified(eligible, protocol.seed, role)[:count]
        for sid in chosen[role]:
            del eligible[sid]
    for role, parent, count in (
        ("pilot_train", "base_expert_fit", protocol.pilot_train_speakers),
        ("pilot_dev", "dev", protocol.pilot_dev_speakers),
    ):
        pool = {sid: inventory.speakers[sid] for sid in chosen[parent]}
        chosen[role] = _stratified(pool, protocol.seed, role)[:count]
    chosen["unused_eligible"] = sorted(eligible)
    chosen["below_eligibility_floor"] = sorted(
        set(inventory.speakers) - eligible.keys() - {sid for ids in chosen.values() for sid in ids}
    )
    return chosen
