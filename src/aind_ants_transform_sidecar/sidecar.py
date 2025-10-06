from __future__ import annotations

import hashlib
import json
import math
import sys
from array import array
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

# --------- Spatial hash config (for Domain) ---------
HASH_DIGITS_DEFAULT = 9
HASH_DIGEST_SIZE = 16
HASH_METHOD = f"LPS_bbox_spacing_shape:intQ{HASH_DIGITS_DEFAULT}:BE"

# --------- Domain spec (layout-invariant) ---------


class BBox(BaseModel):
    L: tuple[float, float]
    P: tuple[float, float]
    S: tuple[float, float]

    @model_validator(mode="after")
    def _check_monotonic(self) -> BBox:
        for axis in ("L", "P", "S"):
            lo, hi = getattr(self, axis)
            if lo > hi:
                raise ValueError(f"bbox[{axis}] min must be ≤ max")
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"bbox[{axis}] contains non-finite value")
        return self


class ContentSignature(BaseModel):
    method: str  # e.g., "canonical_LPS_bbox_spacing_shape"
    blake2b: str


class Domain(BaseModel):
    """
    Layout-invariant domain definition in LPS:
      - 'definition' clarifies that bbox is voxel-center coordinates
      - 'spacing' is canonical spacing per axis (>=0)
      - 'shape_canonical' is voxel counts (optional but recommended)
      - 'bbox' is center-to-center bounds along L,P,S
      - 'spatial_signature' helps ensure makes it easy to compare domains
    """

    definition: Literal["voxel-center"] = "voxel-center"
    spacing: tuple[float, float, float]
    bbox: BBox
    shape_canonical: tuple[int, int, int] | None = None
    spatial_signature: ContentSignature | None = None

    # ---- spatial hash (geometry only) -------------------------------------
    def _spatial_hash(
        self,
        *,
        digits: int = HASH_DIGITS_DEFAULT,
        digest_size: int = HASH_DIGEST_SIZE,
        method_name: str = HASH_METHOD,
    ) -> ContentSignature:
        """Quantize floats, pack as big-endian int64s, blake2b."""
        scale = 10**digits

        def q(x: float) -> int:
            if not math.isfinite(x):
                raise ValueError(f"Non-finite float in domain: {x}")
            return int(round(x * scale))

        parts: list[int] = []
        parts += [q(s) for s in self.spacing]
        parts += [q(v) for axis in ("L", "P", "S") for v in getattr(self.bbox, axis)]
        if self.shape_canonical is not None:
            parts += list(self.shape_canonical)  # already ints

        # Encode the definition text as a small tag to keep hashes distinct
        tag_map = {"voxel-center": 1}
        parts.append(tag_map.get(self.definition, 0))

        arr = array("q", parts)  # native-endian int64s
        if sys.byteorder == "little":
            arr.byteswap()  # normalize to big-endian for cross-platform stability
        buf = arr.tobytes()

        return ContentSignature(
            method=method_name,
            blake2b="b2:" + hashlib.blake2b(buf, digest_size=digest_size).hexdigest(),
        )

    # ---- auto-populate / verify on construction ---------------------------
    @model_validator(mode="after")
    def _auto_spatial_signature(self) -> Domain:
        # basic sanity for spacing
        if any(s <= 0 or not math.isfinite(s) for s in self.spacing):
            raise ValueError("spacing must be positive and finite")

        computed = self._spatial_hash()  # uses defaults above
        if self.spatial_signature is None:
            # populate if missing
            self.spatial_signature = computed
        else:
            # verify if provided
            if self.spatial_signature.blake2b != computed.blake2b:
                raise ValueError(
                    f"spatial_signature mismatch: provided={self.spatial_signature.blake2b} computed={computed.blake2b}"
                )
        return self

    @model_validator(mode="after")
    def _check_shape(self) -> Domain:
        if self.shape_canonical is not None:
            if any(n <= 0 for n in self.shape_canonical):
                raise ValueError("shape_canonical must be positive")
        return self


# ---- Rendered chain & steps (common result type) ----
class StepAffine(BaseModel):
    kind: Literal["affine"] = "affine"
    file: str
    invert: bool = False


class StepField(BaseModel):
    kind: Literal["displacement_field"] = "displacement_field"
    file: str
    role: Literal["forward", "inverse"]
    grid: Literal["fixed"] = "fixed"

    @model_validator(mode="after")
    def _grid_is_fixed(self) -> StepField:
        if self.grid != "fixed":
            raise ValueError("ANTs SyN displacement fields are defined on the fixed grid")
        return self


Step: TypeAlias = Annotated[StepAffine | StepField, Field(discriminator="kind")]


class Chain(BaseModel):
    order: Literal["top_to_bottom"] = "top_to_bottom"
    steps: list[Step]

    def antspy_apply_transforms_args(self) -> tuple[list[str], list[bool]]:
        transforms: list[str] = []
        whichtoinvert: list[bool] = []
        for s in self.steps:
            if s.kind == "affine":
                transforms.append(s.file)
                whichtoinvert.append(s.invert)
            else:
                transforms.append(s.file)
                whichtoinvert.append(False)
        return transforms, whichtoinvert


# ---- Operation variants (discriminated union on 'kind') ----
class SynTriplet(BaseModel):
    kind: Literal["syn"] = "syn"
    affine: str  # e.g., 0GenericAffine.mat
    warp: str  # e.g., 1Warp.nii.gz
    inverse_warp: str  # e.g., 1InverseWarp.nii.gz

    @model_validator(mode="after")
    def _check_triplet(self) -> SynTriplet:
        if not self.warp or not self.inverse_warp or not self.affine:
            raise ValueError("affine, warp, and inverse_warp are required")
        return self

    def forward_chain(self) -> Chain:
        steps = [
            StepField(file=self.warp, role="forward", grid="fixed"),
            StepAffine(file=self.affine, invert=False),
        ]
        return Chain(steps=steps)

    def inverse_chain(self) -> Chain:
        steps = [
            StepAffine(file=self.affine, invert=True),
            StepField(file=self.inverse_warp, role="inverse", grid="fixed"),
        ]
        return Chain(steps=steps)


# Extend later
Operation: TypeAlias = Annotated[SynTriplet, Field(discriminator="kind")]

# --------- Chain + package ---------


class TransformSidecarV1(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    frame: Literal["LPS"] = "LPS"
    units: Literal["mm"] = "mm"

    fixed_domain: Domain | None = None
    moving_domain: Domain | None = None

    transform: Operation  # <-- discriminated union

    # Unified accessors regardless of 'kind'
    def forward_chain(self) -> Chain:
        return self.transform.forward_chain()

    def inverse_chain(self) -> Chain:
        return self.transform.inverse_chain()

    @model_validator(mode="after")
    def _domains_together(self) -> TransformSidecarV1:
        if (self.fixed_domain is None) ^ (self.moving_domain is None):
            raise ValueError("Provide both fixed_domain and moving_domain or neither")
        return self


# For now, Internal == V1. Later, make Internal a stable runtime model and migrate V1→Internal.
InternalModel: TypeAlias = TransformSidecarV1  # type alias for today


# --- Facade API: callers depend on these, not on V1 directly ---
def load_package(src: str | dict) -> InternalModel:
    data = json.loads(src) if isinstance(src, str) else src
    ver = data.get("schema_version")
    if ver is None:
        raise ValueError("Missing 'schema_version'")
    if str(ver).startswith("1."):
        return TransformSidecarV1.model_validate(data)
    # Future: elif ver.startswith("2."): return V2Envelope.model_validate(data).to_internal()
    raise ValueError(f"Unsupported schema_version: {ver}")


def dump_package(model: InternalModel) -> str:
    # Today: V1 emits V1. Future: Internal → V2Envelope.from_internal(...).model_dump_json(...)
    return model.model_dump_json(by_alias=True, exclude_none=True)
