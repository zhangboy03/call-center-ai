"""
SOP Pre-Call Gating Module

Determines whether a patient should be called and what status to assign if skipped.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FollowupStatus(Enum):
    """随访状态枚举"""

    COMPLETED = "已完成"
    COMPLETED_IPG_OFF = "已完成-IPG关机/取出"
    COMPLETED_DECEASED = "已完成-身故(随访得知)"
    LOST = "失访"
    DO_NOT_CONTACT = "不可触碰"
    OUT_OF_CYCLE_BATTERY = "随访周期外-换电池"
    OUT_OF_CYCLE_HOSPITAL = "随访周期外-医院"
    OUT_OF_CYCLE_FOREIGN = "随访周期外-外籍"
    OUT_OF_CYCLE_TRIAL = "随访周期外-临床试验"
    OUT_OF_CYCLE_DECEASED = "随访周期外-身故(未随访得知)"


@dataclass
class PatientInfo:
    """Patient information for gating decisions."""

    patient_id: str
    name: str
    phone: str
    surgery_date: Optional[datetime] = None
    product_line: str = "DBS_PD"
    hospital: str = ""
    tags: list[str] = None
    ipg_status: str = "active"  # active, off, removed
    deceased: bool = False
    is_foreign: bool = False
    in_clinical_trial: bool = False
    battery_replacement_only: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class GatingResult:
    """Result of gating check."""

    should_call: bool
    skip_status: Optional[FollowupStatus] = None
    reason: str = ""


def check_gating(
    patient: PatientInfo, min_days_since_surgery: int = 180
) -> GatingResult:
    """
    Determine if a patient should be called.

    Args:
        patient: Patient information
        min_days_since_surgery: Minimum days since surgery for follow-up (default 180 = 6 months)

    Returns:
        GatingResult with decision and reason
    """
    # Rule 1: Do not contact flag
    if "不可触碰" in patient.tags or "do_not_contact" in patient.tags:
        logger.info(f"[Gating] Skip {patient.patient_id}: 不可触碰")
        return GatingResult(
            should_call=False,
            skip_status=FollowupStatus.DO_NOT_CONTACT,
            reason="患者标记为不可触碰",
        )

    # Rule 2: Deceased (known before call)
    if patient.deceased:
        logger.info(f"[Gating] Skip {patient.patient_id}: 已知身故")
        return GatingResult(
            should_call=False,
            skip_status=FollowupStatus.OUT_OF_CYCLE_DECEASED,
            reason="患者已身故(系统记录)",
        )

    # Rule 3: IPG removed or off
    if patient.ipg_status in ("removed", "off"):
        logger.info(f"[Gating] Skip {patient.patient_id}: IPG已取出/关机")
        return GatingResult(
            should_call=False,
            skip_status=FollowupStatus.COMPLETED_IPG_OFF,
            reason=f"IPG状态: {patient.ipg_status}",
        )

    # Rule 4: Foreign patient
    if patient.is_foreign:
        logger.info(f"[Gating] Skip {patient.patient_id}: 外籍患者")
        return GatingResult(
            should_call=False,
            skip_status=FollowupStatus.OUT_OF_CYCLE_FOREIGN,
            reason="外籍患者",
        )

    # Rule 5: Clinical trial
    if patient.in_clinical_trial:
        logger.info(f"[Gating] Skip {patient.patient_id}: 临床试验")
        return GatingResult(
            should_call=False,
            skip_status=FollowupStatus.OUT_OF_CYCLE_TRIAL,
            reason="临床试验患者",
        )

    # Rule 6: Battery replacement only
    if patient.battery_replacement_only:
        logger.info(f"[Gating] Skip {patient.patient_id}: 仅换电池")
        return GatingResult(
            should_call=False,
            skip_status=FollowupStatus.OUT_OF_CYCLE_BATTERY,
            reason="仅换电池",
        )

    # Rule 7: Too soon after surgery
    if patient.surgery_date:
        days_since = (datetime.now() - patient.surgery_date).days
        if days_since < min_days_since_surgery:
            logger.info(
                f"[Gating] Skip {patient.patient_id}: 术后{days_since}天 < {min_days_since_surgery}天"
            )
            return GatingResult(
                should_call=False,
                skip_status=FollowupStatus.OUT_OF_CYCLE_HOSPITAL,
                reason=f"术后{days_since}天，未到随访周期",
            )

    # All checks passed
    logger.info(f"[Gating] Call {patient.patient_id}: 符合随访条件")
    return GatingResult(should_call=True, reason="符合随访条件")


def infer_final_status(
    call_completed: bool,
    all_slots_filled: bool,
    ipg_removed_during_call: bool = False,
    patient_deceased_during_call: bool = False,
    connection_attempts: int = 0,
) -> Optional[FollowupStatus]:
    """
    Infer final follow-up status based on call outcome.

    Args:
        call_completed: Whether the call was successfully answered
        all_slots_filled: Whether all required slots were collected
        ipg_removed_during_call: Learned IPG was removed during call
        patient_deceased_during_call: Learned patient deceased during call
        connection_attempts: Number of connection attempts

    Returns:
        Final status or None if call should be retried
    """
    if not call_completed:
        if connection_attempts >= 3:
            return FollowupStatus.LOST
        return None  # Retry

    if patient_deceased_during_call:
        return FollowupStatus.COMPLETED_DECEASED

    if ipg_removed_during_call:
        return FollowupStatus.COMPLETED_IPG_OFF

    if all_slots_filled:
        return FollowupStatus.COMPLETED

    return None  # Incomplete, may need retry
