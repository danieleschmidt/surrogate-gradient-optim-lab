"""Comprehensive compliance framework for data protection and regulatory requirements."""

import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
from datetime import datetime, timedelta

from loguru import logger


class ConsentType(Enum):
    """Types of user consent."""
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    OPT_IN = "opt_in"
    OPT_OUT = "opt_out"


class ProcessingPurpose(Enum):
    """Data processing purposes."""
    OPTIMIZATION = "optimization"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    RESEARCH = "research"
    COMPLIANCE = "compliance"
    SECURITY = "security"


class LegalBasis(Enum):
    """Legal basis for data processing (GDPR)."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataSubject:
    """Data subject information."""
    subject_id: str
    email: Optional[str] = None
    region: Optional[str] = None
    age: Optional[int] = None
    consents: Dict[ProcessingPurpose, Dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class ProcessingRecord:
    """Data processing record for audit trail."""
    record_id: str
    subject_id: str
    purpose: ProcessingPurpose
    legal_basis: LegalBasis
    data_categories: List[str]
    processing_system: str
    timestamp: float
    retention_period_days: int
    location: str
    third_parties: List[str] = field(default_factory=list)


@dataclass
class ConsentRecord:
    """Consent management record."""
    consent_id: str
    subject_id: str
    purpose: ProcessingPurpose
    consent_type: ConsentType
    granted: bool
    timestamp: float
    expiry_date: Optional[float] = None
    withdrawal_date: Optional[float] = None
    version: str = "1.0"
    source: str = "system"


@dataclass
class DataBreachIncident:
    """Data breach incident record."""
    incident_id: str
    detected_at: float
    breach_type: str
    affected_records: int
    data_categories: List[str]
    severity: str  # low, medium, high, critical
    contained: bool = False
    notified_authorities: bool = False
    notified_subjects: bool = False
    resolution_notes: str = ""


class ConsentManager:
    """Manages user consent and preferences."""
    
    def __init__(self):
        self._consents: Dict[str, List[ConsentRecord]] = {}
        self._subjects: Dict[str, DataSubject] = {}
        self._lock = threading.RLock()
        
    def register_subject(self, subject_id: str, email: Optional[str] = None, region: Optional[str] = None) -> DataSubject:
        """Register a new data subject."""
        with self._lock:
            if subject_id in self._subjects:
                return self._subjects[subject_id]
                
            subject = DataSubject(
                subject_id=subject_id,
                email=email,
                region=region
            )
            
            self._subjects[subject_id] = subject
            self._consents[subject_id] = []
            
            logger.info(f"Registered data subject: {subject_id}")
            return subject
            
    def grant_consent(
        self,
        subject_id: str,
        purpose: ProcessingPurpose,
        consent_type: ConsentType = ConsentType.EXPLICIT,
        expiry_days: Optional[int] = None
    ) -> str:
        """Grant consent for a specific purpose."""
        with self._lock:
            if subject_id not in self._subjects:
                raise ValueError(f"Subject {subject_id} not registered")
                
            consent_id = str(uuid.uuid4())
            expiry_date = None
            
            if expiry_days:
                expiry_date = time.time() + (expiry_days * 24 * 3600)
                
            consent = ConsentRecord(
                consent_id=consent_id,
                subject_id=subject_id,
                purpose=purpose,
                consent_type=consent_type,
                granted=True,
                timestamp=time.time(),
                expiry_date=expiry_date
            )
            
            self._consents[subject_id].append(consent)
            
            # Update subject record
            subject = self._subjects[subject_id]
            subject.consents[purpose] = {
                "granted": True,
                "timestamp": consent.timestamp,
                "consent_id": consent_id,
                "expiry_date": expiry_date
            }
            subject.updated_at = time.time()
            
            logger.info(f"Consent granted for subject {subject_id}, purpose {purpose.value}")
            return consent_id
            
    def withdraw_consent(self, subject_id: str, purpose: ProcessingPurpose) -> bool:
        """Withdraw consent for a specific purpose."""
        with self._lock:
            if subject_id not in self._subjects:
                return False
                
            # Find active consent
            for consent in reversed(self._consents[subject_id]):
                if consent.purpose == purpose and consent.granted and not consent.withdrawal_date:
                    consent.withdrawal_date = time.time()
                    
                    # Update subject record
                    subject = self._subjects[subject_id]
                    if purpose in subject.consents:
                        subject.consents[purpose]["granted"] = False
                        subject.consents[purpose]["withdrawal_date"] = consent.withdrawal_date
                    subject.updated_at = time.time()
                    
                    logger.info(f"Consent withdrawn for subject {subject_id}, purpose {purpose.value}")
                    return True
                    
            return False
            
    def check_consent(self, subject_id: str, purpose: ProcessingPurpose) -> bool:
        """Check if consent is valid for a purpose."""
        with self._lock:
            if subject_id not in self._subjects:
                return False
                
            subject = self._subjects[subject_id]
            consent_info = subject.consents.get(purpose)
            
            if not consent_info or not consent_info.get("granted"):
                return False
                
            # Check if consent has been withdrawn
            if consent_info.get("withdrawal_date"):
                return False
                
            # Check if consent has expired
            expiry_date = consent_info.get("expiry_date")
            if expiry_date and time.time() > expiry_date:
                logger.info(f"Consent expired for subject {subject_id}, purpose {purpose.value}")
                return False
                
            return True
            
    def get_subject_consents(self, subject_id: str) -> Dict[ProcessingPurpose, Dict[str, Any]]:
        """Get all consents for a subject."""
        with self._lock:
            if subject_id not in self._subjects:
                return {}
                
            return self._subjects[subject_id].consents.copy()
            
    def cleanup_expired_consents(self) -> int:
        """Clean up expired consents."""
        cleaned_count = 0
        current_time = time.time()
        
        with self._lock:
            for subject_id, subject in self._subjects.items():
                for purpose, consent_info in list(subject.consents.items()):
                    expiry_date = consent_info.get("expiry_date")
                    if expiry_date and current_time > expiry_date:
                        subject.consents[purpose]["granted"] = False
                        subject.updated_at = current_time
                        cleaned_count += 1
                        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired consents")
            
        return cleaned_count


class ProcessingLogger:
    """Logs all data processing activities for compliance."""
    
    def __init__(self, max_records: int = 100000):
        self.max_records = max_records
        self._records: List[ProcessingRecord] = []
        self._lock = threading.RLock()
        
    def log_processing(
        self,
        subject_id: str,
        purpose: ProcessingPurpose,
        legal_basis: LegalBasis,
        data_categories: List[str],
        processing_system: str,
        retention_period_days: int = 365,
        location: str = "unknown",
        third_parties: Optional[List[str]] = None
    ) -> str:
        """Log a data processing activity."""
        with self._lock:
            record_id = str(uuid.uuid4())
            
            record = ProcessingRecord(
                record_id=record_id,
                subject_id=subject_id,
                purpose=purpose,
                legal_basis=legal_basis,
                data_categories=data_categories,
                processing_system=processing_system,
                timestamp=time.time(),
                retention_period_days=retention_period_days,
                location=location,
                third_parties=third_parties or []
            )
            
            self._records.append(record)
            
            # Maintain max records limit
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]
                
            logger.debug(f"Logged processing activity: {record_id}")
            return record_id
            
    def get_subject_processing_history(self, subject_id: str) -> List[ProcessingRecord]:
        """Get processing history for a subject."""
        with self._lock:
            return [r for r in self._records if r.subject_id == subject_id]
            
    def get_processing_by_purpose(self, purpose: ProcessingPurpose) -> List[ProcessingRecord]:
        """Get all processing records for a specific purpose."""
        with self._lock:
            return [r for r in self._records if r.purpose == purpose]
            
    def get_records_for_audit(self, start_time: float, end_time: float) -> List[ProcessingRecord]:
        """Get processing records within a time range for audit."""
        with self._lock:
            return [
                r for r in self._records
                if start_time <= r.timestamp <= end_time
            ]
            
    def generate_processing_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate processing activity report."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with self._lock:
            recent_records = [r for r in self._records if r.timestamp > cutoff_time]
            
            # Aggregate statistics
            purpose_counts = {}
            legal_basis_counts = {}
            system_counts = {}
            
            for record in recent_records:
                purpose_counts[record.purpose.value] = purpose_counts.get(record.purpose.value, 0) + 1
                legal_basis_counts[record.legal_basis.value] = legal_basis_counts.get(record.legal_basis.value, 0) + 1
                system_counts[record.processing_system] = system_counts.get(record.processing_system, 0) + 1
                
            return {
                "period_days": days,
                "total_records": len(recent_records),
                "unique_subjects": len(set(r.subject_id for r in recent_records)),
                "purpose_breakdown": purpose_counts,
                "legal_basis_breakdown": legal_basis_counts,
                "system_breakdown": system_counts,
                "generated_at": time.time()
            }


class DataRetentionManager:
    """Manages data retention and deletion policies."""
    
    def __init__(self):
        self._retention_policies: Dict[str, int] = {}  # data_type -> retention_days
        self._deletion_queue: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Default retention policies
        self._set_default_policies()
        
    def _set_default_policies(self) -> None:
        """Set default retention policies."""
        self._retention_policies = {
            "optimization_data": 365,
            "performance_metrics": 180,
            "health_data": 90,
            "error_logs": 30,
            "audit_logs": 2555,  # 7 years
            "user_data": 365,
            "consent_records": 2555,  # 7 years after withdrawal
            "processing_logs": 2555,  # 7 years
        }
        
    def set_retention_policy(self, data_type: str, retention_days: int) -> None:
        """Set retention policy for a data type."""
        with self._lock:
            self._retention_policies[data_type] = retention_days
            logger.info(f"Set retention policy for {data_type}: {retention_days} days")
            
    def schedule_deletion(self, data_identifier: str, data_type: str, created_at: float) -> None:
        """Schedule data for deletion based on retention policy."""
        with self._lock:
            retention_days = self._retention_policies.get(data_type, 365)
            deletion_time = created_at + (retention_days * 24 * 3600)
            
            deletion_item = {
                "data_identifier": data_identifier,
                "data_type": data_type,
                "created_at": created_at,
                "deletion_time": deletion_time,
                "scheduled_at": time.time()
            }
            
            self._deletion_queue.append(deletion_item)
            
    def process_deletions(self) -> int:
        """Process due deletions."""
        current_time = time.time()
        deleted_count = 0
        
        with self._lock:
            due_deletions = [
                item for item in self._deletion_queue
                if item["deletion_time"] <= current_time
            ]
            
            for item in due_deletions:
                try:
                    # Simulate deletion process
                    logger.info(f"Deleting {item['data_type']} data: {item['data_identifier']}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {item['data_identifier']}: {e}")
                    
            # Remove processed items
            self._deletion_queue = [
                item for item in self._deletion_queue
                if item["deletion_time"] > current_time
            ]
            
        if deleted_count > 0:
            logger.info(f"Processed {deleted_count} data deletions")
            
        return deleted_count
        
    def get_retention_status(self) -> Dict[str, Any]:
        """Get retention status summary."""
        with self._lock:
            current_time = time.time()
            
            pending_deletions = len(self._deletion_queue)
            due_deletions = len([
                item for item in self._deletion_queue
                if item["deletion_time"] <= current_time
            ])
            
            return {
                "retention_policies": self._retention_policies.copy(),
                "pending_deletions": pending_deletions,
                "due_deletions": due_deletions,
                "last_check": current_time
            }


class BreachManager:
    """Manages data breach detection and response."""
    
    def __init__(self):
        self._incidents: List[DataBreachIncident] = []
        self._notification_callbacks: List[Callable] = []
        self._lock = threading.RLock()
        
    def register_notification_callback(self, callback: Callable[[DataBreachIncident], None]) -> None:
        """Register callback for breach notifications."""
        with self._lock:
            self._notification_callbacks.append(callback)
            
    def report_breach(
        self,
        breach_type: str,
        affected_records: int,
        data_categories: List[str],
        severity: str = "medium"
    ) -> str:
        """Report a data breach incident."""
        with self._lock:
            incident_id = str(uuid.uuid4())
            
            incident = DataBreachIncident(
                incident_id=incident_id,
                detected_at=time.time(),
                breach_type=breach_type,
                affected_records=affected_records,
                data_categories=data_categories,
                severity=severity
            )
            
            self._incidents.append(incident)
            
            # Trigger notifications
            for callback in self._notification_callbacks:
                try:
                    callback(incident)
                except Exception as e:
                    logger.error(f"Breach notification callback failed: {e}")
                    
            logger.critical(f"Data breach reported: {incident_id} ({severity})")
            return incident_id
            
    def update_incident(self, incident_id: str, **updates) -> bool:
        """Update breach incident details."""
        with self._lock:
            for incident in self._incidents:
                if incident.incident_id == incident_id:
                    for key, value in updates.items():
                        if hasattr(incident, key):
                            setattr(incident, key, value)
                    logger.info(f"Updated breach incident: {incident_id}")
                    return True
            return False
            
    def get_active_incidents(self) -> List[DataBreachIncident]:
        """Get all active (unresolved) incidents."""
        with self._lock:
            return [i for i in self._incidents if not i.contained]
            
    def get_incidents_requiring_notification(self) -> List[DataBreachIncident]:
        """Get incidents requiring authority notification."""
        with self._lock:
            # GDPR requires notification within 72 hours for high-risk breaches
            notification_threshold = time.time() - (72 * 3600)
            
            return [
                i for i in self._incidents
                if (i.severity in ["high", "critical"] and 
                    i.detected_at > notification_threshold and
                    not i.notified_authorities)
            ]


class ComplianceFramework:
    """Main compliance framework orchestrator."""
    
    def __init__(self):
        self.consent_manager = ConsentManager()
        self.processing_logger = ProcessingLogger()
        self.retention_manager = DataRetentionManager()
        self.breach_manager = BreachManager()
        
        # Compliance state
        self._compliance_checks: Dict[str, bool] = {}
        self._last_audit: Optional[float] = None
        
        # Setup breach notifications
        self.breach_manager.register_notification_callback(self._handle_breach_notification)
        
        logger.info("Compliance framework initialized")
        
    def _handle_breach_notification(self, incident: DataBreachIncident) -> None:
        """Handle breach notification internally."""
        # Log the incident
        logger.critical(f"BREACH ALERT: {incident.breach_type} affecting {incident.affected_records} records")
        
        # Auto-update notification status for critical incidents
        if incident.severity == "critical":
            self.breach_manager.update_incident(
                incident.incident_id,
                notified_authorities=True,
                notified_subjects=True
            )
            
    def register_data_subject(self, subject_id: str, **kwargs) -> DataSubject:
        """Register a new data subject."""
        return self.consent_manager.register_subject(subject_id, **kwargs)
        
    def process_data(
        self,
        subject_id: str,
        purpose: ProcessingPurpose,
        legal_basis: LegalBasis,
        data_categories: List[str],
        system_name: str = "surrogate_optimizer"
    ) -> bool:
        """Process data with compliance checks."""
        
        # Check consent if required
        if legal_basis == LegalBasis.CONSENT:
            if not self.consent_manager.check_consent(subject_id, purpose):
                logger.warning(f"Processing denied: no valid consent for subject {subject_id}, purpose {purpose.value}")
                return False
                
        # Log the processing activity
        self.processing_logger.log_processing(
            subject_id=subject_id,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
            processing_system=system_name
        )
        
        # Schedule for retention management
        for data_category in data_categories:
            self.retention_manager.schedule_deletion(
                data_identifier=f"{subject_id}_{data_category}_{int(time.time())}",
                data_type=data_category,
                created_at=time.time()
            )
            
        return True
        
    def handle_subject_request(self, subject_id: str, request_type: str) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        
        if request_type == "access":
            # Right to access
            consents = self.consent_manager.get_subject_consents(subject_id)
            processing_history = self.processing_logger.get_subject_processing_history(subject_id)
            
            return {
                "subject_id": subject_id,
                "consents": consents,
                "processing_history": [
                    {
                        "purpose": r.purpose.value,
                        "legal_basis": r.legal_basis.value,
                        "timestamp": r.timestamp,
                        "data_categories": r.data_categories
                    }
                    for r in processing_history
                ],
                "request_fulfilled_at": time.time()
            }
            
        elif request_type == "deletion":
            # Right to erasure
            # In real implementation, would trigger actual data deletion
            logger.info(f"Processing deletion request for subject {subject_id}")
            
            return {
                "subject_id": subject_id,
                "deletion_scheduled": True,
                "request_fulfilled_at": time.time()
            }
            
        elif request_type == "portability":
            # Right to data portability
            processing_history = self.processing_logger.get_subject_processing_history(subject_id)
            
            return {
                "subject_id": subject_id,
                "data_export": {
                    "format": "json",
                    "processing_records": len(processing_history),
                    "exported_at": time.time()
                }
            }
            
        else:
            return {"error": f"Unknown request type: {request_type}"}
            
    def run_compliance_audit(self) -> Dict[str, Any]:
        """Run comprehensive compliance audit."""
        audit_start = time.time()
        
        # Check consent compliance
        expired_consents = self.consent_manager.cleanup_expired_consents()
        
        # Check retention compliance
        deleted_items = self.retention_manager.process_deletions()
        
        # Check breach response
        active_breaches = len(self.breach_manager.get_active_incidents())
        pending_notifications = len(self.breach_manager.get_incidents_requiring_notification())
        
        # Generate processing report
        processing_report = self.processing_logger.generate_processing_report()
        
        # Get retention status
        retention_status = self.retention_manager.get_retention_status()
        
        audit_result = {
            "audit_timestamp": audit_start,
            "compliance_status": {
                "expired_consents_cleaned": expired_consents,
                "data_items_deleted": deleted_items,
                "active_breaches": active_breaches,
                "pending_notifications": pending_notifications
            },
            "processing_summary": processing_report,
            "retention_summary": retention_status,
            "audit_duration": time.time() - audit_start
        }
        
        self._last_audit = audit_start
        logger.info(f"Compliance audit completed in {audit_result['audit_duration']:.2f}s")
        
        return audit_result
        
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        return {
            "framework_active": True,
            "last_audit": self._last_audit,
            "components": {
                "consent_management": True,
                "processing_logging": True,
                "data_retention": True,
                "breach_management": True
            },
            "metrics": {
                "registered_subjects": len(self.consent_manager._subjects),
                "processing_records": len(self.processing_logger._records),
                "retention_policies": len(self.retention_manager._retention_policies),
                "breach_incidents": len(self.breach_manager._incidents)
            }
        }


# Global compliance framework instance
global_compliance_framework = ComplianceFramework()

# Convenience functions
def register_subject(subject_id: str, **kwargs) -> DataSubject:
    """Register a data subject."""
    return global_compliance_framework.register_data_subject(subject_id, **kwargs)

def grant_consent(subject_id: str, purpose: ProcessingPurpose, **kwargs) -> str:
    """Grant consent for data processing."""
    return global_compliance_framework.consent_manager.grant_consent(subject_id, purpose, **kwargs)

def process_data_compliant(
    subject_id: str,
    purpose: ProcessingPurpose,
    legal_basis: LegalBasis,
    data_categories: List[str]
) -> bool:
    """Process data with compliance checks."""
    return global_compliance_framework.process_data(subject_id, purpose, legal_basis, data_categories)

def report_data_breach(breach_type: str, affected_records: int, data_categories: List[str]) -> str:
    """Report a data breach."""
    return global_compliance_framework.breach_manager.report_breach(
        breach_type, affected_records, data_categories
    )