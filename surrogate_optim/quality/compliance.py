"""Compliance checking utilities for data protection and regulatory requirements."""

from datetime import datetime
import re
from typing import Any, Dict

from ..monitoring.logging import get_logger


class ComplianceChecker:
    """Compliance checker for data protection and regulatory requirements."""

    def __init__(self):
        """Initialize compliance checker."""
        self.logger = get_logger()

    def check_gdpr_compliance(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance requirements.
        
        Args:
            data_config: Data handling configuration
            
        Returns:
            GDPR compliance check results
        """
        issues = []
        compliant_features = []

        # Check for data minimization
        if data_config.get("data_minimization", False):
            compliant_features.append("Data minimization implemented")
        else:
            issues.append({
                "type": "gdpr_violation",
                "severity": "high",
                "article": "Article 5(1)(c)",
                "requirement": "Data minimization",
                "message": "Data minimization not implemented - collect only necessary data",
            })

        # Check for purpose limitation
        if data_config.get("purpose_specification", False):
            compliant_features.append("Purpose specification implemented")
        else:
            issues.append({
                "type": "gdpr_violation",
                "severity": "high",
                "article": "Article 5(1)(b)",
                "requirement": "Purpose limitation",
                "message": "Purpose specification missing - clearly define data use purposes",
            })

        # Check for data retention policies
        if data_config.get("retention_policy", False):
            compliant_features.append("Data retention policy implemented")
        else:
            issues.append({
                "type": "gdpr_violation",
                "severity": "medium",
                "article": "Article 5(1)(e)",
                "requirement": "Storage limitation",
                "message": "Data retention policy missing - implement automatic data deletion",
            })

        # Check for consent management
        if data_config.get("consent_management", False):
            compliant_features.append("Consent management implemented")
        else:
            issues.append({
                "type": "gdpr_violation",
                "severity": "high",
                "article": "Article 6",
                "requirement": "Lawful basis for processing",
                "message": "Consent management missing - implement consent tracking",
            })

        # Check for data subject rights
        rights_implemented = data_config.get("subject_rights", {})
        required_rights = [
            "right_to_access",
            "right_to_rectification",
            "right_to_erasure",
            "right_to_portability",
            "right_to_object",
        ]

        for right in required_rights:
            if rights_implemented.get(right, False):
                compliant_features.append(f"{right} implemented")
            else:
                issues.append({
                    "type": "gdpr_violation",
                    "severity": "medium",
                    "article": "Articles 15-22",
                    "requirement": f"Data subject {right}",
                    "message": f"{right} not implemented - provide mechanism for data subjects",
                })

        # Check for security measures
        security_measures = data_config.get("security_measures", {})
        if security_measures.get("encryption", False):
            compliant_features.append("Data encryption implemented")
        else:
            issues.append({
                "type": "gdpr_violation",
                "severity": "high",
                "article": "Article 32",
                "requirement": "Security of processing",
                "message": "Data encryption missing - implement encryption for sensitive data",
            })

        if security_measures.get("access_control", False):
            compliant_features.append("Access control implemented")
        else:
            issues.append({
                "type": "gdpr_violation",
                "severity": "medium",
                "article": "Article 32",
                "requirement": "Security of processing",
                "message": "Access control missing - implement role-based access control",
            })

        return {
            "regulation": "GDPR",
            "issues": issues,
            "compliant_features": compliant_features,
            "compliance_score": len(compliant_features) / (len(compliant_features) + len(issues)),
            "overall_compliant": len([i for i in issues if i["severity"] == "high"]) == 0,
        }

    def check_ccpa_compliance(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check CCPA compliance requirements.
        
        Args:
            data_config: Data handling configuration
            
        Returns:
            CCPA compliance check results
        """
        issues = []
        compliant_features = []

        # Check for privacy notice
        if data_config.get("privacy_notice", False):
            compliant_features.append("Privacy notice provided")
        else:
            issues.append({
                "type": "ccpa_violation",
                "severity": "high",
                "section": "Section 1798.100",
                "requirement": "Consumer notice",
                "message": "Privacy notice missing - provide clear notice of data collection",
            })

        # Check for consumer rights
        consumer_rights = data_config.get("consumer_rights", {})
        required_rights = [
            "right_to_know",
            "right_to_delete",
            "right_to_opt_out",
            "right_to_non_discrimination",
        ]

        for right in required_rights:
            if consumer_rights.get(right, False):
                compliant_features.append(f"{right} implemented")
            else:
                issues.append({
                    "type": "ccpa_violation",
                    "severity": "medium",
                    "section": "Sections 1798.100-1798.125",
                    "requirement": f"Consumer {right}",
                    "message": f"{right} not implemented - provide mechanism for consumers",
                })

        # Check for opt-out mechanism
        if data_config.get("opt_out_mechanism", False):
            compliant_features.append("Opt-out mechanism implemented")
        else:
            issues.append({
                "type": "ccpa_violation",
                "severity": "high",
                "section": "Section 1798.135",
                "requirement": "Opt-out mechanism",
                "message": "Opt-out mechanism missing - provide 'Do Not Sell' option",
            })

        return {
            "regulation": "CCPA",
            "issues": issues,
            "compliant_features": compliant_features,
            "compliance_score": len(compliant_features) / (len(compliant_features) + len(issues)),
            "overall_compliant": len([i for i in issues if i["severity"] == "high"]) == 0,
        }

    def check_data_protection_best_practices(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check general data protection best practices.
        
        Args:
            data_config: Data handling configuration
            
        Returns:
            Data protection compliance results
        """
        issues = []
        compliant_features = []

        # Check for data classification
        if data_config.get("data_classification", False):
            compliant_features.append("Data classification implemented")
        else:
            issues.append({
                "type": "best_practice_violation",
                "severity": "medium",
                "requirement": "Data classification",
                "message": "Data classification missing - classify data by sensitivity level",
            })

        # Check for audit logging
        if data_config.get("audit_logging", False):
            compliant_features.append("Audit logging implemented")
        else:
            issues.append({
                "type": "best_practice_violation",
                "severity": "medium",
                "requirement": "Audit logging",
                "message": "Audit logging missing - log all data access and modifications",
            })

        # Check for data anonymization
        if data_config.get("anonymization", False):
            compliant_features.append("Data anonymization implemented")
        else:
            issues.append({
                "type": "best_practice_violation",
                "severity": "low",
                "requirement": "Data anonymization",
                "message": "Data anonymization not implemented - consider anonymizing sensitive data",
            })

        # Check for regular compliance audits
        if data_config.get("regular_audits", False):
            compliant_features.append("Regular compliance audits scheduled")
        else:
            issues.append({
                "type": "best_practice_violation",
                "severity": "low",
                "requirement": "Regular audits",
                "message": "Regular audits not scheduled - implement periodic compliance reviews",
            })

        return {
            "standard": "Data Protection Best Practices",
            "issues": issues,
            "compliant_features": compliant_features,
            "compliance_score": len(compliant_features) / (len(compliant_features) + len(issues)),
            "overall_compliant": len([i for i in issues if i["severity"] == "high"]) == 0,
        }

    def detect_pii_in_data(self, data_sample: Any) -> Dict[str, Any]:
        """Detect potential PII in data samples.
        
        Args:
            data_sample: Sample of data to analyze
            
        Returns:
            PII detection results
        """
        pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }

        detected_pii = []

        # Convert data to string for pattern matching
        data_str = str(data_sample)

        for pii_type, pattern in pii_patterns.items():
            matches = re.findall(pattern, data_str)
            if matches:
                detected_pii.append({
                    "pii_type": pii_type,
                    "count": len(matches),
                    "severity": "high" if pii_type in ["ssn", "credit_card"] else "medium",
                })

        return {
            "pii_detected": len(detected_pii) > 0,
            "pii_types": detected_pii,
            "total_pii_instances": sum(pii["count"] for pii in detected_pii),
            "risk_level": "high" if any(pii["severity"] == "high" for pii in detected_pii) else "medium" if detected_pii else "low",
        }

    def check_cross_border_compliance(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check cross-border data transfer compliance.
        
        Args:
            data_config: Data handling configuration including transfer info
            
        Returns:
            Cross-border compliance results
        """
        issues = []
        compliant_features = []

        data_transfers = data_config.get("data_transfers", [])

        for transfer in data_transfers:
            destination = transfer.get("destination_country", "unknown")
            mechanism = transfer.get("transfer_mechanism", None)

            # Check for adequacy decisions (simplified list)
            adequate_countries = [
                "Andorra", "Argentina", "Canada", "Faroe Islands", "Guernsey",
                "Israel", "Isle of Man", "Japan", "Jersey", "New Zealand",
                "Switzerland", "United Kingdom", "Uruguay"
            ]

            if destination in adequate_countries:
                compliant_features.append(f"Transfer to {destination} (adequacy decision)")
            elif mechanism in ["standard_contractual_clauses", "binding_corporate_rules", "certification"]:
                compliant_features.append(f"Transfer to {destination} with {mechanism}")
            else:
                issues.append({
                    "type": "cross_border_violation",
                    "severity": "high",
                    "requirement": "Adequate safeguards",
                    "message": f"Transfer to {destination} lacks adequate safeguards",
                    "destination": destination,
                })

        return {
            "standard": "Cross-border Data Transfers",
            "issues": issues,
            "compliant_features": compliant_features,
            "transfers_checked": len(data_transfers),
            "compliance_score": len(compliant_features) / max(len(data_transfers), 1),
            "overall_compliant": len([i for i in issues if i["severity"] == "high"]) == 0,
        }

    def comprehensive_compliance_check(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive compliance check across all regulations.
        
        Args:
            data_config: Complete data handling configuration
            
        Returns:
            Comprehensive compliance results
        """
        results = {
            "check_timestamp": datetime.now().isoformat(),
            "regulations_checked": {},
            "overall_compliance": {},
            "priority_issues": [],
            "summary": {},
        }

        # Run individual compliance checks
        gdpr_result = self.check_gdpr_compliance(data_config)
        ccpa_result = self.check_ccpa_compliance(data_config)
        best_practices_result = self.check_data_protection_best_practices(data_config)
        cross_border_result = self.check_cross_border_compliance(data_config)

        results["regulations_checked"] = {
            "gdpr": gdpr_result,
            "ccpa": ccpa_result,
            "best_practices": best_practices_result,
            "cross_border": cross_border_result,
        }

        # Calculate overall compliance
        all_issues = []
        all_features = []

        for check_result in results["regulations_checked"].values():
            all_issues.extend(check_result["issues"])
            all_features.extend(check_result["compliant_features"])

        # Identify priority issues (high severity)
        priority_issues = [issue for issue in all_issues if issue.get("severity") == "high"]
        results["priority_issues"] = priority_issues

        # Overall compliance summary
        results["overall_compliance"] = {
            "total_issues": len(all_issues),
            "high_severity_issues": len(priority_issues),
            "medium_severity_issues": len([i for i in all_issues if i.get("severity") == "medium"]),
            "low_severity_issues": len([i for i in all_issues if i.get("severity") == "low"]),
            "compliant_features": len(all_features),
            "overall_score": len(all_features) / max(len(all_features) + len(all_issues), 1),
            "is_compliant": len(priority_issues) == 0,
        }

        # Summary by regulation
        results["summary"] = {
            regulation: {
                "compliant": result["overall_compliant"],
                "score": result["compliance_score"],
                "issues": len(result["issues"]),
            }
            for regulation, result in results["regulations_checked"].items()
        }

        return results

    def generate_compliance_report(self, compliance_results: Dict[str, Any]) -> str:
        """Generate comprehensive compliance report.
        
        Args:
            compliance_results: Results from comprehensive_compliance_check
            
        Returns:
            Formatted compliance report
        """
        report = []
        report.append("DATA PROTECTION COMPLIANCE REPORT")
        report.append("=" * 50)

        overall = compliance_results["overall_compliance"]

        # Overall status
        status = "COMPLIANT" if overall["is_compliant"] else "NON-COMPLIANT"
        report.append(f"\nOVERALL COMPLIANCE STATUS: {status}")
        report.append(f"Overall Score: {overall['overall_score']:.3f}")
        report.append(f"Total Issues: {overall['total_issues']}")
        report.append(f"  High Severity: {overall['high_severity_issues']}")
        report.append(f"  Medium Severity: {overall['medium_severity_issues']}")
        report.append(f"  Low Severity: {overall['low_severity_issues']}")
        report.append(f"Compliant Features: {overall['compliant_features']}")

        # Priority issues
        if compliance_results["priority_issues"]:
            report.append("\nPRIORITY ISSUES (High Severity):")
            report.append("-" * 30)

            for issue in compliance_results["priority_issues"]:
                regulation = issue.get("type", "unknown").replace("_violation", "").upper()
                requirement = issue.get("requirement", "Unknown")
                message = issue.get("message", "No message")

                report.append(f"\n[{regulation}] {requirement}")
                report.append(f"  {message}")

                if "article" in issue:
                    report.append(f"  Reference: {issue['article']}")
                elif "section" in issue:
                    report.append(f"  Reference: {issue['section']}")

        # Regulation-specific results
        report.append("\nREGULATION-SPECIFIC RESULTS:")
        report.append("-" * 30)

        for regulation, summary in compliance_results["summary"].items():
            status = "COMPLIANT" if summary["compliant"] else "NON-COMPLIANT"
            report.append(f"\n{regulation.upper()}: {status} (Score: {summary['score']:.3f})")
            report.append(f"  Issues: {summary['issues']}")

        # Detailed results
        for regulation, result in compliance_results["regulations_checked"].items():
            report.append(f"\n{regulation.upper()} DETAILED RESULTS:")
            report.append("-" * 20)

            # Compliant features
            if result["compliant_features"]:
                report.append("  ✓ Compliant Features:")
                for feature in result["compliant_features"]:
                    report.append(f"    - {feature}")

            # Issues
            if result["issues"]:
                report.append("  ✗ Issues:")
                for issue in result["issues"]:
                    severity = issue.get("severity", "unknown").upper()
                    requirement = issue.get("requirement", "Unknown")
                    message = issue.get("message", "No message")

                    report.append(f"    [{severity}] {requirement}: {message}")

        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 30)

        if compliance_results["priority_issues"]:
            report.append("IMMEDIATE ACTIONS (High Priority):")
            for issue in compliance_results["priority_issues"]:
                report.append(f"  - {issue.get('message', 'Address compliance issue')}")

        report.append("\nGENERAL RECOMMENDATIONS:")
        report.append("- Implement a data protection impact assessment process")
        report.append("- Establish regular compliance monitoring and auditing")
        report.append("- Provide data protection training for all team members")
        report.append("- Maintain detailed records of data processing activities")
        report.append("- Implement privacy by design principles")
        report.append("- Regularly review and update privacy policies")
        report.append("- Establish incident response procedures")

        return "\n".join(report)


def check_compliance(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for comprehensive compliance checking.
    
    Args:
        data_config: Data handling configuration
        
    Returns:
        Compliance check results
    """
    checker = ComplianceChecker()
    return checker.comprehensive_compliance_check(data_config)
