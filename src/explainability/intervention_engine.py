"""KMDA Urban Planning & Engineering Intervention Recommendation Engine."""
import yaml
import os
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("intervention_engine")

class KMDAInterventionEngine:
    """Translates local SHAP drivers and road attributes into urban engineering countermeasures."""

    def __init__(self, rules_path: str = "configs/intervention_rules.yaml"):
        self.rules = []
        if os.path.exists(rules_path):
            with open(rules_path, "r") as f:
                config = yaml.safe_load(f)
                self.rules = config.get("interventions", [])
        else:
            logger.warning(f"Rules config not found at {rules_path}, using default hardcoded rules.")

    def recommend_for_segment(self, segment_row: Dict[str, Any], top_shap_drivers: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Evaluates segment characteristics and SHAP risk factors to prescribe prioritized civil fixes."""
        actions = []
        top_risk_features = set()
        if top_shap_drivers:
            for item in top_shap_drivers[:5]:
                if item["shap_value"] > 0:
                    top_risk_features.add(item["feature"])

        # Check Lighting
        if segment_row.get("is_lit", 1) == 0 or "feat_is_lit" in top_risk_features or segment_row.get("night_crashes", 0) > 2:
            actions.append({
                "intervention_id": "INT_LIGHTING",
                "category": "Electrical & Illumination Infrastructure",
                "recommended_action": "Install KMDA high-mast LED smart illumination with photocell sensors (30m spacing).",
                "priority": "Immediate",
                "authority": "KMDA / KMC Lighting Department",
                "estimated_cost_tier": "Medium"
            })

        # Check Speed & Corridor Calming
        if segment_row.get("speed_limit", 40) >= 50 or "feat_speed_limit" in top_risk_features:
            actions.append({
                "intervention_id": "INT_SPEED_CALMING",
                "category": "Traffic Calming & Automated Enforcement",
                "recommended_action": "Deploy automated speed enforcement cameras, thermoplastic rumble strips, and optical speed bars.",
                "priority": "High",
                "authority": "Kolkata Traffic Police & KMDA Engineering",
                "estimated_cost_tier": "Low"
            })

        # Check Pedestrian & Junction Safety
        if segment_row.get("is_junction", 0) == 1 or segment_row.get("pedestrian_crashes", 0) >= 1 or "feat_is_junction" in top_risk_features:
            actions.append({
                "intervention_id": "INT_PEDESTRIAN_SAFETY",
                "category": "Non-Motorized Transport (NMT) Protection",
                "recommended_action": "Construct signalized raised tabletop crosswalk with refuge island or pedestrian foot overbridge (FOB).",
                "priority": "High",
                "authority": "KMDA Urban Mobility Wing",
                "estimated_cost_tier": "High"
            })

        # Check Median Barrier Separation
        if segment_row.get("has_median", 1) == 0 and segment_row.get("lanes", 2) >= 3:
            actions.append({
                "intervention_id": "INT_MEDIAN_BARRIER",
                "category": "Physical Roadway Separation",
                "recommended_action": "Install continuous reinforced concrete New Jersey crash barriers to eliminate head-on crossover collisions.",
                "priority": "High",
                "authority": "KMDA Roads & Highways Directorate",
                "estimated_cost_tier": "Medium-High"
            })

        # Check Drainage / Wet Weather
        if segment_row.get("monsoon_crashes", 0) >= 2 or "feat_is_monsoon" in top_risk_features:
            actions.append({
                "intervention_id": "INT_DRAINAGE_SURFACE",
                "category": "Road Surface & Stormwater Drainage",
                "recommended_action": "Apply high-friction micro-surfacing (HFST) and augment curb-inlet drainage to prevent hydroplaning.",
                "priority": "Medium",
                "authority": "KMDA Civil & Drainage Division",
                "estimated_cost_tier": "Medium"
            })

        return actions
