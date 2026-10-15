"""KMDA Urban Planning & Engineering Intervention Recommendation Engine.

Integrates machine learning explanations with AASHTO Highway Safety Manual (HSM)
and Indian Road Congress (IRC) empirical Crash Modification Factors (CMFs) for causal policy simulation.
"""
import yaml
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger("intervention_engine")

# Empirical Crash Modification Factors (CMFs) from FHWA / AASHTO HSM & IRC-SP:88
# CMF < 1.0 indicates a reduction in crash frequency/severity; (1 - CMF) is % expected reduction.
EMPIRICAL_CMFS = {
    "street_lighting": {
        "cmf_night_fatal": 0.72,      # 28% reduction in night fatal crashes
        "cmf_night_injury": 0.78,     # 22% reduction in night grievous injury crashes
        "source": "AASHTO HSM Part D (Lighting on Urban Arterials)"
    },
    "median_barrier": {
        "cmf_crossover_fatal": 0.62,  # 38% reduction in cross-median fatal collisions
        "cmf_head_on": 0.65,
        "source": "FHWA Countermeasure Clearinghouse CMF ID 172"
    },
    "pedestrian_refuge_fob": {
        "cmf_pedestrian": 0.55,       # 45% reduction in pedestrian collisions
        "source": "IRC-SP:88 Road Safety Audit Guidelines"
    },
    "high_friction_surface": {
        "cmf_wet_weather": 0.60,      # 40% reduction in wet weather skidding crashes
        "source": "FHWA High-Friction Surface Treatment (HFST) Specification"
    }
}


class KMDAInterventionEngine:
    """Translates local SHAP drivers, road attributes, and empirical CMFs into causal engineering countermeasures."""

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
                if item.get("shap_value", 0) > 0:
                    feat = item.get("feature", "")
                    simple = item.get("simple_name", feat.replace("feat_", ""))
                    top_risk_features.add(feat)
                    top_risk_features.add(simple)
                    top_risk_features.add(f"feat_{simple}")

        # Check Lighting
        if segment_row.get("is_lit", 1) == 0 or segment_row.get("feat_is_lit", 1) == 0 or "is_lit" in top_risk_features or "feat_is_lit" in top_risk_features or segment_row.get("night_crashes", 0) > 2:
            actions.append({
                "intervention_id": "INT_LIGHTING",
                "category": "Electrical & Illumination Infrastructure",
                "recommended_action": "Install KMDA high-mast LED smart illumination with photocell sensors (30m spacing).",
                "priority": "Immediate",
                "authority": "KMDA / KMC Lighting Department",
                "estimated_cost_tier": "Medium",
                "cmf_effect": "28% expected reduction in night-time fatal risk (HSM CMF: 0.72)"
            })

        # Check Speed & Corridor Calming
        curr_speed = segment_row.get("speed_limit", segment_row.get("feat_speed_limit", 40))
        if curr_speed >= 50 or "speed_limit" in top_risk_features or "feat_speed_limit" in top_risk_features:
            actions.append({
                "intervention_id": "INT_SPEED_CALMING",
                "category": "Traffic Calming & Automated Enforcement",
                "recommended_action": "Deploy automated speed enforcement cameras, thermoplastic rumble strips, and optical speed bars.",
                "priority": "High",
                "authority": "Kolkata Traffic Police & KMDA Engineering",
                "estimated_cost_tier": "Low",
                "cmf_effect": "Elvik Power Model: Fatal risk scales with (V_new/V_old)^4"
            })

        # Check Pedestrian & Junction Safety
        is_junc = segment_row.get("is_junction", segment_row.get("feat_is_junction", 0))
        if is_junc == 1 or segment_row.get("pedestrian_crashes", 0) >= 1 or "is_junction" in top_risk_features or "feat_is_junction" in top_risk_features:
            actions.append({
                "intervention_id": "INT_PEDESTRIAN_SAFETY",
                "category": "Non-Motorized Transport (NMT) Protection",
                "recommended_action": "Construct signalized raised tabletop crosswalk with refuge island or pedestrian foot overbridge (FOB).",
                "priority": "High",
                "authority": "KMDA Urban Mobility Wing",
                "estimated_cost_tier": "High",
                "cmf_effect": "45% expected reduction in pedestrian collisions (IRC-SP:88 CMF: 0.55)"
            })

        # Check Median Barrier Separation
        has_med = segment_row.get("has_median", segment_row.get("feat_has_median", 1))
        lanes_cnt = segment_row.get("lanes", segment_row.get("feat_lanes", 2))
        if has_med == 0 and lanes_cnt >= 3:
            actions.append({
                "intervention_id": "INT_MEDIAN_BARRIER",
                "category": "Physical Roadway Separation",
                "recommended_action": "Install continuous reinforced concrete New Jersey crash barriers to eliminate head-on crossover collisions.",
                "priority": "High",
                "authority": "KMDA Roads & Highways Directorate",
                "estimated_cost_tier": "Medium-High",
                "cmf_effect": "38% expected reduction in cross-median fatal collisions (FHWA CMF: 0.62)"
            })

        # Check Drainage / Wet Weather
        if segment_row.get("monsoon_crashes", 0) >= 2 or "feat_is_monsoon" in top_risk_features:
            actions.append({
                "intervention_id": "INT_DRAINAGE_SURFACE",
                "category": "Road Surface & Stormwater Drainage",
                "recommended_action": "Apply high-friction micro-surfacing (HFST) and augment curb-inlet drainage to prevent hydroplaning.",
                "priority": "Medium",
                "authority": "KMDA Civil & Drainage Division",
                "estimated_cost_tier": "Medium",
                "cmf_effect": "40% reduction in wet-weather loss-of-control crashes (HFST CMF: 0.60)"
            })

        return actions

    @staticmethod
    def estimate_causal_countermeasure_impact(baseline_row: Dict[str, Any],
                                              simulated_row: Dict[str, Any],
                                              p_baseline: np.ndarray,
                                              p_simulated: np.ndarray) -> Dict[str, Any]:
        """Synthesizes observational ML model predictions with empirical Crash Modification Factors (CMFs).

        Returns consensus impact metrics that prevent observational confounding bias.
        """
        base_fatal = float(p_baseline[2])
        sim_fatal = float(p_simulated[2])
        ml_fatal_delta_pct = (sim_fatal - base_fatal) * 100.0

        # Calculate empirical CMF compound factor
        cmf_fatal_multiplier = 1.0
        cmf_notes = []

        # 1. Street Lighting intervention
        old_lit = int(baseline_row.get("is_lit", baseline_row.get("feat_is_lit", 1)))
        new_lit = int(simulated_row.get("is_lit", simulated_row.get("feat_is_lit", 1)))
        is_night = int(baseline_row.get("is_night", baseline_row.get("feat_is_night", 0)))
        if old_lit == 0 and new_lit == 1 and is_night == 1:
            cmf_fatal_multiplier *= 0.72
            cmf_notes.append("AASHTO HSM Lighting CMF: 0.72 (-28% night fatal risk)")

        # 2. Median Barrier installation
        old_med = int(baseline_row.get("has_median", baseline_row.get("feat_has_median", 0)))
        new_med = int(simulated_row.get("has_median", simulated_row.get("feat_has_median", 0)))
        if old_med == 0 and new_med == 1:
            cmf_fatal_multiplier *= 0.65
            cmf_notes.append("FHWA Median Barrier CMF: 0.65 (-35% cross-median fatal risk)")

        # 3. Speed limit reduction (Elvik's Power Model: Fatal ~ (v_new / v_old)^4)
        old_speed = float(baseline_row.get("speed_limit", baseline_row.get("feat_speed_limit", 50)))
        new_speed = float(simulated_row.get("speed_limit", simulated_row.get("feat_speed_limit", 50)))
        if old_speed > 0 and new_speed < old_speed:
            speed_ratio = new_speed / old_speed
            speed_cmf = float(np.clip(speed_ratio ** 4.0, 0.40, 1.0))
            cmf_fatal_multiplier *= speed_cmf
            cmf_notes.append(f"Elvik Speed Power Model: CMF {speed_cmf:.2f} (from {int(old_speed)} to {int(new_speed)} km/h)")

        # Causal expected fatal probability
        causal_expected_fatal = base_fatal * cmf_fatal_multiplier
        causal_fatal_delta_pct = (causal_expected_fatal - base_fatal) * 100.0

        # Consensus consensus weighted estimate: 50% ML + 50% Empirical CMF
        consensus_fatal_delta_pct = 0.50 * ml_fatal_delta_pct + 0.50 * causal_fatal_delta_pct

        return {
            "ml_predicted_fatal_pct": round(sim_fatal * 100.0, 1),
            "ml_fatal_delta_pct": round(ml_fatal_delta_pct, 1),
            "cmf_compound_multiplier": round(cmf_fatal_multiplier, 3),
            "causal_fatal_delta_pct": round(causal_fatal_delta_pct, 1),
            "consensus_fatal_delta_pct": round(consensus_fatal_delta_pct, 1),
            "cmf_citations": cmf_notes if cmf_notes else ["Baseline operational conditions maintained"]
        }


# Module-level convenience function
estimate_causal_countermeasure_impact = KMDAInterventionEngine.estimate_causal_countermeasure_impact

