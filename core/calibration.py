from typing import List, Dict, Any, Tuple
import numpy as np


class ThresholdCalibrator:
    """
    Infrastructure for biometric threshold evaluation and calibration.
    Analyzes genuine (intra-identity) and impostor (inter-identity) similarity distributions.
    Computes False Accept Rate (FAR), False Reject Rate (FRR), Precision, and Recall.
    """

    MIN_SAMPLES_FOR_CALIBRATION = 10

    @staticmethod
    def evaluate_distributions(
        genuine_similarities: List[float],
        impostor_similarities: List[float],
        threshold: float
    ) -> Dict[str, Any]:
        """
        Evaluates recognition performance against genuine and impostor pair scores.
        """
        n_gen = len(genuine_similarities)
        n_imp = len(impostor_similarities)

        if n_gen < ThresholdCalibrator.MIN_SAMPLES_FOR_CALIBRATION or n_imp < ThresholdCalibrator.MIN_SAMPLES_FOR_CALIBRATION:
            return {
                "calibration_status": "INSUFFICIENT DATA FOR CALIBRATION",
                "sample_counts": {"genuine": n_gen, "impostor": n_imp},
                "threshold": threshold,
                "message": (
                    f"Insufficient sample pairs for statistical calibration: "
                    f"genuine={n_gen}, impostor={n_imp} "
                    f"(minimum required: {ThresholdCalibrator.MIN_SAMPLES_FOR_CALIBRATION})."
                )
            }

        gen_arr = np.array(genuine_similarities, dtype=np.float32)
        imp_arr = np.array(impostor_similarities, dtype=np.float32)

        false_accepts = int(np.sum(imp_arr >= threshold))
        false_rejects = int(np.sum(gen_arr < threshold))
        true_accepts = int(np.sum(gen_arr >= threshold))
        true_rejects = int(np.sum(imp_arr < threshold))

        far = false_accepts / n_imp if n_imp > 0 else 0.0
        frr = false_rejects / n_gen if n_gen > 0 else 0.0

        precision = true_accepts / (true_accepts + false_accepts) if (true_accepts + false_accepts) > 0 else 0.0
        recall = true_accepts / n_gen if n_gen > 0 else 0.0

        return {
            "calibration_status": "CALIBRATION_COMPUTED",
            "threshold": threshold,
            "genuine_distribution": {
                "count": n_gen,
                "mean": float(np.mean(gen_arr)),
                "std": float(np.std(gen_arr)),
                "min": float(np.min(gen_arr)),
                "max": float(np.max(gen_arr))
            },
            "impostor_distribution": {
                "count": n_imp,
                "mean": float(np.mean(imp_arr)),
                "std": float(np.std(imp_arr)),
                "min": float(np.min(imp_arr)),
                "max": float(np.max(imp_arr))
            },
            "metrics": {
                "FAR": float(far),
                "FRR": float(frr),
                "precision": float(precision),
                "recall": float(recall),
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
                "true_accepts": true_accepts,
                "true_rejects": true_rejects
            }
        }
