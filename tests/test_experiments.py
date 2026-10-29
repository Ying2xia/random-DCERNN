from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_public_archive import archive_members
from fedcernn.baselines import BaselineConfig, fit_baseline
from fedcernn.data_eicu import (
    EICU24HourBuilder,
    EVENT_TABLES,
    PATIENT_COLUMNS,
    load_restricted_eicu_matrices,
    validate_eicu_directory,
)
from fedcernn.data_meps import prepare_meps, validate_meps_feature_columns
from fedcernn.experiments import (
    CSV_FIELDS,
    ExperimentScenario,
    ResultStore,
    _configured_method_rounds,
    _optimizer_config,
    _partition_indices,
    _scenario_seed_key,
    derive_seed,
    expand_scenarios,
    load_config,
    run_experiment,
)
from fedcernn.federated import FedCERNN
from fedcernn.metrics import (
    evaluate_predictions,
    holm_adjust,
    paired_method_tests,
)
from fedcernn.model import CERNN, flatten_parameters, parameter_count
from fedcernn.simulation import (
    SUPPORTED_DGPS,
    SUPPORTED_ERRORS,
    error_expectiles,
    generate_simulation,
)
from fedcernn.training import OptimizerConfig


class SimulationTests(unittest.TestCase):
    def test_public_archive_allowlist_excludes_local_data_and_results(self) -> None:
        members = archive_members(Path(__file__).resolve().parents[1])
        self.assertIn(Path("README.md"), members)
        self.assertIn(Path("build_public_archive.py"), members)
        self.assertFalse(
            any({"data", "results"}.intersection(member.parts) for member in members)
        )

    def test_optimizer_config_parses_stepwise_safeguards(self) -> None:
        config = _optimizer_config(
            {
                "stepwise_lbfgs": True,
                "select_min_gradient_checkpoint": True,
                "max_rms_parameter_change": 0.125,
                "project_rms_parameter_change": True,
            }
        )
        self.assertTrue(config.stepwise_lbfgs)
        self.assertTrue(config.select_min_gradient_checkpoint)
        self.assertEqual(config.max_rms_parameter_change, 0.125)
        self.assertTrue(config.project_rms_parameter_change)

    def test_all_designs_and_errors_have_exact_truth(self) -> None:
        taus = (0.1, 0.5, 0.9)
        for dgp in SUPPORTED_DGPS:
            for error in SUPPORTED_ERRORS:
                first = generate_simulation(24, dgp, error, seed=19, taus=taus)
                second = generate_simulation(24, dgp, error, seed=19, taus=taus)
                self.assertEqual(first.true_expectiles.shape, (24, 3))
                self.assertTrue(np.array_equal(first.X, second.X))
                self.assertTrue(np.array_equal(first.y, second.y))
                expected = (
                    first.location[:, None]
                    + first.scale[:, None] * error_expectiles(error, taus)[None, :]
                )
                np.testing.assert_allclose(first.true_expectiles, expected)
                self.assertTrue(
                    np.all(np.diff(first.true_expectiles, axis=1) > 0.0)
                )

    def test_standardized_error_expectiles_are_ordered_and_centered(self) -> None:
        for error in SUPPORTED_ERRORS:
            values = error_expectiles(error, (0.1, 0.5, 0.9))
            self.assertLess(values[0], 0.0)
            self.assertEqual(values[1], 0.0)
            self.assertGreater(values[2], 0.0)


class MetricTests(unittest.TestCase):
    def test_metrics_and_crossing(self) -> None:
        y = np.asarray([0.0, 2.0])
        perfect = np.asarray([[0.0, 0.0], [2.0, 2.0]])
        metrics = evaluate_predictions(
            y, perfect, (0.25, 0.75), true_expectiles=perfect
        )
        self.assertEqual(metrics.composite_risk, 0.0)
        self.assertEqual(metrics.calibration, 0.0)
        self.assertEqual(metrics.true_mae, 0.0)
        self.assertEqual(metrics.crossing_rate, 0.0)
        crossed = evaluate_predictions(y, perfect[:, ::-1] + [[1.0, 0.0], [1.0, 0.0]], (0.25, 0.75))
        self.assertEqual(crossed.crossing_rate, 1.0)

    def test_calibration_uses_the_scale_normalized_expectile_moment(self) -> None:
        metrics = evaluate_predictions(
            np.asarray([1.0]),
            np.asarray([[0.0]]),
            (0.5,),
        )
        # The reported convention is |mean{a_tau(r) r}|.  The derivative's
        # immaterial common factor of two is deliberately omitted.
        self.assertEqual(metrics.calibration, 0.5)

    def test_survey_weighted_crossing_uses_sample_weights(self) -> None:
        y = np.zeros(2)
        predictions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        first_heavy = evaluate_predictions(
            y,
            predictions,
            (0.25, 0.75),
            sample_weight=np.asarray([99.0, 1.0]),
        )
        second_heavy = evaluate_predictions(
            y,
            predictions,
            (0.25, 0.75),
            sample_weight=np.asarray([1.0, 99.0]),
        )
        self.assertAlmostEqual(first_heavy.crossing_rate, 0.99)
        self.assertAlmostEqual(first_heavy.pairwise_crossing_rate, 0.99)
        self.assertAlmostEqual(second_heavy.crossing_rate, 0.01)
        self.assertAlmostEqual(second_heavy.pairwise_crossing_rate, 0.01)

    def test_holm_and_zero_variance_paired_test(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
        records = []
        for repetition in range(4):
            records.extend(
                [
                    {
                        "method": "FedCERNN",
                        "repetition": repetition,
                        "composite_risk": 1.0,
                    },
                    {
                        "method": "same",
                        "repetition": repetition,
                        "composite_risk": 1.0,
                    },
                ]
            )
        comparison = paired_method_tests(records)[0]
        self.assertEqual(comparison["p_value"], 1.0)
        self.assertEqual(comparison["mean_difference"], 0.0)
        self.assertEqual(comparison["difference_ci_lower"], 0.0)
        self.assertEqual(comparison["difference_ci_upper"], 0.0)


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(18, 3))
        y = X[:, 0] - 0.5 * X[:, 1]
        self.clients = [
            (
                torch.as_tensor(X[index], dtype=torch.float64),
                torch.as_tensor(y[index], dtype=torch.float64),
            )
            for index in np.array_split(np.arange(18), 3)
        ]
        self.config = BaselineConfig(
            rounds=2,
            local_steps=2,
            local_lr=0.01,
            l2_lambda=0.001,
            fedprox_mu=0.01,
            centralized_optimizer=OptimizerConfig(adam_steps=1, lbfgs_steps=0),
            host_optimizer=OptimizerConfig(
                adam_steps=2,
                adam_lr=0.005,
                lbfgs_steps=0,
            ),
            local_optimizer=OptimizerConfig(adam_steps=1, lbfgs_steps=0),
        )

    def _model(self) -> CERNN:
        torch.manual_seed(44)
        return CERNN(3, hidden_dims=(4, 3), taus=(0.25, 0.75))

    def test_remote_only_communication_formulas(self) -> None:
        model_bytes = (
            parameter_count(self._model())
            * next(self._model().parameters()).element_size()
        )
        remote = len(self.clients) - 1
        expectations = {
            "CSL-CERNN": 2 * remote * model_bytes * self.config.rounds,
            "CEASE-CERNN": 2 * remote * model_bytes * self.config.rounds,
            "DANE-CERNN": 4 * remote * model_bytes * self.config.rounds,
            "FedDANE-CERNN": 4 * remote * model_bytes * self.config.rounds,
            "one-shot FedAvg": 2 * remote * model_bytes,
            "multi-round FedAvg": 2 * remote * model_bytes * self.config.rounds,
            "FedProx": 2 * remote * model_bytes * self.config.rounds,
            "SCAFFOLD": 4 * remote * model_bytes * self.config.rounds,
        }
        for method, expected in expectations.items():
            result = fit_baseline(method, self._model(), self.clients, self.config)
            self.assertEqual(result.communication_bytes, expected)

    def test_corrected_surrogate_methods_run_and_track_participation(self) -> None:
        partial = BaselineConfig(
            **{
                **self.config.__dict__,
                "participation_rate": 0.5,
                "participation_seed": 17,
            }
        )
        for method in ("CSL-CERNN", "CEASE-CERNN", "DANE-CERNN", "FedDANE-CERNN"):
            with self.subTest(method=method):
                result = fit_baseline(method, self._model(), self.clients, partial)
                self.assertTrue(np.isfinite(result.communication_bytes))
                self.assertTrue(any("active_clients" in item for item in result.history))

    def test_csl_matches_fedcernn_with_identical_host_and_solver_budgets(self) -> None:
        shared = OptimizerConfig(adam_steps=1, adam_lr=0.001, lbfgs_steps=0)
        config = BaselineConfig(
            rounds=1,
            l2_lambda=0.001,
            centralized_optimizer=shared,
            host_optimizer=shared,
            local_optimizer=shared,
        )
        fed_model = self._model()
        csl_model = self._model()
        fed = FedCERNN(
            fed_model,
            l2_lambda=config.l2_lambda,
            optimizer_config=shared,
            host_optimizer_config=shared,
        ).fit(self.clients, rounds=1)
        csl = fit_baseline("CSL-CERNN", csl_model, self.clients, config)
        torch.testing.assert_close(
            flatten_parameters(fed.model, detach=True),
            flatten_parameters(csl.model, detach=True),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(fed.communication_bytes, csl.communication_bytes)

    def test_heterogeneous_partitions_are_complete(self) -> None:
        X = np.arange(60, dtype=float).reshape(20, 3)
        y = np.arange(20, dtype=float)
        for kind in ("iid", "covariate_shift", "response_shift", "host_shift", "size_imbalance"):
            with self.subTest(kind=kind):
                pieces = _partition_indices(X, y, 4, 29, {"kind": kind})
                values = np.concatenate([piece.numpy() for piece in pieces])
                self.assertEqual(sorted(values.tolist()), list(range(20)))
                self.assertTrue(all(len(piece) > 0 for piece in pieces))

    def test_dcernn_uses_synchronous_gradient_rounds(self) -> None:
        model = self._model()
        model_bytes = parameter_count(model) * next(model.parameters()).element_size()
        result = fit_baseline("DCERNN", model, self.clients, self.config)
        steps = self.config.rounds * self.config.local_steps
        self.assertEqual(len(result.history), steps)
        self.assertTrue(all("gradient_norm" in item for item in result.history))
        self.assertEqual(
            result.communication_bytes,
            2 * (len(self.clients) - 1) * model_bytes * steps,
        )

    def test_local_optimizer_configuration_changes_federated_baselines(self) -> None:
        inactive = BaselineConfig(
            rounds=1,
            local_steps=2,
            local_lr=0.01,
            l2_lambda=0.001,
            local_optimizer=OptimizerConfig(adam_steps=0, lbfgs_steps=0),
        )
        active = BaselineConfig(
            rounds=1,
            local_steps=2,
            local_lr=0.01,
            l2_lambda=0.001,
            local_optimizer=OptimizerConfig(
                adam_steps=2,
                adam_lr=0.01,
                lbfgs_steps=0,
            ),
        )
        for method in ("one-shot FedAvg", "FedProx", "SCAFFOLD"):
            with self.subTest(method=method):
                inactive_model = self._model()
                initial = flatten_parameters(inactive_model, detach=True)
                inactive_result = fit_baseline(
                    method,
                    inactive_model,
                    self.clients,
                    inactive,
                )
                torch.testing.assert_close(
                    flatten_parameters(inactive_result.model, detach=True),
                    initial,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                active_result = fit_baseline(
                    method,
                    self._model(),
                    self.clients,
                    active,
                )
                self.assertGreater(
                    float(
                        torch.linalg.vector_norm(
                            flatten_parameters(active_result.model, detach=True)
                            - initial
                        )
                    ),
                    1.0e-8,
                )

    def test_host_warm_start_is_identical_across_oracles(self) -> None:
        central = fit_baseline(
            "Centralized CERNN", self._model(), self.clients, self.config
        )
        host = fit_baseline("Host CERNN", self._model(), self.clients, self.config)
        fed = FedCERNN(
            self._model(),
            l2_lambda=self.config.l2_lambda,
            optimizer_config=self.config.centralized_optimizer,
            host_optimizer_config=self.config.host_optimizer,
        ).fit(self.clients, rounds=0)
        self.assertIsNotNone(central.host_warm_start)
        torch.testing.assert_close(
            central.host_warm_start,
            flatten_parameters(host.model, detach=True),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            central.host_warm_start,
            flatten_parameters(fed.model, detach=True),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            [item["stage"] for item in central.history],
            ["host_warm_start", "pooled_fit"],
        )


class MEPSTests(unittest.TestCase):
    def test_locked_features_all_ages_and_training_only_target_scaling(self) -> None:
        n = 40
        frame = pd.DataFrame(
            {
                "DUPERSID": [f"P{index:03d}" for index in range(n)],
                "PERWT23F": np.r_[0.0, np.linspace(1.0, 2.0, n - 1)],
                "TOTEXP23": np.linspace(0.0, 39000.0, n),
                "AGE23X": np.arange(n) % 80,
                "FAMINC23": np.linspace(-230.0, 50000.0, n),
                "SEX": np.where(np.arange(n) % 2 == 0, 1, 2),
                "POVCAT23": np.where(np.arange(n) == 0, -1, 3),
                "HIBPDX": np.where(np.arange(n) % 3 == 0, 1, 2),
            }
        )
        splits = prepare_meps(
            frame,
            seed=7,
            feature_columns=(
                "AGE23X",
                "FAMINC23",
                "SEX",
                "POVCAT23",
                "HIBPDX",
            ),
        )
        self.assertEqual(
            len(splits.y_train)
            + len(splits.y_validation)
            + len(splits.y_test),
            n,
        )
        self.assertAlmostEqual(float(np.mean(splits.y_train)), 0.0, places=12)
        self.assertAlmostEqual(float(np.std(splits.y_train)), 1.0, places=12)
        np.testing.assert_allclose(
            splits.inverse_target(splits.y_test), splits.y_test_thousands
        )
        self.assertIn("POVCAT23__missing", splits.feature_names)
        self.assertIn("HIBPDX__missing", splits.feature_names)
        self.assertTrue(
            set(splits.train_person_ids).isdisjoint(splits.test_person_ids)
        )
        with self.assertRaises(ValueError):
            validate_meps_feature_columns(["OBTOTV23"])
        with self.assertRaises(ValueError):
            validate_meps_feature_columns(["TOTEXP23"])


class EICUTests(unittest.TestCase):
    def _patient_frame(self) -> pd.DataFrame:
        rows = []
        specifications = [
            ("A", 100, 1000, 2, 3000, "50"),
            ("A", 101, 1000, 1, 3000, "50"),
            ("B", 200, 2000, 1, 2000, "17"),
            ("C", 300, 3000, 1, 1000, "60"),
            ("D", 400, 4000, 1, 4000, "> 89"),
            ("E", 500, 5000, 1, 5000, "35"),
            ("F", 600, 6000, 1, 1000, "45"),
            ("F", 601, 6000, 2, 3000, "45"),
            # Separate encounters for one deidentified patient are both kept:
            # eICU does not identify their absolute chronological order.
            ("G", 700, 7000, 1, 3000, "55"),
            ("G", 701, 7001, 1, 3000, "55"),
        ]
        for (
            person,
            stay,
            health_system_stay,
            visit,
            discharge,
            age,
        ) in specifications:
            row = {column: 1 for column in PATIENT_COLUMNS}
            row.update(
                {
                    "uniquepid": person,
                    "patientunitstayid": stay,
                    "patienthealthsystemstayid": health_system_stay,
                    "unitvisitnumber": visit,
                    "unitdischargeoffset": discharge,
                    "age": age,
                    "gender": "Female",
                    "ethnicity": "Other",
                    "unitadmitsource": "Other",
                    "apacheadmissiondx": "diagnosis",
                    "unittype": "ICU",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def test_first_hospitalization_stay_and_24h_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._patient_frame().to_csv(root / "patient.csv", index=False)
            lab_spec = EVENT_TABLES["lab"]
            lab = pd.DataFrame(
                {
                    lab_spec.id_column: [101, 101, 101, 101],
                    lab_spec.offset_column: [-1, 0, 1440, 1441],
                    lab_spec.feature_column: ["glucose"] * 4,
                    lab_spec.value_columns[0]: [1.0, 2.0, 3.0, 4.0],
                }
            )
            lab.to_csv(root / "lab.csv", index=False)
            builder = EICU24HourBuilder(root)
            cohort = builder.build_cohort()
            self.assertEqual(
                int(cohort.loc[cohort["uniquepid"] == "A", "patientunitstayid"].iloc[0]),
                101,
            )
            self.assertEqual(
                set(
                    cohort.loc[
                        cohort["uniquepid"] == "G", "patientunitstayid"
                    ].astype(int)
                ),
                {700, 701},
            )
            self.assertNotIn("B", set(cohort["uniquepid"]))
            self.assertNotIn("C", set(cohort["uniquepid"]))
            self.assertNotIn("F", set(cohort["uniquepid"]))
            self.assertNotIn("hospitalid", cohort.columns)
            self.assertAlmostEqual(
                float(
                    cohort.loc[
                        cohort["uniquepid"] == "E",
                        "remaining_icu_los_days",
                    ].iloc[0]
                ),
                (5000.0 - 1440.0) / 1440.0,
            )
            events = list(builder.iter_24h_events("lab", stay_ids=[101]))
            self.assertEqual(sum(len(frame) for frame in events), 2)

    def test_validation_can_require_only_selected_event_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._patient_frame().to_csv(root / "patient.csv", index=False)
            pd.DataFrame(
                {
                    "patientunitstayid": [101],
                    "labresultoffset": [0],
                    "labname": ["glucose"],
                    "labresult": [1.0],
                }
            ).to_csv(root / "lab.csv", index=False)
            report = validate_eicu_directory(
                root, required_event_tables=("lab",)
            )
            self.assertIn("lab", report.files)
            self.assertIn("nursecharting", report.missing_optional_tables)
            with self.assertRaisesRegex(FileNotFoundError, "vitalperiodic"):
                validate_eicu_directory(
                    root, required_event_tables=("lab", "vitalperiodic")
                )

    def test_event_aggregation_is_correct_and_keeps_unobserved_stays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._patient_frame().to_csv(root / "patient.csv", index=False)
            pd.DataFrame(
                {
                    "patientunitstayid": [101, 101, 101, 700],
                    "labresultoffset": [10, 30, 20, 0],
                    "labname": ["glucose", "glucose", "glucose", "sodium"],
                    "labresult": [1.0, 3.0, 2.0, 7.0],
                }
            ).to_csv(root / "lab.csv", index=False)
            builder = EICU24HourBuilder(root)
            summary = builder.aggregate_24h_numeric_events(
                "lab", stay_ids=[101, 700, 999], selected_features=["glucose", "sodium"]
            )
            self.assertEqual(list(summary.index), [101, 700, 999])
            self.assertAlmostEqual(summary.loc[101, "lab__glucose__mean"], 2.0)
            self.assertAlmostEqual(summary.loc[101, "lab__glucose__minimum"], 1.0)
            self.assertAlmostEqual(summary.loc[101, "lab__glucose__maximum"], 3.0)
            self.assertAlmostEqual(summary.loc[101, "lab__glucose__last"], 3.0)
            self.assertAlmostEqual(summary.loc[101, "lab__glucose__count"], 3.0)
            self.assertTrue(np.isnan(summary.loc[999, "lab__glucose__mean"]))

    def test_protected_identifier_free_matrix_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restricted.npz"
            np.savez_compressed(
                path,
                X_train=np.ones((4, 2)),
                X_validation=np.ones((2, 2)),
                X_test=np.ones((2, 2)),
                y_train=np.arange(4.0),
                y_validation=np.arange(2.0),
                y_test=np.arange(2.0),
                feature_names=np.asarray(["a", "b"]),
                target_center=np.asarray([3.0]),
                target_scale=np.asarray([2.0]),
            )
            os.chmod(path, 0o600)
            loaded = load_restricted_eicu_matrices(path)
            self.assertEqual(loaded.X_train.shape, (4, 2))
            self.assertEqual(len(loaded.clients(2, seed=4)), 2)
            np.testing.assert_allclose(loaded.inverse_target(np.asarray([1.0])), [5.0])
            os.chmod(path, 0o644)
            with self.assertRaises(PermissionError):
                load_restricted_eicu_matrices(path)

class RegistryTests(unittest.TestCase):
    def test_result_store_prevents_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = ResultStore(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "already writing"):
                    ResultStore(directory)
            finally:
                first.close()
            second = ResultStore(directory)
            second.close()

    def test_main_sensitivity_is_one_factor_at_a_time(self) -> None:
        config_dir = Path(__file__).parents[1] / "configs" / "hierarchical_sensitivity_r100"
        scenarios = {
            path.stem: expand_scenarios(load_config(path))[0]
            for path in config_dir.glob("*.yaml")
        }
        self.assertEqual(len(scenarios), 17)
        reference = scenarios["reference"].settings

        def factors(settings):
            return (
                settings["simulation"]["n_train"],
                settings["simulation"]["n_clients"],
                tuple(settings["model"]["taus"]),
                tuple(settings["model"]["hidden_dims"]),
                settings["fedcernn"]["rounds"],
                (
                    settings["fedcernn"]["surrogate_optimizer"]["adam_steps"],
                    settings["fedcernn"]["surrogate_optimizer"]["lbfgs_steps"],
                ),
            )

        base = factors(reference)
        for name, scenario in scenarios.items():
            self.assertEqual(scenario.settings["simulation"]["dgp"], "hierarchical_composition")
            self.assertEqual(scenario.settings["simulation"]["error"], "contaminated_normal")
            difference_count = sum(a != b for a, b in zip(base, factors(scenario.settings)))
            self.assertEqual(difference_count, 0 if name == "reference" else 1, name)

    def test_explicit_simulation_pairs_do_not_form_cartesian_product(self) -> None:
        config = {
            "dataset": "simulation",
            "simulation": {
                "n_train": 24,
                "n_test": 12,
                "n_clients": 3,
                "scenarios": [
                    {"dgp": "smooth_additive", "error": "normal"},
                    {"dgp": "friedman", "error": "t3"},
                    {
                        "dgp": "sparse_interaction",
                        "error": "contaminated_normal",
                    },
                ],
            },
            "methods": ["FedCERNN"],
        }
        scenarios = expand_scenarios(config)
        self.assertEqual(
            [
                (
                    scenario.settings["simulation"]["dgp"],
                    scenario.settings["simulation"]["error"],
                )
                for scenario in scenarios
            ],
            [
                ("smooth_additive", "normal"),
                ("friedman", "t3"),
                ("sparse_interaction", "contaminated_normal"),
            ],
        )

    def test_sensitivity_only_config_can_add_one_common_reference(self) -> None:
        config = {
            "dataset": "simulation",
            "simulation": {
                "n_train": 24,
                "n_test": 12,
                "n_clients": 3,
                "dgp": "sparse_interaction",
                "error": "contaminated_normal",
                "include_primary": False,
            },
            "methods": ["FedCERNN"],
            "sensitivity": {
                "enabled": True,
                "include_reference": True,
                "reference": {
                    "simulation": {
                        "dgp": "sparse_interaction",
                        "error": "contaminated_normal",
                    }
                },
                "axes": {"simulation.n_clients": [2, 4]},
            },
        }
        scenarios = expand_scenarios(config)
        self.assertEqual(
            [scenario.kind for scenario in scenarios],
            [
                "sensitivity:reference",
                "sensitivity:simulation.n_clients",
                "sensitivity:simulation.n_clients",
            ],
        )
        self.assertEqual(
            len({_scenario_seed_key(scenario) for scenario in scenarios}),
            1,
        )

    def test_shared_dgp_seed_policy_matches_primary_and_sensitivity(self) -> None:
        settings = {
            "seed_policy": "shared_dgp_v1",
            "simulation": {
                "dgp": "sparse_interaction",
                "error": "contaminated_normal",
            },
        }
        primary = ExperimentScenario("primary-id", "primary", settings)
        sensitivity = ExperimentScenario(
            "sensitivity-id", "sensitivity:model.taus", settings
        )
        self.assertEqual(
            _scenario_seed_key(primary), _scenario_seed_key(sensitivity)
        )
        self.assertEqual(
            derive_seed(20260811, _scenario_seed_key(primary), 7),
            derive_seed(20260811, _scenario_seed_key(sensitivity), 7),
        )

    def test_shared_primary_policy_preserves_registered_primary_seed(self) -> None:
        settings = {
            "seed_policy": "shared_primary_v1",
            "simulation": {
                "dgp": "sparse_interaction",
                "error": "contaminated_normal",
            },
        }
        primary = ExperimentScenario("primary-id", "primary", settings)
        sensitivity = ExperimentScenario(
            "sensitivity-id", "sensitivity:model.hidden_dims", settings
        )
        expected_key = "primary:sparse_interaction:contaminated_normal"
        self.assertEqual(_scenario_seed_key(primary), expected_key)
        self.assertEqual(_scenario_seed_key(sensitivity), expected_key)

    def test_result_store_and_stable_seed(self) -> None:
        self.assertEqual(derive_seed(7, "abc", 2), derive_seed(7, "abc", 2))
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory)
            try:
                store.append(
                    {
                        "run_key": "scenario:0:FedCERNN",
                        "scenario_id": "scenario",
                        "repetition": 0,
                        "seed": 123,
                        "status": "success",
                        "method": "FedCERNN",
                    }
                )
                with store.csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=("run_key", "status", "method"),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "run_key": "scenario:0:FedCERNN",
                            "status": "success",
                            "method": "FedCERNN",
                        }
                    )
            finally:
                store.close()
            reloaded = ResultStore(directory)
            try:
                self.assertEqual(
                    reloaded.completed_keys(),
                    {
                        "scenario:0:FedCERNN",
                        "scenario:0:123:FedCERNN",
                    },
                )
                self.assertTrue(reloaded.csv_path.exists())
                with reloaded.csv_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    self.assertEqual(
                        tuple(csv.DictReader(handle).fieldnames or ()),
                        CSV_FIELDS,
                    )
            finally:
                reloaded.close()

    def test_round_metadata_matches_each_method_communication_process(self) -> None:
        settings = {
            "fedcernn": {"rounds": 3},
            "baselines": {"rounds": 4, "local_steps": 7},
        }
        expected = {
            "FedCERNN": 3,
            "FedCERNN-T5": 5,
            "Centralized CERNN": 0,
            "Host CERNN": 0,
            "one-shot FedAvg": 1,
            "multi-round FedAvg": 4,
            "FedProx": 4,
            "SCAFFOLD": 4,
            "DCERNN": 28,
        }
        self.assertEqual(
            {
                method: _configured_method_rounds(method, settings)
                for method in expected
            },
            expected,
        )

    def test_cli_method_filter_does_not_change_scenario_identity(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
        full = run_experiment(config_path, dry_run=True)
        subset = run_experiment(
            config_path, methods=["FedCERNN"], dry_run=True
        )
        self.assertEqual(
            [item["scenario_id"] for item in full],
            [item["scenario_id"] for item in subset],
        )

    def test_data_seed_identity_ignores_optimizer_configuration(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "meps_solver_revision.yaml"
        first = expand_scenarios(load_config(config_path))[0]
        changed_settings = dict(first.settings)
        changed_settings["fedcernn"] = {
            "rounds": 9,
            "surrogate_optimizer": {"profile": "different"},
        }
        changed = type(first)(
            scenario_id="different-full-scenario-id",
            kind=first.kind,
            settings=changed_settings,
        )
        self.assertEqual(_scenario_seed_key(first), "meps")
        self.assertEqual(
            derive_seed(2023, _scenario_seed_key(first), 4),
            derive_seed(2023, _scenario_seed_key(changed), 4),
        )

    def test_end_to_end_resume(self) -> None:
        config = {
            "experiment": {
                "repetitions": 1,
                "base_seed": 9,
                "torch_threads": 1,
                "fail_fast": True,
                "retry_errors": True,
            },
            "dataset": "simulation",
            "simulation": {
                "n_train": 32,
                "n_test": 16,
                "n_clients": 2,
                "dgps": ["smooth_additive"],
                "errors": ["normal"],
            },
            "model": {"hidden_dims": [4, 3], "taus": [0.25, 0.75]},
            "optimization": {
                "l2_lambda": 0.0001,
                "adam_steps": 1,
                "adam_lr": 0.01,
                "lbfgs_steps": 0,
                "tolerance": 1.0e-5,
            },
            "fedcernn": {"rounds": 1},
            "baselines": {
                "rounds": 1,
                "local_steps": 1,
                "local_lr": 0.01,
            },
            "methods": ["FedCERNN", "Host CERNN"],
        }
        with tempfile.TemporaryDirectory() as directory:
            first = run_experiment(config, output_directory=directory)
            second = run_experiment(config, output_directory=directory)
            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            self.assertTrue(all(record["status"] == "success" for record in second))
            rounds = {record["method"]: record["communication_rounds"] for record in second}
            self.assertEqual(rounds["FedCERNN"], 1)
            self.assertEqual(rounds["Host CERNN"], 0)
            self.assertTrue(
                all(
                    record["rounds"] == record["communication_rounds"]
                    for record in second
                )
            )

            config["experiment"]["base_seed"] = 10
            changed_seed = run_experiment(config, output_directory=directory)
            self.assertEqual(len(changed_seed), 4)
            self.assertEqual(
                {record["base_seed"] for record in changed_seed},
                {9, 10},
            )


if __name__ == "__main__":
    unittest.main()
