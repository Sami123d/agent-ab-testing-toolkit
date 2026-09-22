"""Experiment registry — local SQLite storage for experiment history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import ExperimentReport


class ExperimentRegistry:
    """Manages persistent storage of all experiments in a local SQLite database."""

    def __init__(self, db_path: str | Path = "experiments.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Create the schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    control_name TEXT,
                    treatment_name TEXT,
                    num_tasks INTEGER,
                    total_cost_usd REAL,
                    elapsed_seconds REAL,
                    recommendation TEXT,
                    bayesian_p_better REAL,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id INTEGER,
                    name TEXT,
                    control_mean REAL,
                    treatment_mean REAL,
                    difference REAL,
                    relative_change REAL,
                    p_value REAL,
                    significant BOOLEAN,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            conn.commit()

    def add_report(self, report: ExperimentReport) -> int:
        """Store an ExperimentReport in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Insert experiment summary
            cursor.execute("""
                INSERT INTO experiments (
                    name, control_name, treatment_name, num_tasks, 
                    total_cost_usd, elapsed_seconds, recommendation, 
                    bayesian_p_better, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.name, 
                report.control_name, 
                report.treatment_name, 
                report.num_tasks,
                report.total_cost_usd, 
                report.elapsed_seconds, 
                report.recommendation,
                report.bayesian_p_better,
                json.dumps(report.to_dict())
            ))
            
            exp_id = cursor.lastrowid
            
            # 2. Insert individual metrics
            for mr in report.metric_results:
                cursor.execute("""
                    INSERT INTO metrics (
                        experiment_id, name, control_mean, treatment_mean, 
                        difference, relative_change, p_value, significant
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_id, mr.name, mr.control_mean, mr.treatment_mean,
                    mr.difference, mr.relative_change, mr.p_value, mr.significant
                ))
            
            conn.commit()
            return exp_id

    def list_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent experiments."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, timestamp, recommendation, bayesian_p_better, num_tasks
                FROM experiments ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_report(self, exp_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full details for a previous experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT metadata_json FROM experiments WHERE id = ?", (exp_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row['metadata_json'])
            return None
