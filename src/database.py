"""SQLite database operations."""

import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from threading import Lock

from .models import CaptureResult, MarkMeasurement, CaptureLabel, MarkLabel
from .config import get_config, StorageConfig


class Database:
    """SQLite database for storing inspection data."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_config().storage
        self.db_path = Path(self.config.database_path)
        self.lock = Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create captures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captures (
                    capture_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    
                    camera_height_mm REAL,
                    frame_brightness_avg REAL,
                    tyre_detection_confidence REAL,
                    processing_duration_ms REAL,
                    
                    total_marks_detected INTEGER,
                    red_solids INTEGER,
                    red_donuts INTEGER,
                    yellow_solids INTEGER,
                    yellow_donuts INTEGER,
                    is_empty_tyre BOOLEAN,
                    
                    full_image_path TEXT,
                    tyre_crop_path TEXT,
                    annotated_image_path TEXT,
                    
                    is_labeled BOOLEAN DEFAULT FALSE,
                    labeled_by TEXT,
                    labeled_at TEXT,
                    overall_verdict TEXT,
                    label_notes TEXT,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create marks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS marks (
                    mark_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_id TEXT NOT NULL,
                    mark_index INTEGER NOT NULL,
                    
                    auto_type TEXT,
                    auto_color TEXT,
                    
                    centroid_x INTEGER,
                    centroid_y INTEGER,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    
                    area_pixels REAL,
                    perimeter_pixels REAL,
                    estimated_diameter_mm REAL,
                    circularity REAL,
                    solidity REAL,
                    eccentricity REAL,
                    convexity REAL,
                    edge_roughness REAL,
                    
                    color_h_mean REAL,
                    color_s_mean REAL,
                    color_v_mean REAL,
                    color_h_std REAL,
                    color_s_std REAL,
                    color_v_std REAL,
                    
                    is_donut BOOLEAN,
                    inner_outer_ratio REAL,
                    
                    label_actual_type TEXT,
                    label_actual_color TEXT,
                    label_quality_rating TEXT,
                    label_defects TEXT,
                    label_matches_auto BOOLEAN,
                    
                    mark_image_path TEXT,
                    
                    FOREIGN KEY (capture_id) REFERENCES captures(capture_id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_captures_timestamp ON captures(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_captures_labeled ON captures(is_labeled)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_marks_capture ON marks(capture_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_marks_circularity ON marks(circularity)')
            
            conn.commit()
            conn.close()
    
    def save_capture(self, capture: CaptureResult) -> bool:
        """Save a capture result to the database."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Insert capture
                cursor.execute('''
                    INSERT INTO captures (
                        capture_id, timestamp, sequence_number,
                        camera_height_mm, frame_brightness_avg,
                        tyre_detection_confidence, processing_duration_ms,
                        total_marks_detected, red_solids, red_donuts,
                        yellow_solids, yellow_donuts, is_empty_tyre,
                        full_image_path, tyre_crop_path, annotated_image_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    capture.capture_id,
                    capture.timestamp.isoformat(),
                    capture.sequence_number,
                    capture.camera_height_mm,
                    capture.frame_brightness_avg,
                    capture.tyre_detection_confidence,
                    capture.processing_duration_ms,
                    capture.total_marks_detected,
                    capture.red_solids,
                    capture.red_donuts,
                    capture.yellow_solids,
                    capture.yellow_donuts,
                    capture.is_empty_tyre,
                    capture.full_image_path,
                    capture.tyre_crop_path,
                    capture.annotated_image_path
                ))
                
                # Insert marks
                for mark in capture.marks:
                    cursor.execute('''
                        INSERT INTO marks (
                            capture_id, mark_index,
                            auto_type, auto_color,
                            centroid_x, centroid_y,
                            bbox_x, bbox_y, bbox_w, bbox_h,
                            area_pixels, perimeter_pixels, estimated_diameter_mm,
                            circularity, solidity, eccentricity, convexity, edge_roughness,
                            color_h_mean, color_s_mean, color_v_mean,
                            color_h_std, color_s_std, color_v_std,
                            is_donut, inner_outer_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        capture.capture_id,
                        mark.mark_index,
                        mark.auto_type,
                        mark.auto_color,
                        mark.centroid_x,
                        mark.centroid_y,
                        mark.bbox.x,
                        mark.bbox.y,
                        mark.bbox.w,
                        mark.bbox.h,
                        mark.area_pixels,
                        mark.perimeter_pixels,
                        mark.estimated_diameter_mm,
                        mark.circularity,
                        mark.solidity,
                        mark.eccentricity,
                        mark.convexity,
                        mark.edge_roughness,
                        mark.color_metrics.h_mean,
                        mark.color_metrics.s_mean,
                        mark.color_metrics.v_mean,
                        mark.color_metrics.h_std,
                        mark.color_metrics.s_std,
                        mark.color_metrics.v_std,
                        mark.is_donut,
                        mark.inner_outer_ratio
                    ))
                
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Database error saving capture: {e}")
                return False
    
    def get_capture(self, capture_id: str) -> Optional[Dict[str, Any]]:
        """Get a capture by ID."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM captures WHERE capture_id = ?', (capture_id,))
            row = cursor.fetchone()
            
            if row is None:
                conn.close()
                return None
            
            capture = dict(row)
            
            # Get marks
            cursor.execute('SELECT * FROM marks WHERE capture_id = ? ORDER BY mark_index', 
                          (capture_id,))
            capture['marks'] = [dict(r) for r in cursor.fetchall()]
            
            conn.close()
            return capture
    
    def get_captures(self, 
                     limit: int = 100, 
                     offset: int = 0,
                     labeled_only: bool = False,
                     unlabeled_only: bool = False,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get captures with optional filtering."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM captures WHERE 1=1'
            params = []
            
            if labeled_only:
                query += ' AND is_labeled = 1'
            elif unlabeled_only:
                query += ' AND (is_labeled = 0 OR is_labeled IS NULL)'
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date)
            
            query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            captures = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return captures
    
    def get_capture_count(self, labeled_only: bool = False) -> int:
        """Get total capture count."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if labeled_only:
                cursor.execute('SELECT COUNT(*) FROM captures WHERE is_labeled = 1')
            else:
                cursor.execute('SELECT COUNT(*) FROM captures')
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
    
    def get_next_sequence_number(self) -> int:
        """Get the next sequence number for captures."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT MAX(sequence_number) FROM captures')
            result = cursor.fetchone()[0]
            
            conn.close()
            return (result or 0) + 1
    
    def update_capture_label(self, capture_id: str, label: CaptureLabel) -> bool:
        """Update labeling information for a capture."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE captures SET
                        is_labeled = ?,
                        labeled_by = ?,
                        labeled_at = ?,
                        overall_verdict = ?,
                        label_notes = ?
                    WHERE capture_id = ?
                ''', (
                    label.is_labeled,
                    label.labeled_by,
                    label.labeled_at.isoformat() if label.labeled_at else None,
                    label.overall_verdict,
                    label.label_notes,
                    capture_id
                ))
                
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Database error updating capture label: {e}")
                return False
    
    def update_mark_label(self, capture_id: str, mark_index: int, 
                          label: MarkLabel) -> bool:
        """Update labeling information for a mark."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE marks SET
                        label_actual_type = ?,
                        label_actual_color = ?,
                        label_quality_rating = ?,
                        label_defects = ?,
                        label_matches_auto = ?
                    WHERE capture_id = ? AND mark_index = ?
                ''', (
                    label.label_actual_type,
                    label.label_actual_color,
                    label.label_quality_rating,
                    label.defects_to_json(),
                    label.label_matches_auto,
                    capture_id,
                    mark_index
                ))
                
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Database error updating mark label: {e}")
                return False
    
    def get_marks_for_analytics(self, 
                                labeled_only: bool = False,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all marks with capture info for analytics."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT m.*, c.timestamp, c.is_labeled, c.overall_verdict
                FROM marks m
                JOIN captures c ON m.capture_id = c.capture_id
                WHERE 1=1
            '''
            params = []
            
            if labeled_only:
                query += ' AND c.is_labeled = 1'
            
            if start_date:
                query += ' AND c.timestamp >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND c.timestamp <= ?'
                params.append(end_date)
            
            cursor.execute(query, params)
            marks = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return marks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            stats = {}
            
            # Total captures
            cursor.execute('SELECT COUNT(*) FROM captures')
            stats['total_captures'] = cursor.fetchone()[0]
            
            # Labeled captures
            cursor.execute('SELECT COUNT(*) FROM captures WHERE is_labeled = 1')
            stats['labeled_captures'] = cursor.fetchone()[0]
            
            # Total marks
            cursor.execute('SELECT COUNT(*) FROM marks')
            stats['total_marks'] = cursor.fetchone()[0]
            
            # Marks by type
            cursor.execute('''
                SELECT auto_type, auto_color, COUNT(*) as count
                FROM marks
                GROUP BY auto_type, auto_color
            ''')
            stats['marks_by_type'] = {
                f"{row[1]}_{row[0]}": row[2] for row in cursor.fetchall()
            }
            
            # Average circularity
            cursor.execute('SELECT AVG(circularity) FROM marks')
            stats['avg_circularity'] = cursor.fetchone()[0] or 0
            
            # Empty tyres
            cursor.execute('SELECT COUNT(*) FROM captures WHERE is_empty_tyre = 1')
            stats['empty_tyres'] = cursor.fetchone()[0]
            
            conn.close()
            return stats
    
    def export_to_csv(self, filepath: str, labeled_only: bool = False) -> bool:
        """Export captures to CSV."""
        import csv
        
        captures = self.get_captures(limit=100000, labeled_only=labeled_only)
        
        if not captures:
            return False
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=captures[0].keys())
                writer.writeheader()
                writer.writerows(captures)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def export_marks_to_csv(self, filepath: str, labeled_only: bool = False) -> bool:
        """Export marks to CSV."""
        import csv
        
        marks = self.get_marks_for_analytics(labeled_only=labeled_only)
        
        if not marks:
            return False
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=marks[0].keys())
                writer.writeheader()
                writer.writerows(marks)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
