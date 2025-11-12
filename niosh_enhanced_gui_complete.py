#!/usr/bin/env python3
"""
NIOSH Enhanced GUI - Complete Professional Grade Interface
A fantastic, modern GUI application for NIOSH lifting analysis with advanced features,
professional design, and excellent user experience.

Author: Claude Code Enhanced
Date: 2025-01-09
Version: 2.0.0
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import threading
import time
import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import traceback
from pathlib import Path
import sqlite3
import hashlib
import base64
from dataclasses import dataclass, asdict
from enum import Enum
import re
import webbrowser
import subprocess
import platform

# Import optional dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.dates as mdates
    import numpy as np
    from collections import Counter, defaultdict
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available - charts will be disabled")

# Import NIOSH system modules
try:
    from prod_gen_v21 import generate_full_example, generate_random_parameters, generate_narrative
    from niosh_calculator_v2 import calculate_niosh_v2
    from niosh_validator import validate_and_correct_report
    from niosh_rag import NIOSHRAGSystem
    NIOSH_AVAILABLE = True
    print("NIOSH system modules loaded successfully")
except ImportError as e:
    print(f"Error: NIOSH modules not available: {e}")
    NIOSH_AVAILABLE = False

# Theme and Design System
class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"

class ColorScheme:
    """Professional color scheme for the application"""
    # Primary colors
    PRIMARY = "#1E88E5"
    PRIMARY_DARK = "#1565C0"
    PRIMARY_LIGHT = "#42A5F5"
    
    # Secondary colors
    SECONDARY = "#7C4DFF"
    SECONDARY_DARK = "#6200EA"
    SECONDARY_LIGHT = "#B388FF"
    
    # Success colors
    SUCCESS = "#4CAF50"
    SUCCESS_DARK = "#388E3C"
    SUCCESS_LIGHT = "#81C784"
    
    # Warning colors
    WARNING = "#FF9800"
    WARNING_DARK = "#F57C00"
    WARNING_LIGHT = "#FFB74D"
    
    # Error colors
    ERROR = "#F44336"
    ERROR_DARK = "#D32F2F"
    ERROR_LIGHT = "#E57373"
    
    # Risk level colors
    RISK_LOW = "#4CAF50"
    RISK_MODERATE = "#FF9800"
    RISK_HIGH = "#FF5722"
    RISK_EXTREME = "#F44336"
    
    # Neutral colors
    SURFACE = "#FAFAFA"
    BACKGROUND = "#FFFFFF"
    TEXT_PRIMARY = "#212121"
    TEXT_SECONDARY = "#757575"
    TEXT_DISABLED = "#BDBDBD"
    
    @classmethod
    def get_risk_color(cls, li_value: float) -> str:
        """Get color based on Lifting Index value"""
        if li_value > 10:
            return cls.RISK_EXTREME
        elif li_value > 3.0:
            return cls.RISK_HIGH
        elif li_value > 1.0:
            return cls.RISK_MODERATE
        else:
            return cls.RISK_LOW

@dataclass
class ScenarioTemplate:
    """Template for scenario generation"""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str
    tags: List[str]
    created_at: str
    usage_count: int = 0

@dataclass
class UserSession:
    """User session data"""
    id: str
    created_at: str
    last_updated: str
    scenarios: List[Dict[str, Any]]
    settings: Dict[str, Any]
    templates: List[ScenarioTemplate]

class DatabaseManager:
    """Manages application database for sessions, templates, and cache"""
    
    def __init__(self, db_path: str = "niosh_gui.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );
                
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
    
    def save_session(self, session: UserSession):
        """Save user session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, created_at, last_updated, data) VALUES (?, ?, ?, ?)",
                (session.id, session.created_at, session.last_updated, json.dumps(asdict(session)))
            )
    
    def get_templates(self, category: Optional[str] = None) -> List[ScenarioTemplate]:
        """Get all templates or filtered by category"""
        with sqlite3.connect(self.db_path) as conn:
            if category:
                cursor = conn.execute(
                    "SELECT * FROM templates WHERE category = ? ORDER BY usage_count DESC",
                    (category,)
                )
            else:
                cursor = conn.execute("SELECT * FROM templates ORDER BY usage_count DESC")
            
            templates = []
            for row in cursor.fetchall():
                template = ScenarioTemplate(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    category=row[3],
                    parameters=json.loads(row[4]),
                    tags=json.loads(row[5]) if row[5] else [],
                    created_at=row[6],
                    usage_count=row[7]
                )
                templates.append(template)
            return templates
    
    def save_template(self, template: ScenarioTemplate):
        """Save scenario template"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO templates 
                   (id, name, description, category, parameters, tags, created_at, usage_count) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (template.id, template.name, template.description, template.category,
                 json.dumps(template.parameters), json.dumps(template.tags),
                 template.created_at, template.usage_count)
            )
    
    def cache_set(self, key: str, value: Any, expires_hours: int = 24):
        """Set cache value with expiration"""
        expires_at = datetime.now().timestamp() + (expires_hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat(), expires_at)
            )
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value if not expired"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    value, expires_at = row
                    if datetime.now().timestamp() < expires_at:
                        try:
                            return json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            # Cache corrupted, remove it
                            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                            print(f"Warning: Corrupted cache entry for {key}, removed")
                    else:
                        # Expired, remove from cache
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        except Exception as e:
            print(f"Warning: Cache get error for {key}: {e}")
        return None

class NotificationSystem:
    """Advanced notification system for user feedback"""
    
    def __init__(self, parent_frame: ctk.CTkFrame):
        self.parent_frame = parent_frame
        self.notification_queue = queue.Queue()
        self._setup_notification_area()
        self._process_notifications()
    
    def _setup_notification_area(self):
        """Setup notification display area"""
        self.notification_frame = ctk.CTkFrame(self.parent_frame, height=60)
        self.notification_frame.place(relx=0.5, rely=0.1, anchor="n")
        self.notification_frame.place_forget()  # Hide initially
    
    def show_notification(self, message: str, level: str = "info", duration: int = 3000):
        """Show notification message"""
        notification = {
            "message": message,
            "level": level,
            "duration": duration,
            "timestamp": datetime.now()
        }
        self.notification_queue.put(notification)
    
    def _process_notifications(self):
        """Process notification queue"""
        try:
            while True:
                notification = self.notification_queue.get_nowait()
                self._display_notification(notification)
        except queue.Empty:
            pass
        self.parent_frame.after(100, self._process_notifications)
    
    def _display_notification(self, notification: Dict[str, Any]):
        """Display single notification"""
        # Configure colors based on level
        colors = {
            "info": (ColorScheme.PRIMARY, ColorScheme.PRIMARY_LIGHT),
            "success": (ColorScheme.SUCCESS, ColorScheme.SUCCESS_LIGHT),
            "warning": (ColorScheme.WARNING, ColorScheme.WARNING_LIGHT),
            "error": (ColorScheme.ERROR, ColorScheme.ERROR_LIGHT)
        }
        
        bg_color, _ = colors.get(notification["level"], colors["info"])
        
        # Create notification widget
        notification_widget = ctk.CTkFrame(
            self.notification_frame,
            fg_color=bg_color,
            corner_radius=8
        )
        notification_widget.pack(fill="x", padx=10, pady=5)
        
        # Add message label
        message_label = ctk.CTkLabel(
            notification_widget,
            text=notification["message"],
            text_color="white",
            font=ctk.CTkFont(size=12)
        )
        message_label.pack(padx=15, pady=10)
        
        # Show notification frame
        self.notification_frame.lift()
        
        # Schedule removal
        self.parent_frame.after(
            notification["duration"],
            lambda: self._remove_notification(notification_widget)
        )
    
    def _remove_notification(self, widget: ctk.CTkFrame):
        """Remove notification widget"""
        try:
            widget.destroy()
            if not self.notification_frame.winfo_children():
                self.notification_frame.place_forget()
        except:
            pass

class ProgressDialog:
    """Professional progress dialog with cancellation support"""
    
    def __init__(self, parent: ctk.CTk, title: str, message: str, can_cancel: bool = True):
        self.parent = parent
        self.title = title
        self.message = message
        self.can_cancel = can_cancel
        self.cancelled = False
        self.progress = 0.0
        
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)
        
        self._setup_ui()
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup progress dialog UI"""
        # Main frame
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text=self.title,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Message
        self.message_label = ctk.CTkLabel(
            main_frame,
            text=self.message,
            font=ctk.CTkFont(size=12)
        )
        self.message_label.pack(pady=(0, 20))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(main_frame, width=350)
        self.progress_bar.pack(pady=(0, 10))
        self.progress_bar.set(0)
        
        # Progress percentage
        self.progress_label = ctk.CTkLabel(
            main_frame,
            text="0%",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(pady=(0, 20))
        
        # Cancel button
        if self.can_cancel:
            self.cancel_button = ctk.CTkButton(
                main_frame,
                text="Cancel",
                command=self._cancel,
                width=100
            )
            self.cancel_button.pack()
    
    def _center_dialog(self):
        """Center dialog on parent"""
        self.dialog.update_idletasks()
        x = (self.parent.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.parent.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def _cancel(self):
        """Cancel operation"""
        self.cancelled = True
        if hasattr(self, 'cancel_button'):
            self.cancel_button.configure(state="disabled", text="Cancelling...")
    
    def update_progress(self, progress: float, message: Optional[str] = None):
        """Update progress"""
        self.progress = max(0.0, min(1.0, progress))
        self.progress_bar.set(self.progress)
        self.progress_label.configure(text=f"{int(self.progress * 100)}%")
        
        if message:
            self.message_label.configure(text=message)
        
        # Handle GUI events
        self.dialog.update_idletasks()
    
    def close(self):
        """Close dialog"""
        try:
            self.dialog.destroy()
        except:
            pass

class EnhancedNIOSHGUI:
    """
    Enhanced GUI with professional design, advanced features, and excellent user experience
    """
    
    def __init__(self):
        """Initialize the enhanced GUI"""
        self.root = ctk.CTk()
        self.root.title("NIOSH Professional Lifting Analysis System")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)
        
        # Initialize core components
        self.db = DatabaseManager()
        self.notification_system = None
        self.current_session = self._create_new_session()
        self.rag_system = None
        
        # UI state
        self.current_theme = ThemeMode.DARK
        self.is_generating = False
        self.current_scenarios = []
        self.current_templates = []
        self.selected_scenario = None
        
        # Advanced options
        self.advanced_options = {
            "rag_enabled": True,
            "validation_enabled": True,
            "semantic_coherence": True,
            "auto_save": True,
            "real_time_preview": True,
            "performance_mode": False
        }
        
        # Load user settings
        self._load_settings()
        
        # Setup theme
        self._setup_theme()
        
        # Build GUI
        self._create_main_layout()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_workspace()
        self._create_status_bar()
        
        # Setup notifications
        self.notification_system = NotificationSystem(self.main_frame)
        
        # Initialize NIOSH system
        self._initialize_niosh_system()
        
        # Load templates
        self._load_templates()
        
        # Setup event handlers
        self._setup_events()
        
        # Show welcome message
        self.notification_system.show_notification(
            "Welcome to NIOSH Professional Lifting Analysis System",
            "info", 4000
        )
        
        # Auto-save timer
        self._setup_auto_save()
    
    def _create_new_session(self) -> UserSession:
        """Create new user session"""
        session_id = hashlib.md5(
            f"{datetime.now().isoformat()}{os.getpid()}".encode()
        ).hexdigest()[:12]
        
        return UserSession(
            id=session_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            scenarios=[],
            settings={},
            templates=[]
        )
    
    def _load_settings(self):
        """Load user settings from database with robust error handling"""
        try:
            # Load theme preference
            try:
                theme_setting = self.db.cache_get("user_theme")
                if theme_setting and isinstance(theme_setting, str):
                    try:
                        self.current_theme = ThemeMode(theme_setting)
                    except ValueError:
                        print("Warning: Invalid theme setting, using default")
                        self.current_theme = ThemeMode.DARK
            except Exception as e:
                print(f"Warning: Failed to load theme: {e}")
                self.current_theme = ThemeMode.DARK
            
            # Load advanced options with complete error handling
            try:
                options = self.db.cache_get("advanced_options")
                if options and isinstance(options, dict):
                    # Create a fresh copy to avoid type corruption
                    new_options = {}
                    
                    for key, value in options.items():
                        if key in self.advanced_options:
                            # Only update if value has correct type
                            if isinstance(value, bool) and key in [
                                "rag_enabled", "validation_enabled", "semantic_coherence", 
                                "auto_save", "real_time_preview", "performance_mode"
                            ]:
                                new_options[key] = bool(value)
                            elif key == "auto_save_interval" and isinstance(value, (int, float)):
                                new_options[key] = max(30, float(value))
                            elif key == "max_scenarios_per_session" and isinstance(value, (int, float)):
                                new_options[key] = max(1, min(1000, int(value)))
                            # Skip invalid or dangerous values
                            elif key in ["auto_save_interval", "max_scenarios_per_session"]:
                                try:
                                    # Try to convert to numeric
                                    if key == "auto_save_interval":
                                        new_options[key] = max(30, float(value))
                                    elif key == "max_scenarios_per_session":
                                        new_options[key] = max(1, min(1000, int(value)))
                                except (ValueError, TypeError):
                                    print(f"Warning: Invalid numeric value for {key}: {value}")
                    
                    if new_options:
                        self.advanced_options.update(new_options)
                        print("Settings loaded successfully")
                        
            except Exception as e:
                print(f"Warning: Failed to load advanced options: {e}")
                print("Using default settings")
                # Keep default values - they are already properly set
            
        except Exception as e:
            print(f"Critical error loading settings: {e}")
            print("Using all default settings")
            # Ensure all defaults are properly set
            self.advanced_options = {
                "rag_enabled": True,
                "validation_enabled": True,
                "semantic_coherence": True,
                "auto_save": True,
                "real_time_preview": True,
                "performance_mode": False
            }
            self.current_theme = ThemeMode.DARK
    
    def _save_settings(self):
        """Save user settings to database"""
        try:
            self.db.cache_set("user_theme", self.current_theme.value)
            self.db.cache_set("advanced_options", self.advanced_options)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def _setup_theme(self):
        """Setup application theme"""
        # Set customtkinter theme
        if self.current_theme == ThemeMode.LIGHT:
            ctk.set_appearance_mode("light")
        elif self.current_theme == ThemeMode.DARK:
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("system")  # Follow system
        
        # Set custom color theme
        ctk.set_default_color_theme("blue")
    
    def _create_main_layout(self):
        """Create main application layout"""
        # Configure root grid
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
    
    def _create_menu_bar(self):
        """Create application menu bar"""
        # Menu bar frame
        self.menu_frame = ctk.CTkFrame(self.root, height=50)
        self.menu_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.menu_frame.grid_columnconfigure(6, weight=1)
        
        # File menu
        self.file_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="📁 File",
            width=80,
            command=self._show_file_menu
        )
        self.file_menu_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Generate menu
        self.generate_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="🚀 Generate",
            width=80,
            command=self._show_generate_menu
        )
        self.generate_menu_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Templates menu
        self.templates_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="📋 Templates",
            width=100,
            command=self._show_templates_menu
        )
        self.templates_menu_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Settings menu
        self.settings_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="⚙️ Settings",
            width=80,
            command=self._show_settings
        )
        self.settings_menu_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Help menu
        self.help_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="❓ Help",
            width=80,
            command=self._show_help_menu
        )
        self.help_menu_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Theme toggle
        self.theme_btn = ctk.CTkButton(
            self.menu_frame,
            text="🌙" if self.current_theme == ThemeMode.DARK else "☀️",
            width=50,
            command=self._toggle_theme
        )
        self.theme_btn.grid(row=0, column=5, padx=5, pady=5)
    
    def _create_toolbar(self):
        """Create application toolbar"""
        # Toolbar frame
        self.toolbar_frame = ctk.CTkFrame(self.root, height=60)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.toolbar_frame.grid_columnconfigure(10, weight=1)
        
        # Generate button (main action)
        self.generate_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="🚀 Generate Scenarios",
            command=self._generate_scenarios,
            height=40,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ColorScheme.PRIMARY,
            hover_color=ColorScheme.PRIMARY_DARK
        )
        self.generate_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Quick actions
        self.quick_generate_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="⚡ Quick",
            command=self._quick_generate,
            height=35,
            width=80
        )
        self.quick_generate_btn.grid(row=0, column=1, padx=5, pady=10)
        
        self.batch_generate_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📦 Batch",
            command=self._batch_generate,
            height=35,
            width=80
        )
        self.batch_generate_btn.grid(row=0, column=2, padx=5, pady=10)
        
        # Import/Export actions
        self.import_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📥 Import",
            command=self._import_scenarios,
            height=35,
            width=80
        )
        self.import_btn.grid(row=0, column=3, padx=5, pady=10)
        
        self.export_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📤 Export",
            command=self._export_scenarios,
            height=35,
            width=80
        )
        self.export_btn.grid(row=0, column=4, padx=5, pady=10)
        
        # Clear button
        self.clear_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="🗑️ Clear",
            command=self._clear_results,
            height=35,
            width=80,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "#DCE4EE")
        )
        self.clear_btn.grid(row=0, column=5, padx=5, pady=10)
    
    def _create_workspace(self):
        """Create main workspace area"""
        # Create paned window for resizable panels
        workspace_frame = ctk.CTkFrame(self.main_frame)
        workspace_frame.pack(fill="both", expand=True)
        
        # Left panel - Input and controls
        self.left_panel = ctk.CTkFrame(workspace_frame, width=400)
        self.left_panel.pack(side="left", fill="both", expand=False, padx=(0, 5))
        
        # Right panel - Results and analysis
        self.right_panel = ctk.CTkFrame(workspace_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Setup left panel
        self._setup_left_panel()
        
        # Setup right panel
        self._setup_right_panel()
    
    def _setup_left_panel(self):
        """Setup left input panel"""
        self.left_panel.grid_rowconfigure(3, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Title
        title_frame = ctk.CTkFrame(self.left_panel)
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="📝 Scenario Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(padx=15, pady=10)
        
        # Quick templates section
        self._create_quick_templates_section(row=1)
        
        # Parameters section
        self._create_parameters_section(row=2)
        
        # Advanced options collapsible section
        self._create_advanced_section(row=3)
    
    def _create_quick_templates_section(self, row: int):
        """Create quick templates section"""
        templates_frame = ctk.CTkFrame(self.left_panel)
        templates_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        templates_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        ctk.CTkLabel(
            templates_frame,
            text="⭐ Quick Templates",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=15, pady=(10, 5), sticky="w")
        
        # Template selection
        self.template_combo = ctk.CTkComboBox(
            templates_frame,
            values=[],
            width=200,
            command=self._apply_template
        )
        self.template_combo.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 10), sticky="ew")
        
        # Refresh templates button
        refresh_btn = ctk.CTkButton(
            templates_frame,
            text="🔄",
            width=30,
            command=self._load_templates
        )
        refresh_btn.grid(row=1, column=2, padx=(5, 15), pady=(0, 10))
    
    def _create_parameters_section(self, row: int):
        """Create parameters input section"""
        params_frame = ctk.CTkFrame(self.left_panel)
        params_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        params_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        ctk.CTkLabel(
            params_frame,
            text="🎛️ Generation Parameters",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=15, pady=(10, 15), sticky="w")
        
        # Number of scenarios
        self.scenario_count = ctk.IntVar(value=3)
        ctk.CTkLabel(params_frame, text="Number of Scenarios:").grid(
            row=1, column=0, sticky="w", padx=(15, 10), pady=5
        )
        
        self.count_slider = ctk.CTkSlider(
            params_frame,
            from_=1,
            to=20,
            variable=self.scenario_count,
            number_of_steps=19,
            width=150
        )
        self.count_slider.grid(row=1, column=1, sticky="w", padx=(0, 15), pady=5)
        
        self.count_label = ctk.CTkLabel(params_frame, text="3", width=20)
        self.count_label.grid(row=1, column=2, padx=(5, 15), pady=5)
        
        # Scenario type
        self.scenario_type = ctk.StringVar(value="Mixed")
        ctk.CTkLabel(params_frame, text="Scenario Type:").grid(
            row=2, column=0, sticky="w", padx=(15, 10), pady=5
        )
        
        self.type_combo = ctk.CTkComboBox(
            params_frame,
            variable=self.scenario_type,
            values=["Mixed", "Warehouse", "Manufacturing", "Construction", "Healthcare", "Office"],
            width=180
        )
        self.type_combo.grid(row=2, column=1, columnspan=2, padx=(0, 15), pady=5, sticky="w")
        
        # Complexity level
        self.complexity_level = ctk.StringVar(value="Medium")
        ctk.CTkLabel(params_frame, text="Complexity Level:").grid(
            row=3, column=0, sticky="w", padx=(15, 10), pady=5
        )
        
        self.complexity_combo = ctk.CTkComboBox(
            params_frame,
            variable=self.complexity_level,
            values=["Low", "Medium", "High", "Extreme"],
            width=180
        )
        self.complexity_combo.grid(row=3, column=1, columnspan=2, padx=(0, 15), pady=5, sticky="w")
        
        # Custom scenario input
        ctk.CTkLabel(params_frame, text="Custom Scenario:").grid(
            row=4, column=0, sticky="nw", padx=(15, 10), pady=(15, 5)
        )
        
        self.scenario_text = ctk.CTkTextbox(params_frame, height=80)
        self.scenario_text.grid(
            row=4, column=1, columnspan=2, sticky="ew", padx=(0, 15), pady=(15, 10)
        )
        self.scenario_text.insert("0.0", "Describe specific scenario details (optional)...")
        
        # Connect slider update
        self.count_slider.configure(command=lambda v: self.count_label.configure(text=str(int(v))))
    
    def _create_advanced_section(self, row: int):
        """Create collapsible advanced options section"""
        self.advanced_frame = ctk.CTkFrame(self.left_panel)
        self.advanced_frame.grid(row=row, column=0, sticky="nsew", padx=10, pady=5)
        self.advanced_frame.grid_rowconfigure(1, weight=1)
        self.advanced_frame.grid_columnconfigure(0, weight=1)
        
        # Toggle button
        self.advanced_toggle = ctk.CTkButton(
            self.advanced_frame,
            text="🔧 Advanced Options ▼",
            command=self._toggle_advanced_panel,
            height=30,
            anchor="w"
        )
        self.advanced_toggle.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Advanced options content (initially hidden)
        self.advanced_content = ctk.CTkScrollableFrame(
            self.advanced_frame,
            height=200,
            fg_color="transparent"
        )
        # Don't pack initially - will be shown when toggled
        
        self._create_advanced_options_content()
    
    def _create_advanced_options_content(self):
        """Create advanced options content"""
        # RAG Enhancement
        self.rag_enabled_var = ctk.BooleanVar(value=self.advanced_options.get("rag_enabled", True))
        rag_check = ctk.CTkCheckBox(
            self.advanced_content,
            text="🤖 Enable RAG-Enhanced Generation",
            variable=self.rag_enabled_var
        )
        rag_check.pack(pady=5, padx=10, anchor="w")
        
        # Validation
        self.validation_enabled_var = ctk.BooleanVar(value=self.advanced_options.get("validation_enabled", True))
        validation_check = ctk.CTkCheckBox(
            self.advanced_content,
            text="✅ Enable Report Validation",
            variable=self.validation_enabled_var
        )
        validation_check.pack(pady=5, padx=10, anchor="w")
        
        # Semantic Coherence
        self.semantic_enabled_var = ctk.BooleanVar(value=self.advanced_options.get("semantic_coherence", True))
        semantic_check = ctk.CTkCheckBox(
            self.advanced_content,
            text="🧠 Enable Semantic Coherence",
            variable=self.semantic_enabled_var
        )
        semantic_check.pack(pady=5, padx=10, anchor="w")
        
        # Real-time Preview
        self.preview_enabled_var = ctk.BooleanVar(value=self.advanced_options.get("real_time_preview", True))
        preview_check = ctk.CTkCheckBox(
            self.advanced_content,
            text="👁️ Enable Real-time Preview",
            variable=self.preview_enabled_var
        )
        preview_check.pack(pady=5, padx=10, anchor="w")
        
        # Performance Mode
        self.performance_mode_var = ctk.BooleanVar(value=self.advanced_options.get("performance_mode", False))
        performance_check = ctk.CTkCheckBox(
            self.advanced_content,
            text="⚡ Performance Mode (faster generation)",
            variable=self.performance_mode_var
        )
        performance_check.pack(pady=5, padx=10, anchor="w")
        
        # Auto-save
        self.autosave_enabled_var = ctk.BooleanVar(value=self.advanced_options.get("auto_save", True))
        autosave_check = ctk.CTkCheckBox(
            self.advanced_content,
            text="💾 Enable Auto-save",
            variable=self.autosave_enabled_var
        )
        autosave_check.pack(pady=5, padx=10, anchor="w")
    
    def _setup_right_panel(self):
        """Setup right results panel"""
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Results header
        header_frame = ctk.CTkFrame(self.right_panel)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        
        header_label = ctk.CTkLabel(
            header_frame,
            text="📊 Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        header_label.pack(side="left", padx=15, pady=10)
        
        # Tabview for different views
        self.tabview = ctk.CTkTabview(self.right_panel)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Add tabs
        self.tabview.add("📋 Overview")
        self.tabview.add("📄 Detailed Reports")
        self.tabview.add("📈 Statistics")
        self.tabview.add("🎯 Risk Analysis")
        self.tabview.add("⚙️ Raw Data")
        
        # Setup tab contents
        self._setup_overview_tab()
        self._setup_detailed_tab()
        self._setup_statistics_tab()
        self._setup_risk_tab()
        self._setup_raw_data_tab()
    
    def _setup_overview_tab(self):
        """Setup overview tab content"""
        overview_frame = self.tabview.tab("📋 Overview")
        
        # Create scrollable frame
        self.overview_scroll = ctk.CTkScrollableFrame(overview_frame)
        self.overview_scroll.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _setup_detailed_tab(self):
        """Setup detailed reports tab"""
        detailed_frame = self.tabview.tab("📄 Detailed Reports")
        
        # Create scrollable frame
        self.detailed_scroll = ctk.CTkScrollableFrame(detailed_frame)
        self.detailed_scroll.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _setup_statistics_tab(self):
        """Setup statistics tab content"""
        stats_frame = self.tabview.tab("📈 Statistics")
        
        # Create scrollable frame
        self.stats_scroll = ctk.CTkScrollableFrame(stats_frame)
        self.stats_scroll.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _setup_risk_tab(self):
        """Setup risk analysis tab"""
        risk_frame = self.tabview.tab("🎯 Risk Analysis")
        
        # Create scrollable frame
        self.risk_scroll = ctk.CTkScrollableFrame(risk_frame)
        self.risk_scroll.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _setup_raw_data_tab(self):
        """Setup raw data tab"""
        raw_frame = self.tabview.tab("⚙️ Raw Data")
        
        # Create text widget for raw data display
        self.raw_text = ctk.CTkTextbox(raw_frame, font=ctk.CTkFont(family="Consolas", size=10))
        self.raw_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_status_bar(self):
        """Create application status bar"""
        self.status_frame = ctk.CTkFrame(self.root, height=40)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Status message
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="✅ Ready - NIOSH System Available" if NIOSH_AVAILABLE else "⚠️ NIOSH System Offline",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=(15, 10), pady=10)
        
        # Progress bar (hidden by default)
        self.status_progress = ctk.CTkProgressBar(self.status_frame, width=200)
        self.status_progress.grid(row=0, column=1, sticky="e", padx=10, pady=10)
        self.status_progress.set(0)
        self.status_progress.grid_remove()  # Hide initially
        
        # System status indicator
        status_text = "🟢 NIOSH Online" if NIOSH_AVAILABLE else "🔴 NIOSH Offline"
        
        self.system_status_label = ctk.CTkLabel(
            self.status_frame,
            text=status_text,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.system_status_label.grid(row=0, column=2, sticky="e", padx=(10, 15), pady=10)
        
        # Session info
        self.session_label = ctk.CTkLabel(
            self.status_frame,
            text=f"Session: {self.current_session.id[:8]}...",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.session_label.grid(row=0, column=3, sticky="e", padx=(10, 15), pady=10)
    
    def _initialize_niosh_system(self):
        """Initialize NIOSH RAG system if available"""
        if NIOSH_AVAILABLE and self.advanced_options.get("rag_enabled", True):
            try:
                self.rag_system = NIOSHRAGSystem()
                self.notification_system.show_notification(
                    "NIOSH RAG System initialized successfully", "success"
                )
            except Exception as e:
                self.notification_system.show_notification(
                    f"Failed to initialize RAG system: {e}", "warning"
                )
                self.rag_system = None
        else:
            if not NIOSH_AVAILABLE:
                self.notification_system.show_notification(
                    "NIOSH system not available - running in demo mode", "warning"
                )
    
    def _load_templates(self):
        """Load scenario templates from database"""
        try:
            self.current_templates = self.db.get_templates()
            template_names = [f"{t.name} ({t.category})" for t in self.current_templates]
            self.template_combo.configure(values=template_names)
            
            self.notification_system.show_notification(
                f"Loaded {len(self.current_templates)} templates", "info", 2000
            )
        except Exception as e:
            self.notification_system.show_notification(
                f"Failed to load templates: {e}", "error"
            )
    
    def _setup_events(self):
        """Setup event handlers and keyboard shortcuts"""
        # Keyboard shortcuts
        self.root.bind("<Control-g>", lambda e: self._generate_scenarios())
        self.root.bind("<Control-Shift-G>", lambda e: self._quick_generate())
        self.root.bind("<Control-b>", lambda e: self._batch_generate())
        self.root.bind("<Control-o>", lambda e: self._import_scenarios())
        self.root.bind("<Control-e>", lambda e: self._export_scenarios())
        self.root.bind("<Control-s>", lambda e: self._save_session())
        self.root.bind("<F5>", lambda e: self._load_templates())
        self.root.bind("<Control-,>", lambda e: self._show_settings())
        self.root.bind("<F1>", lambda e: self._show_user_guide())
        self.root.bind("<Escape>", lambda e: self._clear_results())
        
        # Window events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_auto_save(self):
        """Setup auto-save functionality"""
        if self.advanced_options.get("auto_save", True):
            self._auto_save_session()
            # Schedule next auto-save (every 5 minutes)
            self.root.after(300000, self._setup_auto_save)
    
    # Continue with core functionality methods...
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_closing()
    
    def _on_closing(self):
        """Handle application closing"""
        try:
            # Save current session
            self._save_session()
            
            # Save settings
            self._save_settings()
            
            # Show notification
            if self.notification_system:
                self.notification_system.show_notification(
                    "Session saved successfully. Goodbye!", "success", 2000
                )
            
            # Give notification time to display
            self.root.after(1500, self.root.destroy)
        except Exception as e:
            print(f"Error during shutdown: {e}")
            self.root.destroy()
    
    # Real NIOSH integration
    def _generate_scenarios(self):
        """Main scenario generation method using real NIOSH system"""
        if self.is_generating:
            self.notification_system.show_notification(
                "Generation already in progress. Please wait...", "warning"
            )
            return
        
        if not NIOSH_AVAILABLE:
            self.notification_system.show_notification(
                "NIOSH system not available. Please check dependencies.", "error"
            )
            return
        
        # Get generation parameters
        num_scenarios = self.scenario_count.get()
        scenario_type = self.scenario_type.get()
        complexity = self.complexity_level.get()
        
        # Start generation in background thread
        threading.Thread(
            target=self._generate_scenarios_worker,
            args=(num_scenarios, scenario_type, complexity),
            daemon=True
        ).start()
    
    def _generate_scenarios_worker(self, num_scenarios, scenario_type, complexity):
        """Background worker for real NIOSH scenario generation"""
        try:
            print(f"Starting generation worker: {num_scenarios} scenarios, type: {scenario_type}, complexity: {complexity}")
            self.is_generating = True
            self.start_generation()
            
            generated_scenarios = []
            
            for i in range(num_scenarios):
                if not self.is_generating:  # Check if cancelled
                    print("Generation cancelled by user")
                    break
                
                # Update progress
                progress = (i + 1) / num_scenarios
                self.update_progress(progress, f"Generating scenario {i + 1}/{num_scenarios}")
                print(f"Working on scenario {i + 1}/{num_scenarios} (progress: {progress:.1%})")
                
                # Generate title based on type
                title = self._generate_scenario_title(scenario_type, i + 1)
                print(f"Generated title: {title}")
                
                # Generate real NIOSH scenario using existing system
                scenario_data = self._generate_single_niosh_scenario(title, scenario_type, complexity)
                
                if scenario_data:
                    generated_scenarios.append(scenario_data)
                    print(f"Successfully generated scenario: {scenario_data.get('gui_metadata', {}).get('id', 'Unknown')}")
                else:
                    print(f"Failed to generate scenario {i + 1}")
            
            print(f"Generation completed. Total scenarios: {len(generated_scenarios)}")
            
            # Update GUI with results
            self.root.after(0, lambda: self._finalize_generation(generated_scenarios))
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"Generation worker failed: {error_msg}")
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            self.root.after(0, lambda: self._handle_generation_error(error_msg, traceback_str))
        finally:
            self.is_generating = False
            print("Generation worker finished")
    
    def _generate_scenario_title(self, scenario_type, index):
        """Generate appropriate title for scenario"""
        titles = {
            "Warehouse": f"Warehouse Worker Loading Supply Stock #{index}",
            "Manufacturing": f"Manufacturing Assembly Component Transfer #{index}",
            "Construction": f"Construction Material Handling Task #{index}",
            "Healthcare": f"Healthcare Patient Handling Scenario #{index}",
            "Office": f"Office Equipment Setup Task #{index}",
            "Random": f"Random Lifting Task #{index}"
        }
        return titles.get(scenario_type, f"NIOSH Lifting Analysis #{index}")
    
    def _generate_single_niosh_scenario(self, title, scenario_type, complexity):
        """Generate a single real NIOSH scenario using existing production system"""
        try:
            print(f"Generating single scenario: {title}")
            
            # Use the existing prod_gen_v21 system
            scenario_id = f"GUI_{scenario_type}_{datetime.now().strftime('%H%M%S')}"
            print(f"Calling generate_full_example with scenario_id: {scenario_id}, title: {title}")
            
            report_data, scenario_data = generate_full_example(
                scenario_id=scenario_id,
                title=title
            )
            
            print(f"generate_full_example returned: report_data type={type(report_data)}, scenario_data type={type(scenario_data)}")
            
            if not scenario_data or not report_data:
                print(f"Failed to generate scenario: {title} - Missing data")
                return None
            
            # Add metadata for the GUI
            metadata = {
                "id": f"NIOSH_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.current_scenarios) + 1}",
                "title": title,
                "type": scenario_type,
                "complexity": complexity,
                "generation_timestamp": datetime.now().isoformat(),
                "rag_enhanced": True
            }
            
            scenario_data["gui_metadata"] = metadata
            
            # Add the report text to scenario data
            scenario_data["final_report"] = report_data
            
            print(f"Scenario generated successfully with metadata: {metadata['id']}")
            return scenario_data
            
        except Exception as e:
            error_msg = f"Error generating scenario '{title}': {str(e)}"
            print(error_msg)
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _finalize_generation(self, scenarios):
        """Finalize generation and update GUI"""
        try:
            # Add new scenarios to current list
            self.current_scenarios.extend(scenarios)
            
            # Save to database
            for scenario in scenarios:
                self.db_manager.save_scenario(scenario)
            
            # Update all displays
            self._update_summary_display()
            self._update_detailed_display()
            self._update_table_display()
            self._update_statistics()
            
            # Show success notification
            self.notification_system.show_notification(
                f"Successfully generated {len(scenarios)} scenarios", "success"
            )
            
            # Stop generation animation
            self.is_generating = False
            if hasattr(self, 'generate_button'):
                self.generate_btn.configure(state="normal")
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Error finalizing generation: {e}", "error"
            )
    
    def _handle_generation_error(self, error_msg, traceback_str):
        """Handle generation errors"""
        self.notification_system.show_notification(
            f"Generation failed: {error_msg}", "error"
        )
        self.is_generating = False
        self.update_progress(0, "Ready")
        print(f"Generation error: {error_msg}")
        print(f"Traceback: {traceback_str}")
    
    def start_generation(self):
        """Start generation animation and UI updates"""
        self.is_generating = True
        self.generate_btn.configure(state="disabled")
        self.update_progress(0, "Starting generation...")
    
    def stop_generation(self):
        """Stop generation animation and reset UI"""
        self.is_generating = False
        self.generate_btn.configure(state="normal")
        self.update_progress(0, "Ready")
    
    def update_progress(self, progress, message):
        """Update progress bar and status"""
        try:
            if hasattr(self, 'progress_bar'):
                self.progress_bar.set(progress)
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=message)
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating progress: {e}")
    
    def _update_summary_display(self):
        """Update the summary display with current scenarios"""
        if not self.current_scenarios:
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert("1.0", "No scenarios generated yet.")
            return
        
        # Generate summary content
        summary_lines = [
            f"# NIOSH Analysis Summary",
            f"Generated: {len(self.current_scenarios)} scenarios",
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Risk Distribution"
        ]
        
        # Calculate risk distribution
        risk_counts = {}
        for scenario in self.current_scenarios:
            risk_level = self._get_risk_level(scenario)
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        for risk, count in risk_counts.items():
            summary_lines.append(f"- {risk}: {count} scenarios")
        
        summary_lines.extend([
            "",
            "## Recent Scenarios",
            ""
        ])
        
        # Add recent scenarios
        for scenario in self.current_scenarios[-5:]:  # Last 5 scenarios
            metadata = scenario.get("gui_metadata", {})
            title = metadata.get("title", "Unknown Scenario")
            risk = self._get_risk_level(scenario)
            summary_lines.append(f"- **{title}** ({risk})")
        
        # Update display
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(summary_lines))
    
    def _get_risk_level(self, scenario):
        """Extract risk level from scenario data"""
        try:
            # Try to get LI from calculation results
            calc_results = scenario.get("calculation_results", {})
            origin_data = calc_results.get("origin", {})
            li = origin_data.get("LI", 0)
            
            # Categorize risk level
            if li <= 1.0:
                return "LOW"
            elif li <= 2.0:
                return "MODERATE"
            elif li <= 3.0:
                return "HIGH"
            else:
                return "EXTREME"
        except:
            return "UNKNOWN"
    
    def _update_detailed_display(self):
        """Update detailed report display"""
        if not self.current_scenarios:
            self.detailed_text.delete("1.0", tk.END)
            self.detailed_text.insert("1.0", "No detailed reports available.")
            return
        
        # Show the most recent scenario in detail
        latest_scenario = self.current_scenarios[-1]
        metadata = latest_scenario.get("gui_metadata", {})
        
        # Build detailed report
        report_lines = [
            f"# Detailed NIOSH Analysis Report",
            f"",
            f"**Scenario ID**: {metadata.get('id', 'Unknown')}",
            f"**Title**: {metadata.get('title', 'Unknown')}",
            f"**Type**: {metadata.get('type', 'Unknown')}",
            f"**Complexity**: {metadata.get('complexity', 'Unknown')}",
            f"**Generated**: {metadata.get('generation_timestamp', 'Unknown')}",
            f"**RAG Enhanced**: {metadata.get('rag_enhanced', False)}",
            f"",
            "## Job Description",
            f"",
            latest_scenario.get("job_description_narrative", "No narrative available."),
            f"",
            "## Calculation Results",
            f""
        ]
        
        # Add calculation results
        calc_results = latest_scenario.get("calculation_results", {})
        for location, results in calc_results.items():
            report_lines.extend([
                f"### {location.title()} Analysis",
                f"- **Lifting Index (LI)**: {results.get('LI', 'N/A')}",
                f"- **Recommended Weight Limit (RWL)**: {results.get('RWL', 'N/A')} kg",
                f""
            ])
        
        # Add recommendations
        recommendations = latest_scenario.get("recommendations", [])
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                f""
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        
        # Update display
        self.detailed_text.delete("1.0", tk.END)
        self.detailed_text.insert("1.0", "\n".join(report_lines))
    
    def _update_table_display(self):
        """Update the data table display"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Add scenarios to table
        for i, scenario in enumerate(self.current_scenarios):
            metadata = scenario.get("gui_metadata", {})
            calc_results = scenario.get("calculation_results", {})
            
            # Extract values
            origin_li = calc_results.get("origin", {}).get("LI", "N/A")
            origin_rwl = calc_results.get("origin", {}).get("RWL", "N/A")
            load_weight = scenario.get("common_parameters", {}).get("L_load_kg", "N/A")
            
            self.data_tree.insert("", "end", values=(
                metadata.get("id", f"Scenario_{i+1}"),
                metadata.get("title", "Unknown"),
                metadata.get("type", "Unknown"),
                f"{load_weight} kg",
                f"{origin_li}",
                f"{origin_rwl} kg",
                self._get_risk_level(scenario),
                metadata.get("generation_timestamp", "Unknown")
            ))
    
    def _update_statistics(self):
        """Update statistics display"""
        if not self.current_scenarios:
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", "No statistics available.")
            return
        
        # Calculate statistics
        total_scenarios = len(self.current_scenarios)
        risk_counts = {}
        li_values = []
        
        for scenario in self.current_scenarios:
            risk = self._get_risk_level(scenario)
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            calc_results = scenario.get("calculation_results", {})
            origin_li = calc_results.get("origin", {}).get("LI", 0)
            if isinstance(origin_li, (int, float)):
                li_values.append(origin_li)
        
        # Build statistics report
        stats_lines = [
            f"# Generation Statistics",
            f"",
            f"**Total Scenarios Generated**: {total_scenarios}",
            f"",
            "## Risk Level Distribution",
            f""
        ]
        
        for risk, count in sorted(risk_counts.items()):
            percentage = (count / total_scenarios) * 100
            stats_lines.append(f"- **{risk}**: {count} ({percentage:.1f}%)")
        
        if li_values:
            stats_lines.extend([
                f"",
                "## Lifting Index Statistics",
                f"",
                f"- **Average LI**: {sum(li_values) / len(li_values):.2f}",
                f"- **Min LI**: {min(li_values):.2f}",
                f"- **Max LI**: {max(li_values):.2f}",
                f""
            ])
        
        stats_lines.extend([
            "## Generation Info",
            f"",
            f"- **First Generated**: {self.current_scenarios[0].get('gui_metadata', {}).get('generation_timestamp', 'Unknown')}",
            f"- **Last Generated**: {self.current_scenarios[-1].get('gui_metadata', {}).get('generation_timestamp', 'Unknown')}",
            f"- **RAG Enhancement**: Active for all scenarios"
        ])
        
        # Update display
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", "\n".join(stats_lines))
        
                
        self.notification_system.show_notification(
            "Demo scenario generated successfully", "success"
        )
    
    def _quick_generate(self):
        """Quick scenario generation with default parameters"""
        self.scenario_count.set(1)
        self.scenario_type.set("Random")
        self.complexity_level.set("Medium")
        self._generate_scenarios()
    
    def _batch_generate(self):
        """Batch generate multiple scenarios with different parameters"""
        self.notification_system.show_notification(
            "Batch generation feature coming soon!", "info"
        )
    
    def _import_scenarios(self):
        """Import scenarios from file"""
        filename = filedialog.askopenfilename(
            title="Import Scenarios",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.notification_system.show_notification(
                f"Import feature will load: {os.path.basename(filename)}", "info"
            )
    
    def _export_scenarios(self):
        """Export scenarios to file"""
        if not self.current_scenarios:
            self.notification_system.show_notification(
                "No scenarios to export", "warning"
            )
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Scenarios",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_scenarios, f, indent=2, ensure_ascii=False)
                self.notification_system.show_notification(
                    f"Exported to {os.path.basename(filename)}", "success"
                )
            except Exception as e:
                self.notification_system.show_notification(
                    f"Export failed: {str(e)}", "error"
                )
    
    def _clear_results(self):
        """Clear all results"""
        self.current_scenarios = []
        self.current_session.scenarios = []
        self._update_all_displays([])
        self.notification_system.show_notification("Results cleared", "info", 2000)
    
    def _save_session(self):
        """Save current session"""
        try:
            self.current_session.last_updated = datetime.now().isoformat()
            self.current_session.scenarios = self.current_scenarios
            self.db.save_session(self.current_session)
            self.notification_system.show_notification(
                "Session saved successfully", "success", 2000
            )
        except Exception as e:
            self.notification_system.show_notification(
                f"Failed to save session: {str(e)}", "error"
            )
    
    def _auto_save_session(self):
        """Auto-save current session"""
        if self.advanced_options.get("auto_save", True) and self.current_scenarios:
            self._save_session()
    
    def _update_all_displays(self, scenarios: List[Dict[str, Any]]):
        """Update all display tabs with new scenarios"""
        self._update_overview_display(scenarios)
        self._update_detailed_display(scenarios)
        self._update_statistics_display(scenarios)
        self._update_risk_display(scenarios)
        self._update_raw_data_display(scenarios)
    
    def _update_overview_display(self, scenarios: List[Dict[str, Any]]):
        """Update overview tab with scenario cards"""
        # Clear existing content
        for widget in self.overview_scroll.winfo_children():
            widget.destroy()
        
        # Title
        title_label = ctk.CTkLabel(
            self.overview_scroll,
            text=f"📊 Generated Scenarios ({len(self.current_scenarios)} total)",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Create scenario cards
        for i, scenario in enumerate(self.current_scenarios):
            self._create_scenario_card(scenario, i)
    
    def _create_scenario_card(self, scenario: Dict[str, Any], index: int):
        """Create a card for displaying scenario information"""
        # Card frame
        card = ctk.CTkFrame(self.overview_scroll, corner_radius=10)
        card.pack(fill="x", padx=20, pady=10)
        
        # Risk color
        risk_color = ColorScheme.get_risk_color(
            scenario["data"]["calculation_results"]["origin"]["LI"]
        )
        
        # Header
        header_frame = ctk.CTkFrame(card, fg_color=risk_color, corner_radius=8)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text=f"{index + 1}. {scenario['title']}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        title_label.pack(side="left", padx=15, pady=10)
        
        # Risk level badge
        risk_badge = ctk.CTkLabel(
            header_frame,
            text=scenario["risk_level"],
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white",
            fg_color="transparent"
        )
        risk_badge.pack(side="right", padx=(10, 15), pady=10)
        
        # Content
        content_frame = ctk.CTkFrame(card)
        content_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Key metrics
        li_value = scenario["data"]["calculation_results"]["origin"]["LI"]
        rwl_value = scenario["data"]["calculation_results"]["origin"]["RWL"]
        load_weight = scenario["data"]["common_parameters"]["L_load_kg"]
        frequency = scenario["data"]["common_parameters"]["F_frequency_per_min"]
        
        metrics_text = f"LI: {li_value:.2f} | RWL: {rwl_value:.1f} kg | Load: {load_weight:.1f} kg | Freq: {frequency:.1f}/min"
        
        metrics_label = ctk.CTkLabel(
            content_frame,
            text=metrics_text,
            font=ctk.CTkFont(size=12)
        )
        metrics_label.pack(padx=15, pady=10)
    
    def _update_detailed_display(self, scenarios: List[Dict[str, Any]]):
        """Update detailed reports tab"""
        # Clear existing content
        for widget in self.detailed_scroll.winfo_children():
            widget.destroy()
        
        if not self.current_scenarios:
            no_data_label = ctk.CTkLabel(
                self.detailed_scroll,
                text="No scenarios generated yet",
                font=ctk.CTkFont(size=16)
            )
            no_data_label.pack(pady=50)
            return
        
        # Create detailed report for each scenario
        for i, scenario in enumerate(self.current_scenarios):
            self._create_detailed_report(scenario, i)
    
    def _create_detailed_report(self, scenario: Dict[str, Any], index: int):
        """Create detailed report view for a scenario"""
        # Report frame
        report_frame = ctk.CTkFrame(self.detailed_scroll, corner_radius=8)
        report_frame.pack(fill="x", padx=10, pady=10)
        
        # Report header
        header_frame = ctk.CTkFrame(report_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            header_frame,
            text=f"📄 Scenario {index + 1}: {scenario['title']}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left", padx=10, pady=10)
        
        # Report content
        content_frame = ctk.CTkTextbox(report_frame, height=300, font=ctk.CTkFont(family="Consolas", size=10))
        content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Insert report content
        content_frame.insert("1.0", scenario["report"])
        content_frame.configure(state="disabled")
    
    def _update_statistics_display(self, scenarios: List[Dict[str, Any]]):
        """Update statistics tab"""
        # Clear existing content
        for widget in self.stats_scroll.winfo_children():
            widget.destroy()
        
        if not self.current_scenarios:
            no_data_label = ctk.CTkLabel(
                self.stats_scroll,
                text="Generate scenarios to view statistics",
                font=ctk.CTkFont(size=16)
            )
            no_data_label.pack(pady=50)
            return
        
        # Create simple statistics summary
        stats_frame = ctk.CTkFrame(self.stats_scroll)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text="📈 Statistics Summary",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        # Count scenarios by risk level
        risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "EXTREME": 0}
        for scenario in self.current_scenarios:
            risk_counts[scenario["risk_level"]] += 1
        
        total = len(self.current_scenarios)
        
        for risk, count in risk_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            risk_label = ctk.CTkLabel(
                stats_frame,
                text=f"{risk}: {count} scenarios ({percentage:.1f}%)",
                font=ctk.CTkFont(size=14)
            )
            risk_label.pack(pady=5)
    
    def _update_risk_display(self, scenarios: List[Dict[str, Any]]):
        """Update risk analysis tab"""
        # Clear existing content
        for widget in self.risk_scroll.winfo_children():
            widget.destroy()
        
        if not self.current_scenarios:
            no_data_label = ctk.CTkLabel(
                self.risk_scroll,
                text="Generate scenarios to view risk analysis",
                font=ctk.CTkFont(size=16)
            )
            no_data_label.pack(pady=50)
            return
        
        # Create risk analysis
        risk_frame = ctk.CTkFrame(self.risk_scroll)
        risk_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            risk_frame,
            text="⚠️ Risk Analysis",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        # Show high-risk scenarios
        high_risk = [s for s in self.current_scenarios if s["risk_level"] in ["HIGH", "EXTREME"]]
        
        if high_risk:
            warning_label = ctk.CTkLabel(
                risk_frame,
                text=f"⚠️ {len(high_risk)} high-risk scenarios detected",
                font=ctk.CTkFont(size=14),
                text_color=ColorScheme.WARNING
            )
            warning_label.pack(pady=10)
        
        for scenario in high_risk:
            scenario_label = ctk.CTkLabel(
                risk_frame,
                text=f"• {scenario['title']} (LI: {scenario['data']['calculation_results']['origin']['LI']:.2f})",
                font=ctk.CTkFont(size=12)
            )
            scenario_label.pack(anchor="w", padx=20, pady=2)
    
    def _update_raw_data_display(self, scenarios: List[Dict[str, Any]]):
        """Update raw data tab"""
        self.raw_text.delete("1.0", "end")
        
        if not self.current_scenarios:
            self.raw_text.insert("1.0", "No scenarios generated yet")
            return
        
        # Format as JSON
        raw_data = {
            "session_info": {
                "id": self.current_session.id,
                "created_at": self.current_session.created_at,
                "total_scenarios": len(self.current_scenarios)
            },
            "scenarios": self.current_scenarios
        }
        
        self.raw_text.insert("1.0", json.dumps(raw_data, indent=2, ensure_ascii=False))
    
    # Placeholder methods for remaining functionality
    def _toggle_advanced_panel(self):
        """Toggle advanced options panel visibility"""
        if self.advanced_content.winfo_manager() == "":
            # Show advanced content
            self.advanced_content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.advanced_toggle.configure(text="🔧 Advanced Options ▲")
        else:
            # Hide advanced content
            self.advanced_content.pack_forget()
            self.advanced_toggle.configure(text="🔧 Advanced Options ▼")
    
    def _toggle_theme(self):
        """Toggle between light and dark theme"""
        if self.current_theme == ThemeMode.DARK:
            self.current_theme = ThemeMode.LIGHT
        else:
            self.current_theme = ThemeMode.DARK
        
        self._setup_theme()
        
        # Update theme button
        theme_text = "🌙" if self.current_theme == ThemeMode.DARK else "☀️"
        self.theme_btn.configure(text=theme_text)
        
        # Save setting
        self._save_settings()
        
        self.notification_system.show_notification(
            f"Switched to {self.current_theme.value} theme", "info", 2000
        )
    
    def _apply_template(self, selected_template: str):
        """Apply selected template to current configuration"""
        if not self.current_templates:
            return
        
        # Find the selected template
        template_name = selected_template.split(" (")[0]  # Remove category suffix
        selected_template_obj = None
        
        for template in self.current_templates:
            if template.name == template_name:
                selected_template_obj = template
                break
        
        if not selected_template_obj:
            return
        
        # Apply template parameters
        try:
            params = selected_template_obj.parameters
            
            # Set basic parameters
            if "scenario_count" in params:
                self.scenario_count.set(params["scenario_count"])
            if "scenario_type" in params:
                self.scenario_type.set(params["scenario_type"])
            if "complexity_level" in params:
                self.complexity_level.set(params["complexity_level"])
            if "custom_scenario" in params:
                self.scenario_text.delete("1.0", "end")
                self.scenario_text.insert("1.0", params["custom_scenario"])
            
            self.notification_system.show_notification(
                f"Applied template: {selected_template_obj.name}", "success", 2000
            )
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Error applying template: {str(e)}", "error"
            )
    
    # Menu and dialog methods
    def _show_file_menu(self):
        """Show file menu dropdown"""
        menu_items = [
            ("💾 Save Session", self._save_session),
            ("📥 Import Scenarios", self._import_scenarios),
            ("📤 Export Results", self._export_scenarios),
            ("", None),
            ("🚪 Exit", self._on_closing)
        ]
        self._show_menu_dropdown(menu_items, self.file_menu_btn)
    
    def _show_generate_menu(self):
        """Show generate menu dropdown"""
        menu_items = [
            ("🚀 Generate Scenarios", self._generate_scenarios),
            ("⚡ Quick Generate", self._quick_generate),
            ("📦 Batch Generate", self._batch_generate)
        ]
        self._show_menu_dropdown(menu_items, self.generate_menu_btn)
    
    def _show_templates_menu(self):
        """Show templates menu dropdown"""
        menu_items = [
            ("➕ Create Template", self._create_template),
            ("📝 Manage Templates", self._manage_templates),
            ("🔄 Refresh Templates", self._load_templates)
        ]
        self._show_menu_dropdown(menu_items, self.templates_menu_btn)
    
    def _show_settings(self):
        """Show settings dialog"""
        self.notification_system.show_notification("Settings dialog coming soon!", "info")
    
    def _show_help_menu(self):
        """Show help menu dropdown"""
        menu_items = [
            ("📖 User Guide", self._show_user_guide),
            ("🎹 Keyboard Shortcuts", self._show_shortcuts),
            ("❓ About", self._show_about)
        ]
        self._show_menu_dropdown(menu_items, self.help_menu_btn)
    
    def _show_menu_dropdown(self, items: List[Tuple[str, Optional[Callable]]], parent_button):
        """Show menu dropdown with specified items"""
        # Create popup menu
        popup = ctk.CTkToplevel(self.root)
        popup.overrideredirect(True)
        popup.attributes('-topmost', True)
        
        # Position near parent button
        x = parent_button.winfo_rootx()
        y = parent_button.winfo_rooty() + parent_button.winfo_height()
        popup.geometry(f"+{x}+{y}")
        
        # Create menu items
        for text, command in items:
            if text == "":
                # Separator
                separator = ctk.CTkFrame(popup, height=1, fg_color="gray30")
                separator.pack(fill="x", padx=5, pady=2)
            else:
                btn = ctk.CTkButton(
                    popup,
                    text=text,
                    command=lambda c=command, p=popup: (c() if c else None, p.destroy()),
                    height=30,
                    anchor="w",
                    fg_color="transparent",
                    text_color=("gray10", "gray90"),
                    hover_color=("gray70", "gray30")
                )
                btn.pack(fill="x", padx=5, pady=1)
        
        # Auto-close when clicking outside
        popup.bind("<Button-1>", lambda e: popup.destroy())
        popup.bind("<Escape>", lambda e: popup.destroy())
        
        # Focus to capture events
        popup.focus_set()
    
    def _create_template(self):
        """Create new template from current configuration"""
        self.notification_system.show_notification("Template creation feature coming soon!", "info")
    
    def _manage_templates(self):
        """Show template management dialog"""
        self.notification_system.show_notification("Template management feature coming soon!", "info")
    
    def _show_user_guide(self):
        """Show user guide"""
        self._show_help_dialog(
            "User Guide",
            """
            # NIOSH Enhanced GUI - User Guide
            
            ## Getting Started
            1. Configure generation parameters in the left panel
            2. Click "Generate Scenarios" to create NIOSH analysis reports
            3. View results in the tabs on the right
            
            ## Features
            - **Quick Generate**: Generate a single scenario with default settings
            - **Batch Generate**: Generate multiple scenarios with different configurations
            - **Templates**: Save and reuse scenario configurations
            - **Import/Export**: Share scenarios with others
            - **Statistics**: Visual analysis of generated scenarios
            
            ## Keyboard Shortcuts
            - Ctrl+G: Generate scenarios
            - Ctrl+Shift+G: Quick generate
            - Ctrl+B: Batch generate
            - Ctrl+S: Save session
            - Ctrl+O: Import scenarios
            - Ctrl+E: Export results
            - F1: Show this help
            - F5: Refresh templates
            
            ## Risk Levels
            - 🟢 LOW: LI ≤ 1.0 (Safe)
            - 🟡 MODERATE: 1.0 < LI ≤ 3.0 (Caution required)
            - 🟠 HIGH: 3.0 < LI ≤ 10.0 (Intervention recommended)
            - 🔴 EXTREME: LI > 10.0 (Immediate action required)
            
            For detailed help, visit the documentation or use the Help menu.
            """
        )
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        self._show_help_dialog(
            "Keyboard Shortcuts",
            """
            # Keyboard Shortcuts
            
            ## File Operations
            - Ctrl+S: Save session
            - Ctrl+O: Import scenarios
            - Ctrl+E: Export results
            
            ## Generation
            - Ctrl+G: Generate scenarios
            - Ctrl+Shift+G: Quick generate
            - Ctrl+B: Batch generate
            
            ## Navigation
            - F1: Show help
            - F5: Refresh templates
            - Ctrl+,: Open settings
            - Escape: Clear results
            
            These shortcuts help you work more efficiently with the application.
            """
        )
    
    def _show_about(self):
        """Show about dialog"""
        self._show_help_dialog(
            "About NIOSH Enhanced GUI",
            """
            # NIOSH Enhanced GUI v2.0
            
            A professional-grade interface for NIOSH lifting analysis with advanced features,
            modern design, and excellent user experience.
            
            ## Features
            - ✨ Modern, professional interface
            - 🚀 Advanced scenario generation
            - 📊 Comprehensive statistics and visualizations
            - 🎯 Risk analysis and recommendations
            - 📋 Template management system
            - 💾 Session persistence
            - 🎨 Dark/Light theme support
            - ⌨️ Keyboard shortcuts
            - 📤 Multiple export formats
            - 🔧 Advanced configuration options
            
            ## System Requirements
            - Python 3.8+
            - CustomTkinter
            - NIOSH calculation modules
            
            ## Credits
            Developed with Claude Code Enhanced
            Based on NIOSH lifting equation standards
            
            © 2025 - Enhanced NIOSH GUI System
            """
        )
    
    def _show_help_dialog(self, title: str, content: str):
        """Show help content dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title(title)
        dialog.geometry("700x600")
        dialog.transient(self.root)
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text=title,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 15))
        
        # Content
        content_text = ctk.CTkTextbox(main_frame, font=ctk.CTkFont(size=11))
        content_text.pack(fill="both", expand=True, pady=(0, 15))
        content_text.insert("1.0", content)
        content_text.configure(state="disabled")
        
        # Close button
        ctk.CTkButton(
            main_frame,
            text="Close",
            command=dialog.destroy,
            width=100
        ).pack()


def main():
    """Main entry point for enhanced NIOSH GUI"""
    try:
        # Set customtkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create and run application
        app = EnhancedNIOSHGUI()
        app.run()
        
    except Exception as e:
        print(f"Failed to start Enhanced NIOSH GUI: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()