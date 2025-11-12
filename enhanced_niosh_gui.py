#!/usr/bin/env python3
"""
NIOSH Enhanced GUI - Professional Grade Interface
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

# Import NIOSH system modules
try:
    from prod_gen_v21 import generate_full_example, generate_random_parameters, generate_narrative
    from niosh_calculator_v2 import calculate_niosh_v2
    from niosh_validator import validate_and_correct_report
    from niosh_rag import NIOSHRAGSystem
    NIOSH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NIOSH modules not available: {e}")
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

class AnimationManager:
    """Manages animations and transitions"""
    
    @staticmethod
    def fade_in(widget: ctk.CTkBaseClass, duration: int = 300):
        """Simple fade in effect"""
        try:
            widget.set_opacity(0)
            steps = duration // 20
            for i in range(steps):
                opacity = i / steps
                widget.after(20 * i, lambda o=opacity: widget.set_opacity(o))
        except:
            pass  # Fallback if opacity not supported
    
    @staticmethod
    def slide_transition(container: ctk.CTkFrame, new_widget: ctk.CTkBaseClass):
        """Slide transition for widgets"""
        try:
            # Clear existing widgets
            for child in container.winfo_children():
                child.destroy()
            new_widget.pack(fill="both", expand=True)
        except:
            pass

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
    
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load user session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return UserSession(**data)
        return None
    
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                value, expires_at = row
                if datetime.now().timestamp() < expires_at:
                    return json.loads(value)
                else:
                    # Expired, remove from cache
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        return None

class NotificationSystem:
    """Advanced notification system for user feedback"""
    
    def __init__(self, parent_frame: ctk.CTkFrame):
        self.parent_frame = parent_frame
        self.notifications = []
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
        
        bg_color, fg_color = colors.get(notification["level"], colors["info"])
        
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
        """Load user settings from database"""
        try:
            # Load theme preference
            theme_setting = self.db.cache_get("user_theme")
            if theme_setting:
                self.current_theme = ThemeMode(theme_setting)
            
            # Load advanced options
            options = self.db.cache_get("advanced_options")
            if options:
                self.advanced_options.update(options)
            
        except Exception as e:
            print(f"Error loading settings: {e}")
    
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
        
        # Edit menu
        self.edit_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="✏️ Edit",
            width=80,
            command=self._show_edit_menu
        )
        self.edit_menu_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # View menu
        self.view_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="👁️ View",
            width=80,
            command=self._show_view_menu
        )
        self.view_menu_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Tools menu
        self.tools_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="🔧 Tools",
            width=80,
            command=self._show_tools_menu
        )
        self.tools_menu_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Templates menu
        self.templates_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="📋 Templates",
            width=100,
            command=self._show_templates_menu
        )
        self.templates_menu_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Help menu
        self.help_menu_btn = ctk.CTkButton(
            self.menu_frame,
            text="❓ Help",
            width=80,
            command=self._show_help_menu
        )
        self.help_menu_btn.grid(row=0, column=5, padx=5, pady=5)
        
        # Theme toggle
        self.theme_btn = ctk.CTkButton(
            self.menu_frame,
            text="🌙" if self.current_theme == ThemeMode.DARK else "☀️",
            width=50,
            command=self._toggle_theme
        )
        self.theme_btn.grid(row=0, column=7, padx=5, pady=5)
    
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
            width=150,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ColorScheme.PRIMARY,
            hover_color=ColorScheme.PRIMARY_DARK
        )
        self.generate_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Separator
        separator1 = ctk.CTkFrame(self.toolbar_frame, width=2, fg_color="gray30")
        separator1.grid(row=0, column=1, padx=5, pady=10, sticky="ns")
        
        # Quick actions
        self.quick_generate_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="⚡ Quick",
            command=self._quick_generate,
            height=35,
            width=80
        )
        self.quick_generate_btn.grid(row=0, column=2, padx=5, pady=10)
        
        self.batch_generate_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📦 Batch",
            command=self._batch_generate,
            height=35,
            width=80
        )
        self.batch_generate_btn.grid(row=0, column=3, padx=5, pady=10)
        
        # Separator
        separator2 = ctk.CTkFrame(self.toolbar_frame, width=2, fg_color="gray30")
        separator2.grid(row=0, column=4, padx=5, pady=10, sticky="ns")
        
        # Export actions
        self.export_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📤 Export",
            command=self._export_scenarios,
            height=35,
            width=80
        )
        self.export_btn.grid(row=0, column=5, padx=5, pady=10)
        
        self.import_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📥 Import",
            command=self._import_scenarios,
            height=35,
            width=80
        )
        self.import_btn.grid(row=0, column=6, padx=5, pady=10)
        
        # Separator
        separator3 = ctk.CTkFrame(self.toolbar_frame, width=2, fg_color="gray30")
        separator3.grid(row=0, column=7, padx=5, pady=10, sticky="ns")
        
        # Settings
        self.settings_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="⚙️ Settings",
            command=self._show_settings,
            height=35,
            width=100
        )
        self.settings_btn.grid(row=0, column=8, padx=5, pady=10)
        
        # Statistics
        self.stats_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="📊 Statistics",
            command=self._show_statistics,
            height=35,
            width=100
        )
        self.stats_btn.grid(row=0, column=9, padx=5, pady=10)
    
    def _create_workspace(self):
        """Create main workspace area"""
        # Create paned window for resizable panels
        self.paned_window = ctk.CTkFrame(self.main_frame)
        self.paned_window.pack(fill="both", expand=True)
        
        # Left panel - Input and controls
        self.left_panel = ctk.CTkFrame(self.paned_window, width=400)
        self.left_panel.pack(side="left", fill="both", expand=False, padx=(0, 5))
        
        # Right panel - Results and analysis
        self.right_panel = ctk.CTkFrame(self.paned_window)
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
        
        # Clear button
        clear_btn = ctk.CTkButton(
            header_frame,
            text="🗑️ Clear",
            command=self._clear_results,
            width=80,
            height=30,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "#DCE4EE")
        )
        clear_btn.pack(side="right", padx=(10, 15), pady=10)
        
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
        status_color = "green" if NIOSH_AVAILABLE else "red"
        
        self.system_status_label = ctk.CTkLabel(
            self.status_frame,
            text=status_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=status_color
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
        self.root.bind("<F1>", lambda e: self._show_help())
        
        # Window events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_auto_save(self):
        """Setup auto-save functionality"""
        if self.advanced_options.get("auto_save", True):
            self._auto_save_session()
            # Schedule next auto-save (every 5 minutes)
            self.root.after(300000, self._setup_auto_save)
    
    def _toggle_advanced_panel(self):
        """Toggle advanced options panel visibility"""
        if self.advanced_content.winfo_manager() == "":
            # Show advanced content
            self.advanced_content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.advanced_toggle.configure(text="🔧 Advanced Options ▲")
            AnimationManager.fade_in(self.advanced_content, 200)
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
    
    # Menu methods
    def _show_file_menu(self):
        """Show file menu dropdown"""
        self._show_menu_dropdown([
            ("🆕 New Session", self._new_session),
            ("📂 Open Session", self._open_session),
            ("💾 Save Session", self._save_session),
            ("💾 Save As...", self._save_session_as),
            ("", None),  # Separator
            ("📤 Export Results", self._export_scenarios),
            ("📥 Import Scenarios", self._import_scenarios),
            ("", None),  # Separator
            ("🚪 Exit", self._on_closing)
        ], self.file_menu_btn)
    
    def _show_edit_menu(self):
        """Show edit menu dropdown"""
        self._show_menu_dropdown([
            ("↩️ Undo", self._undo),
            ("↪️ Redo", self._redo),
            ("", None),  # Separator
            ("🗑️ Clear All", self._clear_results),
            ("📋 Copy Results", self._copy_results),
            ("🔍 Find", self._find_in_results)
        ], self.edit_menu_btn)
    
    def _show_view_menu(self):
        """Show view menu dropdown"""
        self._show_menu_dropdown([
            ("🎨 Change Theme", self._toggle_theme),
            ("📏 Increase Font", self._increase_font),
            ("📏 Decrease Font", self._decrease_font),
            ("", None),  # Separator
            ("🔄 Refresh", self._refresh_view),
            ("📊 Toggle Statistics", self._toggle_statistics)
        ], self.view_menu_btn)
    
    def _show_tools_menu(self):
        """Show tools menu dropdown"""
        self._show_menu_dropdown([
            ("⚡ Quick Generate", self._quick_generate),
            ("📦 Batch Generate", self._batch_generate),
            ("📈 Statistics", self._show_statistics),
            ("🎯 Risk Analysis", self._show_risk_analysis),
            ("", None),  # Separator
            ("⚙️ Settings", self._show_settings),
            ("🧹 Clear Cache", self._clear_cache)
        ], self.tools_menu_btn)
    
    def _show_templates_menu(self):
        """Show templates menu dropdown"""
        menu_items = []
        
        # Add template categories
        categories = list(set(t.category for t in self.current_templates))
        for category in sorted(categories):
            menu_items.append((f"📁 {category}", lambda c=category: self._show_category_templates(c)))
        
        if menu_items:
            menu_items.append(("", None))  # Separator
        
        menu_items.extend([
            ("➕ Create Template", self._create_template),
            ("📝 Manage Templates", self._manage_templates),
            ("🔄 Refresh Templates", self._load_templates)
        ])
        
        self._show_menu_dropdown(menu_items, self.templates_menu_btn)
    
    def _show_help_menu(self):
        """Show help menu dropdown"""
        self._show_menu_dropdown([
            ("📖 User Guide", self._show_user_guide),
            ("🎹 Keyboard Shortcuts", self._show_shortcuts),
            ("❓ About", self._show_about),
            ("🐛 Report Issue", self._report_issue)
        ], self.help_menu_btn)
    
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
    
    # Core functionality methods will be continued in the next part due to length limits...
    
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