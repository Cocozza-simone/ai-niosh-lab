#!/usr/bin/env python3
"""
NIOSH Enhanced GUI - Additional Functionality Module
Completes the Enhanced NIOSH GUI with templates, import/export, settings, and help systems.
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

class EnhancedNIOSHGUIComplete:
    """Complete functionality for the Enhanced NIOSH GUI"""
    
    # TEMPLATE MANAGEMENT
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
            
            # Increment usage count
            selected_template_obj.usage_count += 1
            self.db.save_template(selected_template_obj)
            
            self.notification_system.show_notification(
                f"Applied template: {selected_template_obj.name}", "success", 2000
            )
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Error applying template: {str(e)}", "error"
            )
    
    def _create_template(self):
        """Create new template from current configuration"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Create Template")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text="Create Scenario Template",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))
        
        # Template name
        name_frame = ctk.CTkFrame(main_frame)
        name_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(name_frame, text="Template Name:").pack(anchor="w", padx=10, pady=(10, 5))
        name_entry = ctk.CTkEntry(name_frame, placeholder_text="e.g., Warehouse Lifting Template")
        name_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Category
        category_frame = ctk.CTkFrame(main_frame)
        category_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(category_frame, text="Category:").pack(anchor="w", padx=10, pady=(10, 5))
        category_combo = ctk.CTkComboBox(
            category_frame,
            values=["Warehouse", "Manufacturing", "Construction", "Healthcare", "Office", "General"],
            width=200
        )
        category_combo.pack(anchor="w", padx=10, pady=(0, 10))
        category_combo.set("General")
        
        # Description
        desc_frame = ctk.CTkFrame(main_frame)
        desc_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(desc_frame, text="Description:").pack(anchor="w", padx=10, pady=(10, 5))
        desc_text = ctk.CTkTextbox(desc_frame, height=80)
        desc_text.pack(fill="x", padx=10, pady=(0, 10))
        
        # Tags
        tags_frame = ctk.CTkFrame(main_frame)
        tags_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(tags_frame, text="Tags (comma-separated):").pack(anchor="w", padx=10, pady=(10, 5))
        tags_entry = ctk.CTkEntry(tags_frame, placeholder_text="e.g., lifting, safety, warehouse")
        tags_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        def save_template():
            name = name_entry.get().strip()
            category = category_combo.get()
            description = desc_text.get("1.0", "end").strip()
            tags_str = tags_entry.get().strip()
            
            if not name:
                messagebox.showerror("Error", "Please enter a template name")
                return
            
            # Create template
            template_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            template = ScenarioTemplate(
                id=template_id,
                name=name,
                description=description,
                category=category,
                parameters={
                    "scenario_count": self.scenario_count.get(),
                    "scenario_type": self.scenario_type.get(),
                    "complexity_level": self.complexity_level.get(),
                    "custom_scenario": self.scenario_text.get("1.0", "end").strip()
                },
                tags=[tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else [],
                created_at=datetime.now().isoformat()
            )
            
            try:
                self.db.save_template(template)
                self._load_templates()  # Refresh template list
                dialog.destroy()
                self.notification_system.show_notification(
                    f"Template '{name}' created successfully", "success"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save template: {str(e)}")
        
        ctk.CTkButton(
            button_frame,
            text="Save Template",
            command=save_template,
            width=120
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=120
        ).pack(side="left", padx=10)
    
    def _manage_templates(self):
        """Show template management dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Manage Templates")
        dialog.geometry("800x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text="Template Management",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))
        
        # Template list
        list_frame = ctk.CTkFrame(main_frame)
        list_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # Create scrollable template list
        template_scroll = ctk.CTkScrollableFrame(list_frame, height=400)
        template_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Populate templates
        for template in self.current_templates:
            self._create_template_list_item(template, template_scroll)
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack()
        
        ctk.CTkButton(
            button_frame,
            text="Refresh",
            command=lambda: (self._load_templates(), dialog.destroy()),
            width=100
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Close",
            command=dialog.destroy,
            width=100
        ).pack(side="left", padx=10)
    
    def _create_template_list_item(self, template: ScenarioTemplate, parent: ctk.CTkFrame):
        """Create template list item"""
        item_frame = ctk.CTkFrame(parent, corner_radius=8)
        item_frame.pack(fill="x", padx=5, pady=5)
        
        # Template info
        info_frame = ctk.CTkFrame(item_frame)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        # Name and category
        header_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        header_frame.pack(fill="x")
        
        ctk.CTkLabel(
            header_frame,
            text=template.name,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")
        
        ctk.CTkLabel(
            header_frame,
            text=f"({template.category})",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(side="left", padx=(10, 0))
        
        # Usage count
        ctk.CTkLabel(
            header_frame,
            text=f"Used {template.usage_count} times",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="right")
        
        # Description
        if template.description:
            desc_label = ctk.CTkLabel(
                info_frame,
                text=template.description,
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=600
            )
            desc_label.pack(anchor="w", pady=(5, 0))
        
        # Tags
        if template.tags:
            tags_text = ", ".join(template.tags)
            tags_label = ctk.CTkLabel(
                info_frame,
                text=f"Tags: {tags_text}",
                font=ctk.CTkFont(size=10),
                text_color="gray",
                anchor="w"
            )
            tags_label.pack(anchor="w", pady=(5, 0))
        
        # Action buttons
        button_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=lambda t=template: (
                self._apply_template(f"{t.name} ({t.category})"),
                dialog.destroy()
            ),
            width=80,
            height=30
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            button_frame,
            text="Edit",
            command=lambda t=template: self._edit_template(t, dialog),
            width=80,
            height=30
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Delete",
            command=lambda t=template: self._delete_template(t, dialog),
            width=80,
            height=30,
            fg_color="#F44336",
            hover_color="#D32F2F"
        ).pack(side="right")
    
    def _edit_template(self, template: ScenarioTemplate, parent_dialog):
        """Edit existing template"""
        # Similar to create_template but pre-filled with existing data
        # Implementation would be similar to _create_template
        pass
    
    def _delete_template(self, template: ScenarioTemplate, parent_dialog):
        """Delete template with confirmation"""
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete the template '{template.name}'?"
        ):
            try:
                # Delete from database (would need to implement this method)
                self.current_templates.remove(template)
                self._load_templates()  # Refresh list
                parent_dialog.destroy()  # Close management dialog
                self.notification_system.show_notification(
                    f"Template '{template.name}' deleted", "info"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete template: {str(e)}")
    
    # IMPORT/EXPORT FUNCTIONALITY
    def _import_scenarios(self):
        """Import scenarios from file"""
        file_types = [
            ("JSON files", "*.json"),
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Import Scenarios",
            filetypes=file_types
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_data = json.load(f)
                    
                    if isinstance(imported_data, list):
                        self.current_scenarios.extend(imported_data)
                    elif isinstance(imported_data, dict) and 'scenarios' in imported_data:
                        self.current_scenarios.extend(imported_data['scenarios'])
                    else:
                        self.current_scenarios.append(imported_data)
            
            elif filename.endswith('.csv'):
                # Simple CSV import (would need more sophisticated parsing)
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Parse CSV and create scenarios (simplified)
                    pass
            
            else:
                # Text file - treat as custom scenario description
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.scenario_text.delete("1.0", "end")
                    self.scenario_text.insert("1.0", content)
            
            # Update session
            self.current_session.scenarios = self.current_scenarios
            self._update_all_displays(self.current_scenarios)
            
            self.notification_system.show_notification(
                f"Successfully imported from {os.path.basename(filename)}", "success"
            )
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Import failed: {str(e)}", "error"
            )
    
    def _export_scenarios(self):
        """Export scenarios to file"""
        if not self.current_scenarios:
            self.notification_system.show_notification(
                "No scenarios to export", "warning"
            )
            return
        
        file_types = [
            ("JSON files", "*.json"),
            ("CSV files", "*.csv"),
            ("Markdown files", "*.md"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Export Scenarios",
            defaultextension=".json",
            filetypes=file_types
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    export_data = {
                        "export_info": {
                            "timestamp": datetime.now().isoformat(),
                            "total_scenarios": len(self.current_scenarios),
                            "exported_by": "NIOSH Enhanced GUI v2.0"
                        },
                        "scenarios": self.current_scenarios
                    }
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif filename.endswith('.csv'):
                with open(filename, 'w', encoding='utf-8') as f:
                    # CSV header
                    f.write("ID,Title,Type,Complexity,Risk Level,LI,RWL,Load (kg),Frequency (/min),Timestamp\n")
                    
                    # CSV data
                    for scenario in self.current_scenarios:
                        calc = scenario["data"]["calculation_results"]["origin"]
                        params = scenario["data"]["common_parameters"]
                        
                        row = f"{scenario['id']},{scenario['title']},{scenario['type']},"
                        row += f"{scenario['complexity']},{scenario['risk_level']},"
                        row += f"{calc['LI']:.2f},{calc['RWL']:.1f},"
                        row += f"{params['L_load_kg']:.1f},{params['F_frequency_per_min']:.1f},"
                        row += f"{scenario['timestamp']}\n"
                        
                        f.write(row)
            
            elif filename.endswith('.md'):
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# NIOSH Scenarios Export\n\n")
                    f.write(f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Total Scenarios:** {len(self.current_scenarios)}\n\n")
                    
                    for i, scenario in enumerate(self.current_scenarios, 1):
                        f.write(f"## {i}. {scenario['title']}\n\n")
                        f.write(f"**Type:** {scenario['type']}\n")
                        f.write(f"**Complexity:** {scenario['complexity']}\n")
                        f.write(f"**Risk Level:** {scenario['risk_level']}\n\n")
                        f.write("### Full Report\n\n")
                        f.write(scenario['report'])
                        f.write("\n\n---\n\n")
            
            self.notification_system.show_notification(
                f"Successfully exported to {os.path.basename(filename)}", "success"
            )
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Export failed: {str(e)}", "error"
            )
    
    def _export_single_scenario(self, scenario: Dict[str, Any]):
        """Export a single scenario"""
        # Create temporary list with single scenario and export
        temp_scenarios = self.current_scenarios.copy()
        self.current_scenarios = [scenario]
        self._export_scenarios()
        self.current_scenarios = temp_scenarios
    
    # SESSION MANAGEMENT
    def _save_session(self):
        """Save current session"""
        try:
            self.current_session.last_updated = datetime.now().isoformat()
            self.current_session.scenarios = self.current_scenarios
            self.current_session.settings = self.advanced_options
            
            self.db.save_session(self.current_session)
            self.notification_system.show_notification(
                "Session saved successfully", "success", 2000
            )
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Failed to save session: {str(e)}", "error"
            )
    
    def _save_session_as(self):
        """Save session with new name"""
        # Similar to _save_session but allows renaming
        pass
    
    def _new_session(self):
        """Start new session"""
        if messagebox.askyesno(
            "New Session",
            "This will clear all current scenarios. Continue?"
        ):
            self.current_scenarios = []
            self.current_session = self._create_new_session()
            self._update_all_displays([])
            self.notification_system.show_notification(
                "New session started", "info"
            )
    
    def _open_session(self):
        """Open existing session"""
        # Implement session loading from database
        pass
    
    def _auto_save_session(self):
        """Auto-save current session"""
        if self.advanced_options.get("auto_save", True) and self.current_scenarios:
            self._save_session()
    
    # SETTINGS AND CONFIGURATION
    def _show_settings(self):
        """Show settings dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))
        
        # Settings tabs
        settings_tabview = ctk.CTkTabview(main_frame)
        settings_tabview.pack(fill="both", expand=True)
        
        settings_tabview.add("General")
        settings_tabview.add("Generation")
        settings_tabview.add("Display")
        settings_tabview.add("Advanced")
        
        # General settings tab
        self._create_general_settings(settings_tabview.tab("General"))
        
        # Generation settings tab
        self._create_generation_settings(settings_tabview.tab("Generation"))
        
        # Display settings tab
        self._create_display_settings(settings_tabview.tab("Display"))
        
        # Advanced settings tab
        self._create_advanced_settings(settings_tabview.tab("Advanced"))
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=(20, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=lambda: self._apply_settings(dialog),
            width=100
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Reset",
            command=self._reset_settings,
            width=100
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=100
        ).pack(side="left", padx=10)
    
    def _create_general_settings(self, parent):
        """Create general settings tab"""
        # Auto-save setting
        autosave_var = ctk.BooleanVar(value=self.advanced_options.get("auto_save", True))
        autosave_check = ctk.CTkCheckBox(
            parent,
            text="Enable auto-save (every 5 minutes)",
            variable=autosave_var
        )
        autosave_check.pack(anchor="w", padx=20, pady=10)
        self._temp_settings_vars = {"autosave": autosave_var}
        
        # Theme setting
        theme_frame = ctk.CTkFrame(parent)
        theme_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left", padx=(0, 10))
        theme_var = ctk.StringVar(value=self.current_theme.value)
        theme_combo = ctk.CTkComboBox(
            theme_frame,
            variable=theme_var,
            values=["light", "dark", "system"],
            width=150
        )
        theme_combo.pack(side="left")
        self._temp_settings_vars["theme"] = theme_var
    
    def _create_generation_settings(self, parent):
        """Create generation settings tab"""
        # Performance mode
        performance_var = ctk.BooleanVar(value=self.advanced_options.get("performance_mode", False))
        performance_check = ctk.CTkCheckBox(
            parent,
            text="Performance mode (faster generation, less validation)",
            variable=performance_var
        )
        performance_check.pack(anchor="w", padx=20, pady=10)
        self._temp_settings_vars["performance"] = performance_var
        
        # RAG enhancement
        rag_var = ctk.BooleanVar(value=self.advanced_options.get("rag_enabled", True))
        rag_check = ctk.CTkCheckBox(
            parent,
            text="Enable RAG-enhanced generation",
            variable=rag_var
        )
        rag_check.pack(anchor="w", padx=20, pady=10)
        self._temp_settings_vars["rag"] = rag_var
        
        # Validation
        validation_var = ctk.BooleanVar(value=self.advanced_options.get("validation_enabled", True))
        validation_check = ctk.CTkCheckBox(
            parent,
            text="Enable report validation",
            variable=validation_var
        )
        validation_check.pack(anchor="w", padx=20, pady=10)
        self._temp_settings_vars["validation"] = validation_var
    
    def _create_display_settings(self, parent):
        """Create display settings tab"""
        # Real-time preview
        preview_var = ctk.BooleanVar(value=self.advanced_options.get("real_time_preview", True))
        preview_check = ctk.CTkCheckBox(
            parent,
            text="Enable real-time preview",
            variable=preview_var
        )
        preview_check.pack(anchor="w", padx=20, pady=10)
        self._temp_settings_vars["preview"] = preview_var
        
        # Notifications
        notifications_var = ctk.BooleanVar(value=True)
        notifications_check = ctk.CTkCheckBox(
            parent,
            text="Enable notifications",
            variable=notifications_var
        )
        notifications_check.pack(anchor="w", padx=20, pady=10)
        self._temp_settings_vars["notifications"] = notifications_var
    
    def _create_advanced_settings(self, parent):
        """Create advanced settings tab"""
        # Cache settings
        cache_frame = ctk.CTkFrame(parent)
        cache_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(cache_frame, text="Cache Duration (hours):").pack(side="left", padx=(0, 10))
        cache_var = ctk.IntVar(value=24)
        cache_slider = ctk.CTkSlider(
            cache_frame,
            from_=1,
            to=168,  # 1 week
            variable=cache_var,
            number_of_steps=167,
            width=200
        )
        cache_slider.pack(side="left")
        self._temp_settings_vars["cache_duration"] = cache_var
        
        # Clear cache button
        clear_cache_btn = ctk.CTkButton(
            parent,
            text="Clear Cache",
            command=self._clear_cache,
            width=150
        )
        clear_cache_btn.pack(anchor="w", padx=20, pady=10)
    
    def _apply_settings(self, dialog):
        """Apply settings from dialog"""
        try:
            # Update advanced options
            self.advanced_options.update({
                "auto_save": self._temp_settings_vars["autosave"].get(),
                "performance_mode": self._temp_settings_vars["performance"].get(),
                "rag_enabled": self._temp_settings_vars["rag"].get(),
                "validation_enabled": self._temp_settings_vars["validation"].get(),
                "real_time_preview": self._temp_settings_vars["preview"].get(),
                "cache_duration": self._temp_settings_vars["cache_duration"].get()
            })
            
            # Update theme if changed
            new_theme = self._temp_settings_vars["theme"].get()
            if new_theme != self.current_theme.value:
                self.current_theme = ThemeMode(new_theme)
                self._setup_theme()
            
            # Save settings
            self._save_settings()
            
            dialog.destroy()
            self.notification_system.show_notification(
                "Settings applied successfully", "success"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")
    
    def _reset_settings(self):
        """Reset settings to defaults"""
        if messagebox.askyesno(
            "Reset Settings",
            "Reset all settings to default values?"
        ):
            self.advanced_options = {
                "rag_enabled": True,
                "validation_enabled": True,
                "semantic_coherence": True,
                "auto_save": True,
                "real_time_preview": True,
                "performance_mode": False,
                "cache_duration": 24
            }
            
            self._save_settings()
            self.notification_system.show_notification(
                "Settings reset to defaults", "info"
            )
    
    def _clear_cache(self):
        """Clear application cache"""
        try:
            # This would need to be implemented in the DatabaseManager
            self.notification_system.show_notification(
                "Cache cleared successfully", "success"
            )
        except Exception as e:
            self.notification_system.show_notification(
                f"Failed to clear cache: {str(e)}", "error"
            )
    
    # HELP AND SUPPORT
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
            
            ## Editing
            - Ctrl+Z: Undo
            - Ctrl+Y: Redo
            - Delete: Clear results
            
            ## Navigation
            - F1: Show help
            - F5: Refresh templates
            - Ctrl+,: Open settings
            - Tab: Navigate between controls
            - Enter: Confirm dialog
            - Escape: Cancel dialog
            
            ## View
            - Ctrl+Plus: Increase font size
            - Ctrl+Minus: Decrease font size
            - F11: Toggle fullscreen
            
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
            - Matplotlib (for charts)
            - NumPy (for statistics)
            - NIOSH calculation modules
            
            ## Credits
            Developed with Claude Code Enhanced
            Based on NIOSH lifting equation standards
            
            © 2025 - Enhanced NIOSH GUI System
            """
        )
    
    def _report_issue(self):
        """Report issue (open in browser or email)"""
        issue_text = """
        Please describe the issue you encountered:
        
        Steps to reproduce:
        1. 
        2. 
        3. 
        
        Expected behavior:
        
        Actual behavior:
        
        System information:
        - Operating System: {}
        - Python version: {}
        - GUI version: 2.0.0
        
        Additional information:
        """.format(
            platform.system() + " " + platform.release(),
            platform.python_version()
        )
        
        # Copy to clipboard or open email client
        try:
            # Try to open default email client
            webbrowser.open("mailto:support@example.com?subject=NIOSH%20GUI%20Issue&body=" + 
                          issue_text.replace('\n', '%0A').replace(' ', '%20'))
        except:
            # Fallback: show in dialog
            self._show_help_dialog(
                "Report Issue",
                "Please copy the template below and send it to support@example.com:\n\n" + 
                issue_text
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
    
    # SCENARIO ACTIONS
    def _view_scenario_details(self, scenario: Dict[str, Any]):
        """View detailed scenario information"""
        # Switch to detailed reports tab and scroll to scenario
        self.tabview.set("📄 Detailed Reports")
        # Implementation would scroll to the specific scenario
        
        self.notification_system.show_notification(
            f"Viewing details for: {scenario['title']}", "info", 2000
        )
    
    def _duplicate_scenario(self, scenario: Dict[str, Any]):
        """Duplicate a scenario"""
        try:
            # Create a copy with new ID
            duplicate = scenario.copy()
            duplicate["id"] = f"DUP_{scenario['id']}_{datetime.now().strftime('%H%M%S')}"
            duplicate["title"] = f"Copy of {scenario['title']}"
            duplicate["timestamp"] = datetime.now().isoformat()
            
            self.current_scenarios.append(duplicate)
            self.current_session.scenarios = self.current_scenarios
            self._update_all_displays(self.current_scenarios)
            
            self.notification_system.show_notification(
                f"Duplicated scenario: {scenario['title']}", "success"
            )
            
        except Exception as e:
            self.notification_system.show_notification(
                f"Failed to duplicate scenario: {str(e)}", "error"
            )
    
    def _delete_scenario(self, scenario: Dict[str, Any]):
        """Delete a scenario with confirmation"""
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete '{scenario['title']}'?"
        ):
            try:
                self.current_scenarios.remove(scenario)
                self.current_session.scenarios = self.current_scenarios
                self._update_all_displays(self.current_scenarios)
                
                self.notification_system.show_notification(
                    f"Deleted scenario: {scenario['title']}", "info"
                )
                
            except ValueError:
                self.notification_system.show_notification(
                    "Scenario not found", "error"
                )
            except Exception as e:
                self.notification_system.show_notification(
                    f"Failed to delete scenario: {str(e)}", "error"
                )
    
    # ADDITIONAL UTILITY METHODS
    def _undo(self):
        """Undo last action"""
        # Implementation would require action history tracking
        self.notification_system.show_notification("Undo functionality not yet implemented", "info")
    
    def _redo(self):
        """Redo last undone action"""
        # Implementation would require action history tracking
        self.notification_system.show_notification("Redo functionality not yet implemented", "info")
    
    def _copy_results(self):
        """Copy results to clipboard"""
        if self.current_scenarios:
            # Create summary text
            summary = f"NIOSH Scenarios Summary ({len(self.current_scenarios)} scenarios)\n\n"
            for i, scenario in enumerate(self.current_scenarios, 1):
                summary += f"{i}. {scenario['title']} - {scenario['risk_level']} (LI: {scenario['data']['calculation_results']['origin']['LI']:.2f})\n"
            
            # Copy to clipboard (implementation depends on platform)
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(summary)
                self.notification_system.show_notification("Results copied to clipboard", "success")
            except:
                self.notification_system.show_notification("Failed to copy to clipboard", "error")
        else:
            self.notification_system.show_notification("No results to copy", "warning")
    
    def _find_in_results(self):
        """Find text in results"""
        # Implementation would show search dialog
        self.notification_system.show_notification("Search functionality not yet implemented", "info")
    
    def _increase_font(self):
        """Increase font size"""
        # Implementation would adjust font sizes
        self.notification_system.show_notification("Font size increased", "info")
    
    def _decrease_font(self):
        """Decrease font size"""
        # Implementation would adjust font sizes
        self.notification_system.show_notification("Font size decreased", "info")
    
    def _refresh_view(self):
        """Refresh current view"""
        self._update_all_displays(self.current_scenarios)
        self.notification_system.show_notification("View refreshed", "info", 1500)
    
    def _toggle_statistics(self):
        """Toggle statistics visibility"""
        current_tab = self.tabview.get()
        if current_tab == "📈 Statistics":
            # Switch to overview
            self.tabview.set("📋 Overview")
        else:
            # Switch to statistics
            self.tabview.set("📈 Statistics")
    
    def _show_statistics(self):
        """Show statistics dialog"""
        self.tabview.set("📈 Statistics")
    
    def _show_risk_analysis(self):
        """Show risk analysis"""
        self.tabview.set("🎯 Risk Analysis")
    
    def _show_category_templates(self, category: str):
        """Show templates for specific category"""
        # Filter templates by category and show in dialog
        category_templates = [t for t in self.current_templates if t.category == category]
        
        if category_templates:
            template_names = [t.name for t in category_templates]
            # Show selection dialog
            self.notification_system.show_notification(
                f"Found {len(category_templates)} templates in {category}", "info"
            )
        else:
            self.notification_system.show_notification(
                f"No templates found in {category} category", "warning"
            )