#!/usr/bin/env python3
"""
NIOSH Enhanced GUI - Core Functionality Module
Continues the EnhancedNIOSHGUI class with core functionality methods.
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import numpy as np
from collections import Counter, defaultdict

# Continue the EnhancedNIOSHGUI class with core functionality

class EnhancedNIOSHGUIExtended:
    """Extended functionality for the Enhanced NIOSH GUI"""
    
    # SCENARIO GENERATION METHODS
    def _generate_scenarios(self):
        """Main scenario generation method"""
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
        
        # Start generation in background thread
        thread = threading.Thread(target=self._generation_worker, daemon=True)
        thread.start()
    
    def _generation_worker(self):
        """Background worker for scenario generation"""
        try:
            self.is_generating = True
            self.root.after(0, self._set_generation_state, True)
            
            # Get parameters
            count = self.scenario_count.get()
            scenario_type = self.scenario_type.get()
            complexity = self.complexity_level.get()
            custom_scenario = self.scenario_text.get("0.0", "end").strip()
            
            # Update advanced options
            self._update_advanced_options()
            
            # Create progress dialog
            self.root.after(0, lambda: self._create_progress_dialog(
                f"Generating {count} {scenario_type.lower()} scenarios...",
                "Initializing NIOSH system..."
            ))
            
            generated_scenarios = []
            
            for i in range(count):
                # Check if generation was cancelled
                if hasattr(self, 'progress_dialog') and self.progress_dialog.cancelled:
                    break
                
                # Update progress
                progress = (i + 1) / count * 0.8  # Leave 20% for processing
                scenario_title = f"{scenario_type} Scenario {i+1}"
                
                self.root.after(0, lambda p=progress, t=scenario_title: 
                              self._update_progress(p, f"Generating {t}..."))
                
                # Generate scenario
                try:
                    scenario_id = f"{scenario_type.upper()}_{i+1:03d}"
                    
                    # Call NIOSH generation system
                    if custom_scenario and i == 0:
                        # Use custom scenario for first generation
                        final_report, calculated_data = generate_full_example(
                            scenario_id, custom_scenario
                        )
                    else:
                        final_report, calculated_data = generate_full_example(
                            scenario_id, scenario_title
                        )
                    
                    if final_report and calculated_data:
                        scenario = {
                            "id": scenario_id,
                            "title": scenario_title,
                            "type": scenario_type,
                            "complexity": complexity,
                            "report": final_report,
                            "data": calculated_data,
                            "timestamp": datetime.now().isoformat(),
                            "risk_level": self._calculate_risk_level(calculated_data)
                        }
                        generated_scenarios.append(scenario)
                        
                        self.root.after(0, lambda: self.notification_system.show_notification(
                            f"Generated scenario {i+1}/{count}", "success", 1500
                        ))
                    else:
                        self.root.after(0, lambda: self.notification_system.show_notification(
                            f"Failed to generate scenario {i+1}", "error", 2000
                        ))
                
                except Exception as e:
                    self.root.after(0, lambda: self.notification_system.show_notification(
                        f"Error generating scenario {i+1}: {str(e)}", "error"
                    ))
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.3)
            
            # Update UI with results
            if generated_scenarios:
                self.current_scenarios.extend(generated_scenarios)
                self.current_session.scenarios = self.current_scenarios
                
                self.root.after(0, lambda: self._update_all_displays(generated_scenarios))
                self.root.after(0, lambda: self._close_progress_dialog())
                
                self.root.after(0, lambda: self.notification_system.show_notification(
                    f"Successfully generated {len(generated_scenarios)} scenarios", "success"
                ))
            else:
                self.root.after(0, lambda: self._close_progress_dialog())
                self.root.after(0, lambda: self.notification_system.show_notification(
                    "No scenarios were generated", "warning"
                ))
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.root.after(0, lambda: self.notification_system.show_notification(
                error_msg, "error"
            ))
            self.root.after(0, lambda: self._close_progress_dialog())
        
        finally:
            self.is_generating = False
            self.root.after(0, self._set_generation_state, False)
            self.root.after(0, self._hide_progress_bar)
    
    def _quick_generate(self):
        """Quick scenario generation with default parameters"""
        # Set default parameters for quick generation
        self.scenario_count.set(1)
        self.scenario_type.set("Random")
        self.complexity_level.set("Medium")
        
        # Generate immediately
        self._generate_scenarios()
    
    def _batch_generate(self):
        """Batch generate multiple scenarios with different parameters"""
        if self.is_generating:
            self.notification_system.show_notification(
                "Generation already in progress", "warning"
            )
            return
        
        # Create batch generation dialog
        self._show_batch_generation_dialog()
    
    def _show_batch_generation_dialog(self):
        """Show batch generation configuration dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Batch Generation Configuration")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main frame
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text="Batch Generation Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))
        
        # Batch configuration
        config_frame = ctk.CTkScrollableFrame(main_frame, height=250)
        config_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # Scenario types to generate
        ctk.CTkLabel(
            config_frame,
            text="Select Scenario Types:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5))
        
        # Checkboxes for different types
        self.batch_types = {}
        scenario_types = ["Warehouse", "Manufacturing", "Construction", "Healthcare", "Office"]
        
        for scenario_type in scenario_types:
            var = ctk.BooleanVar(value=True)
            self.batch_types[scenario_type] = var
            checkbox = ctk.CTkCheckBox(
                config_frame,
                text=f"{scenario_type} (3 scenarios)",
                variable=var
            )
            checkbox.pack(anchor="w", padx=20, pady=2)
        
        # Complexity levels
        ctk.CTkLabel(
            config_frame,
            text="Complexity Levels:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(15, 5))
        
        self.batch_complexities = {}
        complexities = ["Low", "Medium", "High", "Extreme"]
        
        for complexity in complexities:
            var = ctk.BooleanVar(value=True if complexity != "Extreme" else False)
            self.batch_complexities[complexity] = var
            checkbox = ctk.CTkCheckBox(
                config_frame,
                text=f"{complexity} complexity",
                variable=var
            )
            checkbox.pack(anchor="w", padx=20, pady=2)
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(pady=(0, 10))
        
        ctk.CTkButton(
            button_frame,
            text="Generate Batch",
            command=lambda: self._execute_batch_generation(dialog),
            width=120
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=120
        ).pack(side="left", padx=10)
    
    def _execute_batch_generation(self, dialog):
        """Execute batch generation with selected configurations"""
        dialog.destroy()
        
        # Calculate total scenarios to generate
        selected_types = [t for t, var in self.batch_types.items() if var.get()]
        selected_complexities = [c for c, var in self.batch_complexities.items() if var.get()]
        
        total_scenarios = len(selected_types) * len(selected_complexities)
        
        if total_scenarios == 0:
            self.notification_system.show_notification(
                "Please select at least one scenario type and complexity level", "warning"
            )
            return
        
        if total_scenarios > 50:
            if not messagebox.askyesno(
                "Large Batch Generation",
                f"You are about to generate {total_scenarios} scenarios. This may take some time. Continue?"
            ):
                return
        
        # Start batch generation
        thread = threading.Thread(
            target=self._batch_generation_worker,
            args=(selected_types, selected_complexities),
            daemon=True
        )
        thread.start()
    
    def _batch_generation_worker(self, scenario_types: List[str], complexities: List[str]):
        """Background worker for batch generation"""
        try:
            self.is_generating = True
            self.root.after(0, self._set_generation_state, True)
            
            generated_scenarios = []
            total_count = len(scenario_types) * len(complexities)
            current_count = 0
            
            # Create progress dialog
            self.root.after(0, lambda: self._create_progress_dialog(
                "Batch Generation",
                f"Generating {total_count} scenarios..."
            ))
            
            for scenario_type in scenario_types:
                for complexity in complexities:
                    current_count += 1
                    progress = current_count / total_count
                    
                    self.root.after(0, lambda p=progress, st=scenario_type, c=complexity: 
                                  self._update_progress(p, f"Generating {st} - {c}..."))
                    
                    # Check for cancellation
                    if hasattr(self, 'progress_dialog') and self.progress_dialog.cancelled:
                        break
                    
                    try:
                        # Generate scenario
                        scenario_id = f"BATCH_{scenario_type.upper()}_{complexity.upper()}_{current_count:03d}"
                        scenario_title = f"{scenario_type} - {complexity} Complexity"
                        
                        final_report, calculated_data = generate_full_example(scenario_id, scenario_title)
                        
                        if final_report and calculated_data:
                            scenario = {
                                "id": scenario_id,
                                "title": scenario_title,
                                "type": scenario_type,
                                "complexity": complexity,
                                "report": final_report,
                                "data": calculated_data,
                                "timestamp": datetime.now().isoformat(),
                                "risk_level": self._calculate_risk_level(calculated_data),
                                "batch_generated": True
                            }
                            generated_scenarios.append(scenario)
                    
                    except Exception as e:
                        print(f"Error in batch generation: {e}")
                    
                    # Small delay
                    time.sleep(0.2)
                
                # Check for cancellation
                if hasattr(self, 'progress_dialog') and self.progress_dialog.cancelled:
                    break
            
            # Update UI
            if generated_scenarios:
                self.current_scenarios.extend(generated_scenarios)
                self.current_session.scenarios = self.current_scenarios
                
                self.root.after(0, lambda: self._update_all_displays(generated_scenarios))
                self.root.after(0, lambda: self._close_progress_dialog())
                
                self.root.after(0, lambda: self.notification_system.show_notification(
                    f"Batch generation completed: {len(generated_scenarios)} scenarios", "success"
                ))
            else:
                self.root.after(0, lambda: self._close_progress_dialog())
                self.root.after(0, lambda: self.notification_system.show_notification(
                    "Batch generation failed", "error"
                ))
        
        except Exception as e:
            self.root.after(0, lambda: self.notification_system.show_notification(
                f"Batch generation error: {str(e)}", "error"
            ))
            self.root.after(0, lambda: self._close_progress_dialog())
        
        finally:
            self.is_generating = False
            self.root.after(0, self._set_generation_state, False)
    
    # DISPLAY UPDATE METHODS
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
            fg_color="transparent",
            corner_radius=15
        )
        risk_badge.pack(side="right", padx=(10, 15), pady=10)
        
        # Content
        content_frame = ctk.CTkFrame(card)
        content_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Key metrics
        metrics_data = scenario["data"]["calculation_results"]["origin"]
        li_value = metrics_data["LI"]
        rwl_value = metrics_data["RWL"]
        load_weight = scenario["data"]["common_parameters"]["L_load_kg"]
        frequency = scenario["data"]["common_parameters"]["F_frequency_per_min"]
        
        # Create metrics grid
        metrics_grid = ctk.CTkFrame(content_frame, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=15, pady=10)
        
        # LI
        li_frame = ctk.CTkFrame(metrics_grid)
        li_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(li_frame, text="Lifting Index", font=ctk.CTkFont(size=10)).pack()
        ctk.CTkLabel(li_frame, text=f"{li_value:.2f}", font=ctk.CTkFont(size=14, weight="bold")).pack()
        
        # RWL
        rwl_frame = ctk.CTkFrame(metrics_grid)
        rwl_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(rwl_frame, text="RWL (kg)", font=ctk.CTkFont(size=10)).pack()
        ctk.CTkLabel(rwl_frame, text=f"{rwl_value:.1f}", font=ctk.CTkFont(size=14, weight="bold")).pack()
        
        # Load
        load_frame = ctk.CTkFrame(metrics_grid)
        load_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(load_frame, text="Load (kg)", font=ctk.CTkFont(size=10)).pack()
        ctk.CTkLabel(load_frame, text=f"{load_weight:.1f}", font=ctk.CTkFont(size=14, weight="bold")).pack()
        
        # Frequency
        freq_frame = ctk.CTkFrame(metrics_grid)
        freq_frame.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(freq_frame, text="Freq (/min)", font=ctk.CTkFont(size=10)).pack()
        ctk.CTkLabel(freq_frame, text=f"{frequency:.1f}", font=ctk.CTkFont(size=14, weight="bold")).pack()
        
        # Configure grid weights
        for i in range(4):
            metrics_grid.grid_columnconfigure(i, weight=1)
        
        # Action buttons
        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        view_btn = ctk.CTkButton(
            button_frame,
            text="👁️ View Details",
            command=lambda s=scenario: self._view_scenario_details(s),
            width=100,
            height=30
        )
        view_btn.pack(side="left", padx=(0, 5))
        
        export_btn = ctk.CTkButton(
            button_frame,
            text="📤 Export",
            command=lambda s=scenario: self._export_single_scenario(s),
            width=100,
            height=30
        )
        export_btn.pack(side="left", padx=5)
        
        duplicate_btn = ctk.CTkButton(
            button_frame,
            text="📋 Duplicate",
            command=lambda s=scenario: self._duplicate_scenario(s),
            width=100,
            height=30,
            fg_color="transparent",
            border_width=1
        )
        duplicate_btn.pack(side="left", padx=5)
        
        delete_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ Delete",
            command=lambda s=scenario: self._delete_scenario(s),
            width=100,
            height=30,
            fg_color=ColorScheme.ERROR,
            hover_color=ColorScheme.ERROR_DARK
        )
        delete_btn.pack(side="right", padx=(5, 0))
    
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
        """Update statistics tab with visualizations"""
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
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Create statistics visualizations
        self._create_statistics_charts(stats)
        self._create_statistics_summary(stats)
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not self.current_scenarios:
            return {}
        
        # Extract data
        li_values = []
        rwl_values = []
        load_values = []
        freq_values = []
        risk_levels = []
        scenario_types = []
        
        for scenario in self.current_scenarios:
            calc = scenario["data"]["calculation_results"]["origin"]
            li_values.append(calc["LI"])
            rwl_values.append(calc["RWL"])
            load_values.append(scenario["data"]["common_parameters"]["L_load_kg"])
            freq_values.append(scenario["data"]["common_parameters"]["F_frequency_per_min"])
            risk_levels.append(scenario["risk_level"])
            scenario_types.append(scenario["type"])
        
        return {
            "total_scenarios": len(self.current_scenarios),
            "li_values": li_values,
            "rwl_values": rwl_values,
            "load_values": load_values,
            "freq_values": freq_values,
            "risk_levels": risk_levels,
            "scenario_types": scenario_types,
            "risk_distribution": Counter(risk_levels),
            "type_distribution": Counter(scenario_types),
            "li_stats": {
                "mean": np.mean(li_values),
                "median": np.median(li_values),
                "std": np.std(li_values),
                "min": np.min(li_values),
                "max": np.max(li_values)
            },
            "rwl_stats": {
                "mean": np.mean(rwl_values),
                "median": np.median(rwl_values),
                "std": np.std(rwl_values),
                "min": np.min(rwl_values),
                "max": np.max(rwl_values)
            }
        }
    
    def _create_statistics_charts(self, stats: Dict[str, Any]):
        """Create statistical charts"""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.patch.set_facecolor('white' if self.current_theme == ThemeMode.LIGHT else '#2b2b2b')
            
            # LI Distribution
            ax1.hist(stats["li_values"], bins=15, alpha=0.7, color=ColorScheme.PRIMARY, edgecolor='black')
            ax1.set_title('Lifting Index Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Lifting Index')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('white' if self.current_theme == ThemeMode.LIGHT else '#3b3b3b')
            
            # Risk Level Distribution
            risk_labels = list(stats["risk_distribution"].keys())
            risk_counts = list(stats["risk_distribution"].values())
            risk_colors = [ColorScheme.get_risk_color(
                10 if label == "EXTREME" else 
                3.0 if label == "HIGH" else 
                1.0 if label == "MODERATE" else 0.5
            ) for label in risk_labels]
            
            ax2.pie(risk_counts, labels=risk_labels, colors=risk_colors, autopct='%1.1f%%')
            ax2.set_title('Risk Level Distribution', fontsize=12, fontweight='bold')
            
            # Scenario Type Distribution
            type_labels = list(stats["type_distribution"].keys())
            type_counts = list(stats["type_distribution"].values())
            
            ax3.bar(type_labels, type_counts, color=ColorScheme.SECONDARY, alpha=0.7)
            ax3.set_title('Scenario Type Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Scenario Type')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.set_facecolor('white' if self.current_theme == ThemeMode.LIGHT else '#3b3b3b')
            
            # Load vs LI Scatter Plot
            ax4.scatter(stats["load_values"], stats["li_values"], 
                       alpha=0.6, color=ColorScheme.WARNING, s=50)
            ax4.set_title('Load Weight vs Lifting Index', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Load Weight (kg)')
            ax4.set_ylabel('Lifting Index')
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('white' if self.current_theme == ThemeMode.LIGHT else '#3b3b3b')
            
            plt.tight_layout()
            
            # Embed in tkinter
            chart_frame = ctk.CTkFrame(self.stats_scroll)
            chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            error_label = ctk.CTkLabel(
                self.stats_scroll,
                text=f"Error creating charts: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color=ColorScheme.ERROR
            )
            error_label.pack(pady=20)
    
    def _create_statistics_summary(self, stats: Dict[str, Any]):
        """Create statistics summary frame"""
        summary_frame = ctk.CTkFrame(self.stats_scroll)
        summary_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Title
        ctk.CTkLabel(
            summary_frame,
            text="📈 Statistical Summary",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))
        
        # Summary grid
        grid_frame = ctk.CTkFrame(summary_frame)
        grid_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Total scenarios
        self._add_stat_row(grid_frame, "Total Scenarios:", f"{stats['total_scenarios']}", 0)
        
        # LI statistics
        self._add_stat_row(grid_frame, "Mean LI:", f"{stats['li_stats']['mean']:.2f}", 1)
        self._add_stat_row(grid_frame, "Median LI:", f"{stats['li_stats']['median']:.2f}", 2)
        self._add_stat_row(grid_frame, "LI Range:", f"{stats['li_stats']['min']:.2f} - {stats['li_stats']['max']:.2f}", 3)
        
        # RWL statistics
        self._add_stat_row(grid_frame, "Mean RWL:", f"{stats['rwl_stats']['mean']:.1f} kg", 4)
        self._add_stat_row(grid_frame, "RWL Range:", f"{stats['rwl_stats']['min']:.1f} - {stats['rwl_stats']['max']:.1f} kg", 5)
        
        # Most common risk level
        most_common_risk = stats["risk_levels"].most_common(1)[0][0]
        self._add_stat_row(grid_frame, "Most Common Risk:", most_common_risk, 6)
    
    def _add_stat_row(self, parent: ctk.CTkFrame, label: str, value: str, row: int):
        """Add a statistics row to the grid"""
        label_widget = ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12, weight="bold"))
        label_widget.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        
        value_widget = ctk.CTkLabel(parent, text=value, font=ctk.CTkFont(size=12))
        value_widget.grid(row=row, column=1, sticky="e", padx=10, pady=5)
    
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
        
        # Create risk analysis sections
        self._create_risk_overview()
        self._create_risk_recommendations()
        self._create_risk_mitigation_strategies()
    
    def _create_risk_overview(self):
        """Create risk overview section"""
        overview_frame = ctk.CTkFrame(self.risk_scroll)
        overview_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            overview_frame,
            text="⚠️ Risk Overview",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Count scenarios by risk level
        risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "EXTREME": 0}
        for scenario in self.current_scenarios:
            risk_counts[scenario["risk_level"]] += 1
        
        total = len(self.current_scenarios)
        
        # Risk level cards
        risk_frame = ctk.CTkFrame(overview_frame)
        risk_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        risk_info = [
            ("LOW", risk_counts["LOW"], ColorScheme.RISK_LOW, "🟢"),
            ("MODERATE", risk_counts["MODERATE"], ColorScheme.RISK_MODERATE, "🟡"),
            ("HIGH", risk_counts["HIGH"], ColorScheme.RISK_HIGH, "🟠"),
            ("EXTREME", risk_counts["EXTREME"], ColorScheme.RISK_EXTREME, "🔴")
        ]
        
        for i, (level, count, color, icon) in enumerate(risk_info):
            card = ctk.CTkFrame(risk_frame, fg_color=color, corner_radius=8)
            card.grid(row=0, column=i, padx=5, pady=10, sticky="ew")
            risk_frame.grid_columnconfigure(i, weight=1)
            
            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=24)).pack()
            ctk.CTkLabel(card, text=level, font=ctk.CTkFont(size=12, weight="bold"), text_color="white").pack()
            ctk.CTkLabel(card, text=f"{count} ({count/total*100:.1f}%)", font=ctk.CTkFont(size=16, weight="bold"), text_color="white").pack()
    
    def _create_risk_recommendations(self):
        """Create risk recommendations section"""
        rec_frame = ctk.CTkFrame(self.risk_scroll)
        rec_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            rec_frame,
            text="💡 Risk Recommendations",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Analyze high-risk scenarios
        high_risk_scenarios = [s for s in self.current_scenarios if s["risk_level"] in ["HIGH", "EXTREME"]]
        
        if high_risk_scenarios:
            recommendations_frame = ctk.CTkScrollableFrame(rec_frame, height=200)
            recommendations_frame.pack(fill="x", padx=15, pady=(0, 15))
            
            for scenario in high_risk_scenarios[:5]:  # Show top 5
                self._create_scenario_recommendation(scenario, recommendations_frame)
        else:
            no_risk_label = ctk.CTkLabel(
                rec_frame,
                text="✅ No high-risk scenarios detected. All scenarios are within acceptable limits.",
                font=ctk.CTkFont(size=14),
                text_color=ColorScheme.SUCCESS
            )
            no_risk_label.pack(pady=20)
    
    def _create_scenario_recommendation(self, scenario: Dict[str, Any], parent: ctk.CTkFrame):
        """Create recommendation for a specific scenario"""
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.pack(fill="x", padx=5, pady=5)
        
        # Scenario info
        calc = scenario["data"]["calculation_results"]["origin"]
        li_value = calc["LI"]
        
        # Header
        header_frame = ctk.CTkFrame(card, fg_color=ColorScheme.WARNING, corner_radius=6)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            header_frame,
            text=f"⚠️ {scenario['title']} (LI: {li_value:.2f})",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white"
        ).pack(padx=10, pady=8)
        
        # Recommendations
        recommendations = self._generate_scenario_recommendations(scenario)
        
        for rec in recommendations:
            ctk.CTkLabel(
                card,
                text=f"• {rec}",
                font=ctk.CTkFont(size=11),
                anchor="w"
            ).pack(anchor="w", padx=15, pady=2)
    
    def _generate_scenario_recommendations(self, scenario: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for a scenario"""
        recommendations = []
        calc = scenario["data"]["calculation_results"]["origin"]
        params = scenario["data"]["common_parameters"]
        
        # High LI recommendations
        if calc["LI"] > 3.0:
            if calc["LI"] > 10:
                recommendations.append("URGENT: Immediate intervention required - extreme risk level")
            
            # Load reduction
            if params["L_load_kg"] > calc["RWL"]:
                reduction_needed = params["L_load_kg"] - calc["RWL"]
                recommendations.append(f"Reduce load weight by at least {reduction_needed:.1f} kg")
            
            # Frequency reduction
            if params["F_frequency_per_min"] > 4:
                recommendations.append("Reduce lifting frequency to below 4 lifts per minute")
            
            # Improve posture
            if scenario["data"]["origin_parameters"]["H_origin_cm"] < 38:
                recommendations.append("Increase horizontal distance to improve lifting posture")
            
            if scenario["data"]["origin_parameters"]["V_origin_cm"] < 75:
                recommendations.append("Raise starting height to reduce vertical distance")
            
            # Duration
            if params["duration_hours"] > 2:
                recommendations.append("Implement work-rest cycles to reduce continuous duration")
        
        return recommendations
    
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
    
    # UTILITY METHODS
    def _calculate_risk_level(self, calculated_data: Dict[str, Any]) -> str:
        """Calculate risk level based on Lifting Index"""
        li_value = calculated_data["calculation_results"]["origin"]["LI"]
        
        if li_value > 10:
            return "EXTREME"
        elif li_value > 3.0:
            return "HIGH"
        elif li_value > 1.0:
            return "MODERATE"
        else:
            return "LOW"
    
    def _update_advanced_options(self):
        """Update advanced options from UI"""
        self.advanced_options.update({
            "rag_enabled": self.rag_enabled_var.get() if hasattr(self, 'rag_enabled_var') else True,
            "validation_enabled": self.validation_enabled_var.get() if hasattr(self, 'validation_enabled_var') else True,
            "semantic_coherence": self.semantic_enabled_var.get() if hasattr(self, 'semantic_enabled_var') else True,
            "real_time_preview": self.preview_enabled_var.get() if hasattr(self, 'preview_enabled_var') else True,
            "performance_mode": self.performance_mode_var.get() if hasattr(self, 'performance_mode_var') else False,
            "auto_save": self.autosave_enabled_var.get() if hasattr(self, 'autosave_enabled_var') else True
        })
        
        # Save to database
        self._save_settings()
    
    def _set_generation_state(self, generating: bool):
        """Set UI state during generation"""
        state = "disabled" if generating else "normal"
        
        self.generate_btn.configure(state=state)
        self.quick_generate_btn.configure(state=state)
        self.batch_generate_btn.configure(state=state)
        
        if generating:
            self.generate_btn.configure(text="⏳ Generating...")
            self.status_progress.grid()  # Show progress bar
        else:
            self.generate_btn.configure(text="🚀 Generate Scenarios")
            self.status_progress.grid_remove()  # Hide progress bar
            self.status_progress.set(0)
    
    def _create_progress_dialog(self, title: str, message: str, can_cancel: bool = True):
        """Create progress dialog"""
        self.progress_dialog = ProgressDialog(self.root, title, message, can_cancel)
        self.status_label.configure(text=message)
        self.status_progress.set(0)
        self.status_progress.grid()
    
    def _update_progress(self, progress: float, message: str = None):
        """Update progress dialog"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_progress(progress, message)
            self.status_progress.set(progress)
            if message:
                self.status_label.configure(text=message)
    
    def _close_progress_dialog(self):
        """Close progress dialog"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            delattr(self, 'progress_dialog')
    
    def _hide_progress_bar(self):
        """Hide status progress bar"""
        self.status_progress.set(0)
        self.status_progress.grid_remove()
        self.status_label.configure(text="✅ Ready")
    
    def _clear_results(self):
        """Clear all results"""
        self.current_scenarios = []
        self.current_session.scenarios = []
        self._update_all_displays([])
        self.notification_system.show_notification("Results cleared", "info", 2000)