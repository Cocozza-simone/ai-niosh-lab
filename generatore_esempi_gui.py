#!/usr/bin/env python3
"""
Generatore di Esempi - Modern GUI for NIOSH Report Generation
Integrates with existing NIOSH production system to provide user-friendly interface
for generating, managing, and exporting NIOSH lifting analysis reports.

Author: Claude Code
Date: 2025-01-09
Version: 1.0.0
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback
from pathlib import Path

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

# Configure customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GeneratoreEsempiGUI:
    """
    Main GUI class for NIOSH Example Generator
    Provides modern interface for generating, managing, and exporting NIOSH reports
    """
    
    def __init__(self):
        """Initialize the GUI application"""
        self.root = ctk.CTk()
        self.root.title("Generatore di Esempi - NIOSH Report Generator")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize variables
        self.scenario_count = ctk.IntVar(value=1)
        self.scenario_type = ctk.StringVar(value="Random")
        self.complexity_level = ctk.StringVar(value="Medium")
        self.current_scenarios = []
        self.generation_in_progress = False
        self.rag_system = None
        
        # Configure grid weights for responsive layout
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Build GUI components first
        self._create_main_layout()
        self._create_input_section()
        self._create_control_section()
        self._create_output_section()
        self._create_status_section()
        
        # Setup event handlers
        self._setup_events()
        
        # Initialize NIOSH system after GUI is ready
        self._initialize_niosh_system()
        
        # Start GUI update loop
        self._update_gui_loop()
    
    def _initialize_niosh_system(self):
        """Initialize NIOSH RAG system if available"""
        if NIOSH_AVAILABLE:
            try:
                self.rag_system = NIOSHRAGSystem()
                self.log_message("NIOSH RAG System initialized successfully", "success")
            except Exception as e:
                self.log_message(f"Failed to initialize RAG system: {e}", "warning")
                self.rag_system = None
        else:
            self.log_message("NIOSH system not available - running in demo mode", "warning")
    
    def _create_main_layout(self):
        """Create the main layout structure"""
        # Main container with padding
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Generatore di Esempi - NIOSH Report Generator",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.grid(row=0, column=0, pady=(0, 20))
    
    def _create_input_section(self):
        """Create input controls section"""
        # Input section frame
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        self.input_frame.grid_columnconfigure(1, weight=1)
        
        # Input section title
        ctk.CTkLabel(
            self.input_frame,
            text="Generation Parameters",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=3, pady=(10, 15), sticky="w")
        
        # Number of examples
        ctk.CTkLabel(self.input_frame, text="Number of Examples:").grid(row=1, column=0, sticky="w", padx=(20, 10), pady=5)
        self.count_slider = ctk.CTkSlider(
            self.input_frame,
            from_=1,
            to=10,
            variable=self.scenario_count,
            number_of_steps=9
        )
        self.count_slider.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=5)
        self.count_label = ctk.CTkLabel(self.input_frame, text="1")
        self.count_label.grid(row=1, column=2, sticky="w", padx=(0, 20), pady=5)
        
        # Scenario type
        ctk.CTkLabel(self.input_frame, text="Scenario Type:").grid(row=2, column=0, sticky="w", padx=(20, 10), pady=5)
        self.type_combo = ctk.CTkComboBox(
            self.input_frame,
            variable=self.scenario_type,
            values=["Random", "Warehouse", "Manufacturing", "Construction", "Healthcare", "Office"],
            width=200
        )
        self.type_combo.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=5)
        
        # Complexity level
        ctk.CTkLabel(self.input_frame, text="Complexity Level:").grid(row=3, column=0, sticky="w", padx=(20, 10), pady=5)
        self.complexity_combo = ctk.CTkComboBox(
            self.input_frame,
            variable=self.complexity_level,
            values=["Low", "Medium", "High", "Extreme"],
            width=200
        )
        self.complexity_combo.grid(row=3, column=1, sticky="w", padx=(0, 20), pady=5)
        
        # Custom title input
        ctk.CTkLabel(self.input_frame, text="Custom Title (optional):").grid(row=4, column=0, sticky="w", padx=(20, 10), pady=5)
        self.title_entry = ctk.CTkEntry(self.input_frame, placeholder_text="e.g., Worker lifting cardboard boxes")
        self.title_entry.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(0, 20), pady=5)
        
        # Custom scenario input
        ctk.CTkLabel(self.input_frame, text="Custom Scenario (optional):").grid(row=5, column=0, sticky="nw", padx=(20, 10), pady=(15, 5))
        self.scenario_text = ctk.CTkTextbox(self.input_frame, height=80)
        self.scenario_text.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(0, 20), pady=(15, 5))
        self.scenario_text.insert("0.0", "Describe specific scenario details...")
    
    def _create_control_section(self):
        """Create control buttons section"""
        # Control buttons frame
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        
        # Main action buttons
        self.generate_btn = ctk.CTkButton(
            self.control_frame,
            text="Generate Examples",
            command=self._generate_examples,
            height=40,
            width=150,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.generate_btn.grid(row=0, column=0, padx=10, pady=10)
        
        self.clear_btn = ctk.CTkButton(
            self.control_frame,
            text="Clear Output",
            command=self._clear_output,
            height=40,
            width=120,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE")
        )
        self.clear_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Secondary action buttons
        self.import_btn = ctk.CTkButton(
            self.control_frame,
            text="Import Scenarios",
            command=self._import_scenarios,
            height=35,
            width=120
        )
        self.import_btn.grid(row=0, column=2, padx=10, pady=10)
        
        self.export_btn = ctk.CTkButton(
            self.control_frame,
            text="Export Results",
            command=self._export_results,
            height=35,
            width=120
        )
        self.export_btn.grid(row=0, column=3, padx=10, pady=10)
        
        # Advanced options
        self.advanced_btn = ctk.CTkButton(
            self.control_frame,
            text="Advanced Options",
            command=self._toggle_advanced,
            height=35,
            width=150
        )
        self.advanced_btn.grid(row=0, column=4, padx=10, pady=10)
        
        self.save_session_btn = ctk.CTkButton(
            self.control_frame,
            text="Save Session",
            command=self._save_session,
            height=35,
            width=120
        )
        self.save_session_btn.grid(row=0, column=5, padx=10, pady=10)
    
    def _create_output_section(self):
        """Create output display section"""
        # Output frame with tabs
        self.output_frame = ctk.CTkFrame(self.main_frame)
        self.output_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 20))
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_rowconfigure(1, weight=1)
        
        # Output section title
        ctk.CTkLabel(
            self.output_frame,
            text="Generated Results",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=(10, 15), sticky="w")
        
        # Create tabview for different output views
        self.tabview = ctk.CTkTabview(self.output_frame)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Add tabs
        self.tabview.add("Summary")
        self.tabview.add("Detailed Reports")
        self.tabview.add("Data Table")
        self.tabview.add("Statistics")
        
        # Summary tab - scrollable text
        self.summary_text = scrolledtext.ScrolledText(
            self.tabview.tab("Summary"),
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 10),
            bg=ctk.ThemeManager.theme["CTkTextbox"]["fg_color"][1] if ctk.get_appearance_mode() == "dark" else "white"
        )
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Detailed reports tab - text widget
        self.detailed_text = scrolledtext.ScrolledText(
            self.tabview.tab("Detailed Reports"),
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 9),
            bg=ctk.ThemeManager.theme["CTkTextbox"]["fg_color"][1] if ctk.get_appearance_mode() == "dark" else "white"
        )
        self.detailed_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Data table tab - text widget (simplified)
        self.table_text = scrolledtext.ScrolledText(
            self.tabview.tab("Data Table"),
            wrap=tk.NONE,
            height=20,
            font=("Consolas", 9),
            bg=ctk.ThemeManager.theme["CTkTextbox"]["fg_color"][1] if ctk.get_appearance_mode() == "dark" else "white"
        )
        self.table_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Statistics tab - text widget
        self.stats_text = scrolledtext.ScrolledText(
            self.tabview.tab("Statistics"),
            wrap=tk.WORD,
            height=20,
            font=("Consolas", 10),
            bg=ctk.ThemeManager.theme["CTkTextbox"]["fg_color"][1] if ctk.get_appearance_mode() == "dark" else "white"
        )
        self.stats_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_status_section(self):
        """Create status bar and progress indicator"""
        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=4, column=0, sticky="ew")
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready to generate examples",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=(20, 10), pady=10)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        self.progress_bar.set(0)
        
        # NIOSH status indicator
        self.niosh_status_label = ctk.CTkLabel(
            self.status_frame,
            text="✓ NIOSH System" if NIOSH_AVAILABLE else "⚠ NIOSH System Offline",
            font=ctk.CTkFont(size=10),
            text_color="green" if NIOSH_AVAILABLE else "orange"
        )
        self.niosh_status_label.grid(row=0, column=2, sticky="e", padx=(10, 20), pady=10)
    
    def _setup_events(self):
        """Setup event handlers"""
        # Update count label when slider changes
        def update_count_label(value):
            self.count_label.configure(text=str(int(value)))
        
        self.count_slider.configure(command=update_count_label)
        
        # Keyboard shortcuts
        self.root.bind("<Control-g>", lambda e: self._generate_examples())
        self.root.bind("<Control-o>", lambda e: self._import_scenarios())
        self.root.bind("<Control-s>", lambda e: self._save_session())
        self.root.bind("<Control-e>", lambda e: self._export_results())
        self.root.bind("<Escape>", lambda e: self._clear_output())
    
    def _update_gui_loop(self):
        """Periodic GUI update loop"""
        if self.generation_in_progress:
            # Animate progress bar
            current = self.progress_bar.get()
            if current < 0.9:
                self.progress_bar.set(current + 0.01)
        
        # Schedule next update
        self.root.after(100, self._update_gui_loop)
    
    def _generate_examples(self):
        """Generate NIOSH examples in a separate thread"""
        if self.generation_in_progress:
            messagebox.showwarning("Generation in Progress", "Please wait for current generation to complete.")
            return
        
        if not NIOSH_AVAILABLE:
            messagebox.showerror("NIOSH System Unavailable", 
                                "NIOSH modules are not available. Please ensure all dependencies are installed.")
            return
        
        # Start generation in background thread
        thread = threading.Thread(target=self._generation_worker, daemon=True)
        thread.start()
    
    def _generation_worker(self):
        """Background worker for example generation"""
        try:
            self.generation_in_progress = True
            self.root.after(0, self._set_ui_generation_state, True)
            
            count = self.scenario_count.get()
            scenario_type = self.scenario_type.get()
            complexity = self.complexity_level.get()
            custom_title = self.title_entry.get().strip()
            custom_scenario = self.scenario_text.get("0.0", "end").strip()
            
            self.log_message(f"Starting generation of {count} {scenario_type.lower()} scenarios with {complexity.lower()} complexity...")
            
            generated_scenarios = []
            
            for i in range(count):
                # Update progress
                progress = (i + 1) / count * 0.8  # Leave 20% for processing
                self.root.after(0, self.progress_bar.set, progress)
                
                # Generate scenario
                scenario_id = f"{scenario_type.upper()}_{i+1:03d}"
                title = custom_title if custom_title else f"{scenario_type} Lifting Task {i+1}"
                
                self.log_message(f"Generating scenario {i+1}/{count}: {title}")
                
                # Call NIOSH generation system
                final_report, calculated_data = generate_full_example(scenario_id, title)
                
                if final_report and calculated_data:
                    scenario = {
                        "id": scenario_id,
                        "title": title,
                        "type": scenario_type,
                        "complexity": complexity,
                        "report": final_report,
                        "data": calculated_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    generated_scenarios.append(scenario)
                    self.log_message(f"Successfully generated scenario {i+1}")
                else:
                    self.log_message(f"Failed to generate scenario {i+1}", "error")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.5)
            
            # Update UI with results
            self.current_scenarios = generated_scenarios
            self.root.after(0, self._update_display_with_results, generated_scenarios)
            
            self.log_message(f"Generation completed: {len(generated_scenarios)}/{count} scenarios generated successfully", "success")
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.log_message(error_msg, "error")
            self.root.after(0, messagebox.showerror, "Generation Error", error_msg)
            
        finally:
            self.generation_in_progress = False
            self.root.after(0, self._set_ui_generation_state, False)
            self.root.after(0, self.progress_bar.set, 1.0)
    
    def _set_ui_state(self, enabled):
        """Enable or disable UI controls"""
        state = "normal" if enabled else "disabled"
        
        self.generate_btn.configure(state=state)
        self.count_slider.configure(state=state)
        self.type_combo.configure(state=state)
        self.complexity_combo.configure(state=state)
        self.title_entry.configure(state=state)
        self.scenario_text.configure(state=state)
    
    def _set_ui_generation_state(self, generating):
        """Set UI state for generation process"""
        self._set_ui_state(not generating)
        
        if generating:
            self.generate_btn.configure(text="Generating...")
        else:
            self.generate_btn.configure(text="Generate Examples")
            self.progress_bar.set(0)
    
    def _update_display_with_results(self, scenarios):
        """Update display with generated results"""
        if not scenarios:
            self.log_message("No scenarios to display", "warning")
            return
        
        # Update Summary tab
        self._update_summary_tab(scenarios)
        
        # Update Detailed Reports tab
        self._update_detailed_tab(scenarios)
        
        # Update Data Table tab
        self._update_table_tab(scenarios)
        
        # Update Statistics tab
        self._update_statistics_tab(scenarios)
    
    def _update_summary_tab(self, scenarios):
        """Update summary tab with scenario overview"""
        self.summary_text.delete("1.0", tk.END)
        
        summary = "="*80 + "\n"
        summary += "GENERATION SUMMARY\n"
        summary += "="*80 + "\n\n"
        
        summary += f"Total Scenarios Generated: {len(scenarios)}\n"
        summary += f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"NIOSH System Status: {'Available' if NIOSH_AVAILABLE else 'Offline'}\n\n"
        
        # Scenario breakdown
        summary += "SCENARIO BREAKDOWN:\n"
        summary += "-"*40 + "\n"
        
        for i, scenario in enumerate(scenarios, 1):
            data = scenario["data"]["calculation_results"]
            li_origin = data["origin"]["LI"]
            rwl_origin = data["origin"]["RWL"]
            
            # Determine risk level
            if li_origin > 10:
                risk_level = "EXTREME"
                risk_color = "🔴"
            elif li_origin > 3.0:
                risk_level = "HIGH"
                risk_color = "🟠"
            elif li_origin > 1.0:
                risk_level = "MODERATE"
                risk_color = "🟡"
            else:
                risk_level = "LOW"
                risk_color = "🟢"
            
            summary += f"{i}. {scenario['title']}\n"
            summary += f"   Risk Level: {risk_color} {risk_level}\n"
            summary += f"   LI: {li_origin:.2f}, RWL: {rwl_origin:.1f} kg\n"
            summary += f"   Load: {scenario['data']['common_parameters']['L_load_kg']:.1f} kg\n"
            summary += f"   Frequency: {scenario['data']['common_parameters']['F_frequency_per_min']:.1f}/min\n\n"
        
        self.summary_text.insert("1.0", summary)
    
    def _update_detailed_tab(self, scenarios):
        """Update detailed reports tab with full reports"""
        self.detailed_text.delete("1.0", tk.END)
        
        detailed = "="*80 + "\n"
        detailed += "DETAILED NIOSH REPORTS\n"
        detailed += "="*80 + "\n\n"
        
        for i, scenario in enumerate(scenarios, 1):
            detailed += f"{'='*80}\n"
            detailed += f"SCENARIO {i}: {scenario['title']}\n"
            detailed += f"{'='*80}\n\n"
            detailed += scenario["report"]
            detailed += "\n\n"
        
        self.detailed_text.insert("1.0", detailed)
    
    def _update_table_tab(self, scenarios):
        """Update data table tab with structured data"""
        self.table_text.delete("1.0", tk.END)
        
        # Create CSV-like table
        table = "ID,Title,Risk Level,LI,RWL,Load (kg),Frequency (/min),Duration (h),H (cm),V (cm),A (deg)\n"
        table += "-"*120 + "\n"
        
        for scenario in scenarios:
            data = scenario["data"]
            calc = data["calculation_results"]
            
            li_origin = calc["origin"]["LI"]
            if li_origin > 10:
                risk_level = "EXTREME"
            elif li_origin > 3.0:
                risk_level = "HIGH"
            elif li_origin > 1.0:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            row = f"{scenario['id']},{scenario['title']},{risk_level},{li_origin:.2f},{calc['origin']['RWL']:.1f},"
            row += f"{data['common_parameters']['L_load_kg']:.1f},{data['common_parameters']['F_frequency_per_min']:.1f},"
            row += f"{data['common_parameters']['duration_hours']:.1f},{data['origin_parameters']['H_origin_cm']},"
            row += f"{data['origin_parameters']['V_origin_cm']},{data['origin_parameters']['A_origin_degrees']}\n"
            
            table += row
        
        self.table_text.insert("1.0", table)
    
    def _update_statistics_tab(self, scenarios):
        """Update statistics tab with analysis"""
        self.stats_text.delete("1.0", tk.END)
        
        if not scenarios:
            return
        
        stats = "="*80 + "\n"
        stats += "GENERATION STATISTICS\n"
        stats += "="*80 + "\n\n"
        
        # Risk level distribution
        risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "EXTREME": 0}
        li_values = []
        rwl_values = []
        load_values = []
        freq_values = []
        
        for scenario in scenarios:
            calc = scenario["data"]["calculation_results"]
            li = calc["origin"]["LI"]
            
            if li > 10:
                risk_counts["EXTREME"] += 1
            elif li > 3.0:
                risk_counts["HIGH"] += 1
            elif li > 1.0:
                risk_counts["MODERATE"] += 1
            else:
                risk_counts["LOW"] += 1
            
            li_values.append(li)
            rwl_values.append(calc["origin"]["RWL"])
            load_values.append(scenario["data"]["common_parameters"]["L_load_kg"])
            freq_values.append(scenario["data"]["common_parameters"]["F_frequency_per_min"])
        
        # Risk distribution
        stats += "RISK LEVEL DISTRIBUTION:\n"
        stats += "-"*30 + "\n"
        for risk, count in risk_counts.items():
            percentage = (count / len(scenarios)) * 100
            stats += f"{risk}: {count} scenarios ({percentage:.1f}%)\n"
        
        stats += "\n"
        
        # Statistical summary
        stats += "STATISTICAL SUMMARY:\n"
        stats += "-"*30 + "\n"
        stats += f"Lifting Index (LI):\n"
        stats += f"  Mean: {sum(li_values)/len(li_values):.2f}\n"
        stats += f"  Range: {min(li_values):.2f} - {max(li_values):.2f}\n"
        stats += f"  Median: {sorted(li_values)[len(li_values)//2]:.2f}\n\n"
        
        stats += f"Recommended Weight Limit (RWL):\n"
        stats += f"  Mean: {sum(rwl_values)/len(rwl_values):.1f} kg\n"
        stats += f"  Range: {min(rwl_values):.1f} - {max(rwl_values):.1f} kg\n\n"
        
        stats += f"Load Weight:\n"
        stats += f"  Mean: {sum(load_values)/len(load_values):.1f} kg\n"
        stats += f"  Range: {min(load_values):.1f} - {max(load_values):.1f} kg\n\n"
        
        stats += f"Frequency:\n"
        stats += f"  Mean: {sum(freq_values)/len(freq_values):.2f} lifts/min\n"
        stats += f"  Range: {min(freq_values):.2f} - {max(freq_values):.2f} lifts/min\n"
        
        self.stats_text.insert("1.0", stats)
    
    def _clear_output(self):
        """Clear all output displays"""
        self.summary_text.delete("1.0", tk.END)
        self.detailed_text.delete("1.0", tk.END)
        self.table_text.delete("1.0", tk.END)
        self.stats_text.delete("1.0", tk.END)
        self.current_scenarios = []
        self.progress_bar.set(0)
        self.log_message("Output cleared")
    
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
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    if filename.endswith('.json'):
                        imported_scenarios = json.load(f)
                        if isinstance(imported_scenarios, list):
                            self.current_scenarios.extend(imported_scenarios)
                        else:
                            self.current_scenarios.append(imported_scenarios)
                    else:
                        # For text files, try to parse as scenario descriptions
                        content = f.read()
                        self.title_entry.delete(0, tk.END)
                        self.title_entry.insert(0, f"Imported from {os.path.basename(filename)}")
                        self.scenario_text.delete("1.0", tk.END)
                        self.scenario_text.insert("1.0", content)
                
                self.log_message(f"Successfully imported scenarios from {os.path.basename(filename)}", "success")
                self._update_display_with_results(self.current_scenarios)
                
            except Exception as e:
                error_msg = f"Failed to import scenarios: {str(e)}"
                self.log_message(error_msg, "error")
                messagebox.showerror("Import Error", error_msg)
    
    def _export_results(self):
        """Export results to file"""
        if not self.current_scenarios:
            messagebox.showwarning("No Data", "No scenarios to export. Generate examples first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Markdown files", "*.md"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.current_scenarios, f, indent=4, ensure_ascii=False)
                
                elif filename.endswith('.md'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("# NIOSH Scenarios Export\n\n")
                        for i, scenario in enumerate(self.current_scenarios, 1):
                            f.write(f"## {i}. {scenario['title']}\n\n")
                            f.write(scenario['report'])
                            f.write("\n\n---\n\n")
                
                elif filename.endswith('.csv'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("ID,Title,Risk Level,LI,RWL,Load,Frequency,Duration\n")
                        for scenario in self.current_scenarios:
                            data = scenario["data"]
                            calc = data["calculation_results"]
                            li = calc["origin"]["LI"]
                            
                            if li > 10:
                                risk = "EXTREME"
                            elif li > 3.0:
                                risk = "HIGH"
                            elif li > 1.0:
                                risk = "MODERATE"
                            else:
                                risk = "LOW"
                            
                            f.write(f"{scenario['id']},{scenario['title']},{risk},{li:.2f},")
                            f.write(f"{calc['origin']['RWL']:.1f},{data['common_parameters']['L_load_kg']:.1f},")
                            f.write(f"{data['common_parameters']['F_frequency_per_min']:.1f},{data['common_parameters']['duration_hours']:.1f}\n")
                
                self.log_message(f"Successfully exported results to {os.path.basename(filename)}", "success")
                messagebox.showinfo("Export Complete", f"Results exported to {filename}")
                
            except Exception as e:
                error_msg = f"Failed to export results: {str(e)}"
                self.log_message(error_msg, "error")
                messagebox.showerror("Export Error", error_msg)
    
    def _save_session(self):
        """Save current session state"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "count": self.scenario_count.get(),
                "type": self.scenario_type.get(),
                "complexity": self.complexity_level.get(),
                "title": self.title_entry.get(),
                "scenario": self.scenario_text.get("0.0", "end")
            },
            "scenarios": self.current_scenarios
        }
        
        filename = filedialog.asksaveasfilename(
            title="Save Session",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=4, ensure_ascii=False)
                self.log_message(f"Session saved to {os.path.basename(filename)}", "success")
            except Exception as e:
                error_msg = f"Failed to save session: {str(e)}"
                self.log_message(error_msg, "error")
                messagebox.showerror("Save Error", error_msg)
    
    def _toggle_advanced(self):
        """Toggle advanced options panel"""
        # Create advanced options window
        advanced_window = ctk.CTkToplevel(self.root)
        advanced_window.title("Advanced Options")
        advanced_window.geometry("500x400")
        advanced_window.transient(self.root)
        advanced_window.grab_set()
        
        # Advanced options content
        options_frame = ctk.CTkScrollableFrame(advanced_window)
        options_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            options_frame,
            text="Advanced Generation Options",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))
        
        # RAG System toggle
        rag_enabled = ctk.BooleanVar(value=self.rag_system is not None)
        rag_check = ctk.CTkCheckBox(
            options_frame,
            text="Enable RAG-Enhanced Generation",
            variable=rag_enabled
        )
        rag_check.pack(pady=10)
        
        # Validation toggle
        validation_enabled = ctk.BooleanVar(value=True)
        validation_check = ctk.CTkCheckBox(
            options_frame,
            text="Enable Report Validation",
            variable=validation_enabled
        )
        validation_check.pack(pady=10)
        
        # Semantic coherence toggle
        semantic_enabled = ctk.BooleanVar(value=True)
        semantic_check = ctk.CTkCheckBox(
            options_frame,
            text="Enable Semantic Coherence",
            variable=semantic_enabled
        )
        semantic_check.pack(pady=10)
        
        # Ollama configuration
        ctk.CTkLabel(
            options_frame,
            text="Ollama Configuration",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 10))
        
        ollama_url = ctk.CTkEntry(
            options_frame,
            placeholder_text="http://localhost:11434/api/generate",
            width=400
        )
        ollama_url.pack(pady=5)
        
        ollama_model = ctk.CTkEntry(
            options_frame,
            placeholder_text="llama3.2:latest",
            width=400
        )
        ollama_model.pack(pady=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(options_frame)
        button_frame.pack(pady=20)
        
        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=lambda: self._apply_advanced_options(advanced_window, {
                "rag_enabled": rag_enabled.get(),
                "validation_enabled": validation_enabled.get(),
                "semantic_enabled": semantic_enabled.get(),
                "ollama_url": ollama_url.get(),
                "ollama_model": ollama_model.get()
            })
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=advanced_window.destroy
        ).pack(side="left", padx=10)
    
    def _apply_advanced_options(self, window, options):
        """Apply advanced options settings"""
        # Store options for use in generation
        self.advanced_options = options
        window.destroy()
        self.log_message("Advanced options applied", "success")
    
    def log_message(self, message, level="info"):
        """Log message to status bar"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add appropriate icon based on level
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        icon = icons.get(level, "ℹ️")
        status_text = f"{icon} [{timestamp}] {message}"
        
        self.status_label.configure(text=status_text)
        
        # Color code based on level (in a real implementation, might want to use text_color parameter)
        colors = {
            "info": ("gray10", "gray90"),
            "success": ("green", "darkgreen"),
            "warning": ("orange", "darkorange"),
            "error": ("red", "darkred")
        }
        
        # Update console for debugging
        print(f"{level.upper()}: {message}")
    
    def run(self):
        """Start the GUI application"""
        self.log_message("Generatore di Esempi started successfully")
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        app = GeneratoreEsempiGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()