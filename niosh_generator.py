import requests
import json
import os
from datetime import datetime

class NIOSHScenarioGenerator:
    """
    Generatore di scenari NIOSH per analisi di movimentazione manuale dei carichi
    utilizzando Ollama con llama3.2
    """
    
    # Domini NIOSH specifici per manual lifting
    NIOSH_DOMAINS = [
        "loading/unloading pallets at various heights",
        "transferring items between conveyors and storage",
        "lifting materials from floor to elevated surfaces",
        "warehouse order picking and shelving",
        "packaging and boxing operations",
        "supply roll/reel handling and positioning",
        "depalletizing and stacking operations",
        "cart loading and material transfer",
        "assembly line parts handling",
        "storage rack replenishment"
    ]
    
    def __init__(self, ollama_url="http://localhost:11434", model="llama3.2"):
        """
        Inizializza il generatore
        
        Args:
            ollama_url: URL del server Ollama
            model: Modello da utilizzare
        """
        self.ollama_url = ollama_url
        self.model = model
        self.output_file = "scenari_niosh.txt"
    
    def check_ollama_connection(self):
        """Verifica la connessione con Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_scenario(self, domain=None):
        """
        Genera uno scenario NIOSH per manual lifting
        
        Args:
            domain: Dominio specifico NIOSH (opzionale)
            
        Returns:
            Testo dello scenario generato
        """
        if domain:
            prompt = f"""Generate a NIOSH manual lifting scenario for: {domain}

EXAMPLES OF CORRECT FORMAT:
- The worker lifts a box from the floor and places it on a shelf above shoulder height.
- The employee transfers small packages from a conveyor to a pallet repeatedly during the shift.
- A warehouse operator moves containers from ground level to an elevated storage rack.
- The operator loads heavy reels from floor level onto a machine at chest height.
- The worker unloads cartons from various shelf heights and places them on a cart.

Requirements:
- ONE sentence only
- Describe a manual lifting/lowering action
- Include origin and destination of the lift
- NO numbers, weights, or measurements
- Use "The worker" or "The employee" or "A worker" or similar
- Focus on manual material handling tasks
- In English
- Simple and direct

Generate ONLY the scenario sentence, nothing else."""
        else:
            prompt = """Generate a NIOSH manual lifting scenario.

EXAMPLES OF CORRECT FORMAT:
- The worker lifts a box from the floor and places it on a shelf above shoulder height.
- The employee transfers small packages from a conveyor to a pallet repeatedly during the shift.
- A warehouse operator moves containers from ground level to an elevated storage rack.
- The operator loads heavy reels from floor level onto a machine at chest height.
- The worker unloads cartons from various shelf heights and places them on a cart.

Requirements:
- ONE sentence only
- Describe a manual lifting/lowering action
- Include origin and destination of the lift
- NO numbers, weights, or measurements
- Use "The worker" or "The employee" or "A worker" or similar
- Focus on manual material handling tasks
- In English
- Simple and direct

Generate ONLY the scenario sentence, nothing else."""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.8
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                scenario = result.get("response", "").strip()
                # Rimuovi eventuali virgolette o punteggiatura extra
                scenario = scenario.strip('"').strip("'")
                return scenario
            else:
                return None
        except Exception as e:
            print(f"Errore durante la generazione: {e}")
            return None
    
    def save_scenarios(self, scenarios, mode="append"):
        """
        Salva gli scenari nel file txt
        
        Args:
            scenarios: Lista di scenari da salvare
            mode: 'append' per aggiungere, 'write' per sovrascrivere
        """
        file_mode = "a" if mode == "append" else "w"
        
        with open(self.output_file, file_mode, encoding="utf-8") as f:
            for scenario in scenarios:
                f.write(f"{scenario}\n")
        
        print(f"✓ {len(scenarios)} scenari salvati in '{self.output_file}'")
    
    def load_existing_scenarios(self):
        """Carica gli scenari esistenti dal file"""
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"✓ File esistente trovato: {self.output_file}")
            return content
        return None
    
    def show_menu(self):
        """Mostra il menu principale"""
        print("\n" + "="*70)
        print("NIOSH MANUAL LIFTING SCENARIO GENERATOR")
        print("="*70)
        print("\n1. Generate random lifting scenarios")
        print("2. Generate domain-specific scenarios")
        print("3. Show available NIOSH domains")
        print("4. View current file content")
        print("5. Exit")
        print("\n" + "-"*70)
    
    def show_domains(self):
        """Mostra i domini NIOSH disponibili"""
        print("\n" + "="*70)
        print("AVAILABLE NIOSH MANUAL LIFTING DOMAINS")
        print("="*70)
        for i, domain in enumerate(self.NIOSH_DOMAINS, 1):
            print(f"{i:2d}. {domain}")
        print("-"*70)
    
    def run(self):
        """Esegue il programma principale"""
        # Verifica connessione Ollama
        print("\nChecking Ollama connection...")
        if not self.check_ollama_connection():
            print("❌ ERROR: Cannot connect to Ollama.")
            print(f"Make sure Ollama is running on {self.ollama_url}")
            print("Run: ollama serve")
            return
        
        print(f"✓ Connected to Ollama at {self.ollama_url}")
        print(f"✓ Using model: {self.model}")
        
        # Verifica se esiste già un file
        existing = self.load_existing_scenarios()
        
        while True:
            self.show_menu()
            choice = input("\nChoose an option (1-5): ").strip()
            
            if choice == "1":
                # Genera scenari random
                try:
                    n = int(input("\nHow many scenarios to generate? "))
                    if n <= 0:
                        print("❌ Enter a positive number!")
                        continue
                    
                    mode_choice = input("\n(A)ppend to existing or (O)verwrite? [A/O]: ").strip().upper()
                    mode = "append" if mode_choice == "A" else "write"
                    
                    print(f"\n🔄 Generating {n} random lifting scenarios...")
                    scenarios = []
                    
                    for i in range(n):
                        print(f"   Scenario {i+1}/{n}...", end="\r")
                        scenario = self.generate_scenario()
                        if scenario:
                            scenarios.append(scenario)
                    
                    print(f"\n✓ Generated {len(scenarios)} scenarios")
                    self.save_scenarios(scenarios, mode)
                    
                except ValueError:
                    print("❌ Enter a valid number!")
            
            elif choice == "2":
                # Genera scenari di dominio specifico
                self.show_domains()
                try:
                    domain_idx = int(input("\nChoose a domain (1-10): "))
                    if 1 <= domain_idx <= len(self.NIOSH_DOMAINS):
                        domain = self.NIOSH_DOMAINS[domain_idx - 1]
                        
                        n = int(input(f"\nHow many scenarios for '{domain}'? "))
                        if n <= 0:
                            print("❌ Enter a positive number!")
                            continue
                        
                        mode_choice = input("\n(A)ppend to existing or (O)verwrite? [A/O]: ").strip().upper()
                        mode = "append" if mode_choice == "A" else "write"
                        
                        print(f"\n🔄 Generating {n} scenarios for '{domain}'...")
                        scenarios = []
                        
                        for i in range(n):
                            print(f"   Scenario {i+1}/{n}...", end="\r")
                            scenario = self.generate_scenario(domain)
                            if scenario:
                                scenarios.append(scenario)
                        
                        print(f"\n✓ Generated {len(scenarios)} scenarios")
                        self.save_scenarios(scenarios, mode)
                    else:
                        print("❌ Invalid choice!")
                except ValueError:
                    print("❌ Enter a valid number!")
            
            elif choice == "3":
                # Visualizza domini
                self.show_domains()
                input("\nPress ENTER to continue...")
            
            elif choice == "4":
                # Mostra contenuto file
                if os.path.exists(self.output_file):
                    with open(self.output_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    print("\n" + "="*70)
                    print(f"CONTENT OF {self.output_file}")
                    print("="*70)
                    print(content)
                    print("="*70)
                    
                    # Conta le righe
                    lines = content.strip().split('\n')
                    print(f"\nTotal scenarios: {len([l for l in lines if l.strip()])}")
                else:
                    print(f"\n❌ File {self.output_file} doesn't exist yet!")
                input("\nPress ENTER to continue...")
            
            elif choice == "5":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice!")


if __name__ == "__main__":
    generator = NIOSHScenarioGenerator()
    generator.run()