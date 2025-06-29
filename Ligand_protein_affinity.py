import os
import yaml
import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import List
from rdkit import Chem
## run_pipeline_from_structures()
# === CONFIGURATION ===
protein_sequence = (
    "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"
)
ligand_dir = Path("ligands")  # directory containing ligand files
output_dir = Path("yaml_inputs")
output_dir.mkdir(parents=True, exist_ok=True)
results_file = "boltz_predictions_ranked.csv"
boltz_predict_script = "predict_affinity.py"  # adjust path if needed

# === STEP 0: CONVERT STRUCTURES TO SMILES ===
def convert_ligand_files_to_smiles(folder: Path, extensions={".sdf", ".mol", ".pdb"}) -> List[str]:
    smiles_list = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in extensions:
            mol = None
            if file.suffix == ".sdf":
                suppl = Chem.SDMolSupplier(str(file))
                mol = next((m for m in suppl if m is not None), None)
            else:
                mol = Chem.MolFromMolFile(str(file), sanitize=True)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
                print(f"Converted {file.name} -> {smiles}")
            else:
                print(f"Warning: failed to parse {file}")
    return smiles_list

# === STEP 1: CREATE YAML FILES ===
def create_yaml_files(smiles_list: List[str]):
    for i, smiles in enumerate(smiles_list):
        yaml_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": protein_sequence}},
                {"ligand": {"id": "B", "smiles": smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        file_path = output_dir / f"ligand_{i}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(yaml_data, f)
        print(f"YAML created: {file_path}")

# === STEP 2: RUN BOLTZ-2 PREDICTION ===
def run_boltz_prediction(yaml_file: Path, output_json: Path):
    cmd = [
        "python",
        boltz_predict_script,
        "--input",
        str(yaml_file),
        "--output",
        str(output_json),
    ]
    subprocess.run(cmd, check=True)

# === STEP 3: PARSE JSON RESULTS AND COMPILE TABLE ===
def collect_and_rank_predictions(smiles_list: List[str]) -> pd.DataFrame:
    results = []
    for i, smiles in enumerate(smiles_list):
        json_path = output_dir / f"ligand_{i}_output.json"
        if not json_path.exists():
            print(f"Warning: missing output for {json_path}")
            continue
        with open(json_path, "r") as f:
            pred = json.load(f)
        results.append({
            "index": i,
            "smiles": smiles,
            "affinity_pred_value": pred.get("affinity_pred_value"),
            "affinity_probability_binary": pred.get("affinity_probability_binary"),
            "affinity_pred_value1": pred.get("affinity_pred_value1"),
            "affinity_pred_value2": pred.get("affinity_pred_value2"),
            "affinity_probability_binary1": pred.get("affinity_probability_binary1"),
            "affinity_probability_binary2": pred.get("affinity_probability_binary2"),
        })
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="affinity_pred_value", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(results_file, index=False)
    print(f"\nRanked results saved to: {results_file}")
    return df_sorted

# === MAIN PIPELINE ===
def run_pipeline_from_structures():
    smiles_list = convert_ligand_files_to_smiles(ligand_dir)
    create_yaml_files(smiles_list)
    for i in range(len(smiles_list)):
        yaml_file = output_dir / f"ligand_{i}.yaml"
        output_json = output_dir / f"ligand_{i}_output.json"
        print(f"Running Boltz-2 prediction for ligand_{i}")
        run_boltz_prediction(yaml_file, output_json)
    ranked_df = collect_and_rank_predictions(smiles_list)
    print("\nTop ranked ligands:")
    print(ranked_df.head())
    return ranked_df
