#!/bin/bash

cd utils/preprocess
cp ../../solvation_energies_raw.csv .

python - <<EOF
import pandas as pd
df = pd.read_csv('solvation_energies_raw.csv', usecols=lambda col: col != 'iupac')
df.to_csv('solvation_energies_temp.csv', index=False)
EOF

echo "smiles,expt,calc,atoms,bonds,charges" > ../../solvation_energies.csv

while IFS=, read -r smiles expt calc || [[ -n $smiles ]]; do

    if [[ $smiles == 'smiles' ]]; then
        continue
    fi

    acpype -i $smiles -b mol -c bcc -n 0 -m 1 -a gaff2 -o gmx -w
    eval "$(python top_to_lists.py "mol.acpype/mol_GMX.itp")"
    echo "$smiles,$expt,$calc,\"$atoms\",\"$bonds\",\"$charges\"" >> ../../solvation_energies.csv
    rm -rf mol.acpype
done < solvation_energies_temp.csv

rm solvation_energies_raw.csv
rm solvation_energies_temp.csv
cd ../..
