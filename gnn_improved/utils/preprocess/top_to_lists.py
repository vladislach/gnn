import sys

filename = sys.argv[1]

with open(filename, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line == '[ atoms ]\n':
            atoms_start_idx = i
            for j, line in enumerate(lines[i:]):
                if line == '\n':
                    atoms_end_idx = i + j
                    break
        if line == '[ bonds ]\n':
            bonds_start_idx = i
            for j, line in enumerate(lines[i:]):
                if line == '\n':
                    bonds_end_idx = i + j
                    break
    
    atoms_lines = lines[atoms_start_idx+2:atoms_end_idx]
    bonds_lines = lines[bonds_start_idx+2:bonds_end_idx]
    
    atoms = ','.join(line.split()[1] for line in atoms_lines)
    charges = ','.join(line.split()[6] for line in atoms_lines)
    bonds = ','.join([f'{line.split()[0]}-{line.split()[1]}' for line in bonds_lines])
    
    print(f"atoms={atoms}")
    print(f"charges={charges}")
    print(f"bonds={bonds}")
    