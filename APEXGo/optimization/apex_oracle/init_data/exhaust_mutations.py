from itertools import combinations, product
from Levenshtein import distance

def enumerate_mutations_with_indels(peptide_sequence, all_amino_acids, max_distance=3):
    """
    Generate all possible mutations of a peptide sequence with insertions, deletions, and substitutions
    such that the edited sequence is at most `max_distance` away from the original.
    
    Args:
        peptide_sequence (str): Original peptide sequence
        all_amino_acids (list): List of possible amino acids
        max_distance (int): Maximum edit distance
        
    Returns:
        list: List of mutated sequences
    """
    mutations = set()  # Using set to avoid duplicates
    sequence_length = len(peptide_sequence)
    
    # Recursive function to generate sequences with given operations
    def generate_recursive(sequence, operations_left):
        if operations_left <= 0:
            if distance(peptide_sequence, sequence) <= max_distance and sequence != peptide_sequence:
                mutations.add(sequence)
            return
        
        # Attempt substitutions
        for i in range(len(sequence)):
            for aa in all_amino_acids:
                if sequence[i] != aa:  # If the amino acid is different from the original
                    new_sequence = sequence[:i] + aa + sequence[i+1:]
                    generate_recursive(new_sequence, operations_left - 1)

        # Attempt insertions
        for i in range(len(sequence) + 1):
            for aa in all_amino_acids:
                new_sequence = sequence[:i] + aa + sequence[i:]
                generate_recursive(new_sequence, operations_left - 1)

        # Attempt deletions
        for i in range(len(sequence)):
            new_sequence = sequence[:i] + sequence[i+1:]
            generate_recursive(new_sequence, operations_left - 1)

    # Start recursive generation with initial operations
    generate_recursive(peptide_sequence, max_distance)
    
    return sorted(list(mutations))

ALL_AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

def main():
    peptide_sequence = "GHLLIHLIGKATLAL"
    max_distance = 3
    mutations = enumerate_mutations_with_indels(peptide_sequence, ALL_AMINO_ACIDS, max_distance)
    print(f"Number of mutations within {max_distance} edit distance: {len(mutations)}")
    # save mutations to file
    with open(f"mutations_{peptide_sequence}.txt", "w") as f:
        f.write("\n".join(mutations))
    

if __name__ == "__main__":
    main()