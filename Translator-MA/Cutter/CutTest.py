# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:46:54 2023

@author: User
"""

import random

def reduce_lines(input_file, output_file, reduction_percentage, encoding='utf-16 LE'):
    with open(input_file, 'r', encoding='utf-16 LE') as infile:
        lines = infile.readlines()

    original_line_count = len(lines)
    target_line_count = int(original_line_count * (1 - reduction_percentage / 100))

    if target_line_count >= original_line_count:
        print("Target line count is greater than or equal to the original line count. No reduction needed.")
        return

    reduced_indices = random.sample(range(original_line_count), target_line_count)
    reduced_indices.sort()

    reduced_lines = [lines[i] for i in reduced_indices]

    with open(output_file, 'w', encoding='utf-16 LE') as outfile:
        outfile.writelines(reduced_lines)

    print(f"Reduced the document to {reduction_percentage}% of its original line count.")
    print(f"Original lines: {original_line_count}, Reduced lines: {len(reduced_lines)}")

# Example usage:
input_file = 'input.txt'  # Replace with the path to your input file
output_file = 'output.txt'  # Replace with the desired output file path
reduction_percentage = 50  # Replace with the desired reduction percentage
encoding = 'utf-16 LE'  # Replace with the desired encoding (e.g., 'utf-8', 'latin-1', etc.)

reduce_lines(input_file, output_file, reduction_percentage, encoding)


