import argparse
import sys

def merge_segments(input_file):
    """
    Merges adjoining segments with the same labels from a given segment file and outputs the results.

    Args:
        input_file (str): Path to the input segment file.

    Returns:
        list: A list of merged segments, where each segment is a tuple (start_time, end_time, label).
    """
    segments = []

    # Read the input file and sort segments
    with open(input_file, 'r') as file:
        for line in file:
            start_time, end_time, label = line.strip().split()
            segments.append((float(start_time), float(end_time), label))
    segments.sort(key=lambda x: x[0])

    # Iterate over the segments and merge adjoining segments with the same label
    merged_segments = []
    current_start, current_end, current_label = segments[0]

    for start_time, end_time, label in segments[1:]:
        # Check if the current segment can be merged with the previous segment
        if start_time == current_end and label == current_label:
            # Update the end time of the merged segment
            current_end = end_time
        else:
            # Add the previous merged segment to the list of merged segments
            merged_segments.append((round(current_start, 6), round(current_end, 6), current_label))
            # Update the current segment variables
            current_start, current_end, current_label = start_time, end_time, label

    # Add the last merged segment to the list of merged segments
    merged_segments.append((round(current_start, 6), round(current_end, 6), current_label))

    return merged_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge adjoining segments with the same labels. When no value of --mseg specified, output the merged segments to the stdout.')
    parser.add_argument('--seg', type=str, default='./test_label.txt', help='Path to the input segment file')
    parser.add_argument('--mseg', type=str, default='', help='Path to the merged segment file')

    args = parser.parse_args()

    # Merge segments
    merged_segments = merge_segments(args.seg)

    # Output the merged segments
    if args.mseg:
        # Write the merged segments to the output file
        with open(args.mseg, 'w') as file:
            for start_time, end_time, label in merged_segments:
                file.write(f"{start_time}\t{end_time}\t{label}\n")
    else:
        # Output the merged segments to the standard output
        for start_time, end_time, label in merged_segments:
            sys.stdout.write(f"{start_time}\t{end_time}\t{label}\n")
