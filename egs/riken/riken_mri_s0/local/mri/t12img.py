#!/usr/bin/env python3
"""
t12img.py - T1w MRI Scan Visualization Tool

This script loads T1-weighted MRI scans in NIfTI format (.nii or .nii.gz) and generates
a comprehensive visualization showing three orthogonal views (sagittal, coronal, and axial)
for each scan. It displays metadata including orientation, voxel size, file size, and 
intensity ranges.

The output is saved as a PNG image with all scans arranged in a grid layout, using
neurological convention (L=Left) for display.

Usage:
    python t12img.py scan1.nii.gz scan2.nii.gz scan3.nii.gz
    python t12img.py -o custom_output scan1.nii.gz scan2.nii.gz
    python t12img.py -d /path/to/output/dir scan1.nii.gz

Author: Modified from Harvard MRI visualization script
"""

import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import sys
from pathlib import Path


def get_file_size_mb(filepath):
    """
    Calculate file size in megabytes.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the file
        
    Returns
    -------
    float
        File size in MB
    """
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


def visualize_scans(scan_paths, output_path, output_dir):
    """
    Create comprehensive visualization of multiple T1w MRI scans.
    
    Parameters
    ----------
    scan_paths : list of str
        List of paths to NIfTI files (.nii or .nii.gz)
    output_path : str
        Name of the output PNG file (without directory)
    output_dir : str or Path
        Directory where the output image will be saved
        
    Returns
    -------
    str
        Full path to the saved output image
    """
    # Validate input files
    valid_scans = []
    for scan_path in scan_paths:
        if not os.path.exists(scan_path):
            print(f"Warning: File not found: {scan_path}", file=sys.stderr)
        elif not (scan_path.endswith('.nii') or scan_path.endswith('.nii.gz')):
            print(f"Warning: Not a NIfTI file: {scan_path}", file=sys.stderr)
        else:
            valid_scans.append(scan_path)
    
    if not valid_scans:
        raise ValueError("No valid NIfTI files found to process")
    
    n_scans = len(valid_scans)
    print(f"Processing {n_scans} scan(s)...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with appropriate size
    fig_height = max(12, n_scans * 4)
    fig, axes = plt.subplots(n_scans, 3, figsize=(18, fig_height))
    
    # Handle single scan case (axes won't be 2D array)
    if n_scans == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        'T1w MRI Scans Visualization\n[Neurological Convention: L=Left]',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    # Process each scan
    for idx, scan_path in enumerate(valid_scans):
        # Load image
        img = nib.load(scan_path)
        data = img.get_fdata()
        
        # Extract metadata
        filename = os.path.basename(scan_path)
        orientation = nib.aff2axcodes(img.affine)
        shape = data.shape
        voxel_size = img.header.get_zooms()[:3]
        file_size = get_file_size_mb(scan_path)
        data_range = (data.min(), data.max())
        
        # Create title with metadata
        title_info = (
            f"{filename}\n"
            f"Shape: {shape} | Orient: {orientation}\n"
            f"Voxel: {voxel_size[0]:.2f}×{voxel_size[1]:.2f}×{voxel_size[2]:.2f}mm | "
            f"Size: {file_size:.1f}MB\n"
            f"Range: [{data_range[0]:.0f}, {data_range[1]:.0f}]"
        )
        
        # Sagittal (Side View) - middle slice along X axis
        axes[idx, 0].imshow(data[data.shape[0]//2, :, :].T, cmap='gray', origin='lower')
        if idx == 0:
            axes[idx, 0].set_title(f'Sagittal (Side)\n{title_info}', fontsize=9)
        else:
            axes[idx, 0].set_title(title_info, fontsize=9)
        axes[idx, 0].axis('off')
        axes[idx, 0].text(
            0.02, 0.98, 'Sagittal', transform=axes[idx, 0].transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Coronal (Front View) - middle slice along Y axis
        coronal = data[:, data.shape[1]//2, :].T  # (z, x) after .T
        if orientation[0] == 'L':  # If radiological (LAS) → flip to make L=left
            coronal = np.fliplr(coronal)
        axes[idx, 1].imshow(coronal, cmap='gray', origin='lower')
        if idx == 0:
            axes[idx, 1].set_title(f'Coronal (Front)\n{title_info}', fontsize=9)
        else:
            axes[idx, 1].set_title(title_info, fontsize=9)
        axes[idx, 1].axis('off')
        axes[idx, 1].text(
            0.02, 0.98, 'Coronal', transform=axes[idx, 1].transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        # Add L/R labels for coronal view
        axes[idx, 1].text(
            0.02, 0.02, 'L', transform=axes[idx, 1].transAxes,
            fontsize=12, color='white', fontweight='bold',
            verticalalignment='bottom'
        )
        axes[idx, 1].text(
            0.98, 0.02, 'R', transform=axes[idx, 1].transAxes,
            fontsize=12, color='white', fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right'
        )
        
        # Axial (Top View) - middle slice along Z axis
        axial = data[:, :, data.shape[2]//2].T  # (y, x) after .T
        if orientation[0] == 'L':  # If radiological (LAS) → flip to make L=left
            axial = np.fliplr(axial)
        axes[idx, 2].imshow(axial, cmap='gray', origin='lower')
        if idx == 0:
            axes[idx, 2].set_title(f'Axial (Top)\n{title_info}', fontsize=9)
        else:
            axes[idx, 2].set_title(title_info, fontsize=9)
        axes[idx, 2].axis('off')
        axes[idx, 2].text(
            0.02, 0.98, 'Axial', transform=axes[idx, 2].transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        )
        # Add L/R labels for axial view
        axes[idx, 2].text(
            0.02, 0.02, 'L', transform=axes[idx, 2].transAxes,
            fontsize=12, color='white', fontweight='bold',
            verticalalignment='bottom'
        )
        axes[idx, 2].text(
            0.98, 0.02, 'R', transform=axes[idx, 2].transAxes,
            fontsize=12, color='white', fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right'
        )
        
        print(f"  [{idx+1}/{n_scans}] {filename} - Shape: {shape}, "
              f"Size: {file_size:.1f}MB, Orient: {orientation}")
    
    plt.tight_layout()
    
    # Save figure
    full_output_path = os.path.join(output_dir, output_path)
    plt.savefig(full_output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {full_output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*110)
    print("SUMMARY TABLE")
    print("="*110)
    print(f"{'Filename':<40} {'Shape':<18} {'Voxel Size (mm)':<20} {'File Size':<12} {'Orient'}")
    print("-"*110)
    
    for scan_path in valid_scans:
        img = nib.load(scan_path)
        data = img.get_fdata()
        
        filename = os.path.basename(scan_path)
        orientation = ''.join(nib.aff2axcodes(img.affine))
        shape = str(data.shape)
        voxel_size = img.header.get_zooms()[:3]
        voxel_str = f"{voxel_size[0]:.2f}×{voxel_size[1]:.2f}×{voxel_size[2]:.2f}"
        file_size = get_file_size_mb(scan_path)
        
        print(f"{filename:<40} {shape:<18} {voxel_str:<20} {file_size:>8.1f} MB   {orientation}")
    
    print("="*110)
    
    return full_output_path


def main():
    """
    Main function to parse arguments and execute visualization.
    """
    parser = argparse.ArgumentParser(
        description='Visualize T1-weighted MRI scans in NIfTI format with three orthogonal views.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single scan
  python t12img.py scan1.nii.gz
  
  # Visualize multiple scans
  python t12img.py scan1.nii.gz scan2.nii.gz scan3.nii.gz
  
  # Specify custom output filename
  python t12img.py -o my_scans.png scan1.nii.gz scan2.nii.gz
  
  # Specify custom output directory
  python t12img.py -d /path/to/output scan1.nii.gz
  
  # Use both custom directory and filename
  python t12img.py -d results -o analysis.png scan1.nii.gz scan2.nii.gz
  
  # Examples
  python t12img.py -d results -o analysis.png scan1.nii.gz scan2.nii.gz
        """
    )
    
    parser.add_argument(
        'scans',
        nargs='+',
        help='One or more NIfTI files (.nii or .nii.gz) to visualize'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='t1w_scans_visualization.png',
        help='Output filename (default: t1w_scans_visualization.png)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        default='mri_images',
        help='Output directory (default: mri_images)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    args = parser.parse_args()
    
    try:
        output_file = visualize_scans(args.scans, args.output, args.directory)
        print(f"\n✓ Success! Output saved to: {output_file}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())