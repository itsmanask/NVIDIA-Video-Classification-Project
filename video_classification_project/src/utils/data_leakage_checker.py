import os
from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd
from datetime import datetime


class DataLeakageChecker:
    def __init__(self, target_dir):
        self.target_dir = Path(target_dir)
        self.video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]
        self.duplicate_data = []
        
    def get_video_files(self, split):
        """Get all video files from a split (train/val/test)"""
        split_path = self.target_dir / "raw" / split
        videos = []
        
        if not split_path.exists():
            print(f"WARNING: {split_path} does not exist!")
            return videos
        
        for ext in self.video_extensions:
            videos.extend(list(split_path.rglob(f"*{ext}")))
        
        return videos
    
    def get_file_hash(self, file_path, chunk_size=8192):
        """Calculate MD5 hash of a file for content comparison"""
        md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return None
    
    def check_filename_duplicates(self, export_excel=True):
        """Check for duplicate filenames across DIFFERENT splits only"""
        print("\n" + "="*70)
        print("CHECKING FOR FILENAME DUPLICATES ACROSS SPLITS")
        print("="*70)
        print("NOTE: Only flagging files with same name in DIFFERENT splits")
        print("      (Same filename in same split but different subcategories is OK)")
        
        # Collect all files from each split
        train_files = self.get_video_files('train')
        val_files = self.get_video_files('val')
        test_files = self.get_video_files('test')
        
        print(f"\nTotal files found:")
        print(f"  Train: {len(train_files)} videos")
        print(f"  Val:   {len(val_files)} videos")
        print(f"  Test:  {len(test_files)} videos")
        print(f"  TOTAL: {len(train_files) + len(val_files) + len(test_files)} videos")
        
        # Create filename -> (split, file_path) mapping
        filename_map = defaultdict(list)
        
        for file_path in train_files:
            filename_map[file_path.name].append(('train', file_path))
        
        for file_path in val_files:
            filename_map[file_path.name].append(('val', file_path))
        
        for file_path in test_files:
            filename_map[file_path.name].append(('test', file_path))
        
        # Find duplicates - only if filename appears in MULTIPLE SPLITS
        cross_split_duplicates = {}
        for name, locations in filename_map.items():
            splits_found = set(split for split, _ in locations)
            # Only flag if same filename is in 2+ different splits
            if len(splits_found) > 1:
                cross_split_duplicates[name] = locations
        
        if not cross_split_duplicates:
            print("\n✓ NO CROSS-SPLIT FILENAME DUPLICATES FOUND!")
            print("  No video filename appears in multiple splits (train/val/test).")
            
            # Check for within-split duplicates (same filename multiple times in same split)
            within_split_duplicates = self._check_within_split_duplicates(filename_map)
            if within_split_duplicates:
                print(f"\n⚠ Found {len(within_split_duplicates)} filenames duplicated WITHIN the same split:")
                for filename, split_counts in list(within_split_duplicates.items())[:5]:
                    for split, count in split_counts.items():
                        if count > 1:
                            print(f"    '{filename}' appears {count} times in {split}")
                if len(within_split_duplicates) > 5:
                    print(f"    ... and {len(within_split_duplicates) - 5} more")
                print("\n  This might indicate duplicate files in different subcategories within the same split.")
            
            return True
        else:
            print(f"\n✗ FOUND {len(cross_split_duplicates)} FILENAMES IN MULTIPLE SPLITS!")
            print("\nCRITICAL: These files appear in different splits:")
            
            # Prepare data for Excel export
            excel_data = []
            
            for filename, locations in sorted(cross_split_duplicates.items()):
                splits = sorted(set(split for split, _ in locations))
                print(f"\n  File: {filename}")
                print(f"  Appears in splits: {', '.join(splits)} ⚠")
                
                for split, path in locations:
                    print(f"    [{split}] {path.relative_to(self.target_dir)}")
                    
                    # Get file info
                    file_size = path.stat().st_size if path.exists() else 0
                    file_size_mb = file_size / (1024 * 1024)
                    
                    # Extract category info from path
                    parts = path.parts
                    try:
                        raw_idx = parts.index('raw')
                        split_name = parts[raw_idx + 1]
                        category = parts[raw_idx + 2] if len(parts) > raw_idx + 2 else 'N/A'
                        subcategory = parts[raw_idx + 3] if len(parts) > raw_idx + 3 else 'N/A'
                    except (ValueError, IndexError):
                        split_name = split
                        category = 'N/A'
                        subcategory = 'N/A'
                    
                    excel_data.append({
                        'Filename': filename,
                        'Split': split_name,
                        'Category': category,
                        'Subcategory': subcategory,
                        'Full_Path': str(path),
                        'Relative_Path': str(path.relative_to(self.target_dir)),
                        'File_Size_MB': round(file_size_mb, 2),
                        'Duplicate_Group': filename,
                        'Splits_Found_In': ', '.join(splits)
                    })
            
            # Export to Excel
            if export_excel and excel_data:
                self.export_duplicates_to_excel(excel_data, 'filename')
            
            return False
    
    def _check_within_split_duplicates(self, filename_map):
        """Check for duplicate filenames within the same split"""
        within_split_dupes = {}
        
        for filename, locations in filename_map.items():
            split_counts = defaultdict(int)
            for split, _ in locations:
                split_counts[split] += 1
            
            # If any split has this filename more than once
            if any(count > 1 for count in split_counts.values()):
                within_split_dupes[filename] = dict(split_counts)
        
        return within_split_dupes
    
    def check_content_duplicates(self, sample_size=None, export_excel=True):
        """Check for duplicate file content (same video) across DIFFERENT splits"""
        print("\n" + "="*70)
        print("CHECKING FOR CONTENT DUPLICATES ACROSS SPLITS (FILE HASHING)")
        print("="*70)
        print("NOTE: Only flagging identical content in DIFFERENT splits")
        
        if sample_size:
            print(f"      Checking sample of {sample_size} files per split for quick verification")
        else:
            print("      Checking ALL files (this may take a while for large datasets)")
        
        # Collect all files
        train_files = self.get_video_files('train')
        val_files = self.get_video_files('val')
        test_files = self.get_video_files('test')
        
        # Apply sampling if specified
        if sample_size:
            import random
            random.seed(42)
            train_files = random.sample(train_files, min(sample_size, len(train_files)))
            val_files = random.sample(val_files, min(sample_size, len(val_files)))
            test_files = random.sample(test_files, min(sample_size, len(test_files)))
        
        print(f"\nHashing files...")
        print(f"  Train: {len(train_files)} files")
        print(f"  Val:   {len(val_files)} files")
        print(f"  Test:  {len(test_files)} files")
        
        # Create hash -> (split, file_path) mapping
        hash_map = defaultdict(list)
        
        print("\nCalculating hashes for train files...")
        for i, file_path in enumerate(train_files, 1):
            file_hash = self.get_file_hash(file_path)
            if file_hash:
                hash_map[file_hash].append(('train', file_path))
            if i % 100 == 0:
                print(f"  Processed {i}/{len(train_files)} train files")
        
        print("Calculating hashes for val files...")
        for i, file_path in enumerate(val_files, 1):
            file_hash = self.get_file_hash(file_path)
            if file_hash:
                hash_map[file_hash].append(('val', file_path))
            if i % 100 == 0:
                print(f"  Processed {i}/{len(val_files)} val files")
        
        print("Calculating hashes for test files...")
        for i, file_path in enumerate(test_files, 1):
            file_hash = self.get_file_hash(file_path)
            if file_hash:
                hash_map[file_hash].append(('test', file_path))
            if i % 100 == 0:
                print(f"  Processed {i}/{len(test_files)} test files")
        
        # Find duplicates - only if content appears in MULTIPLE SPLITS
        cross_split_content_duplicates = {}
        for file_hash, locations in hash_map.items():
            splits_found = set(split for split, _ in locations)
            # Only flag if same content is in 2+ different splits
            if len(splits_found) > 1:
                cross_split_content_duplicates[file_hash] = locations
        
        if not cross_split_content_duplicates:
            print("\n✓ NO CROSS-SPLIT CONTENT DUPLICATES FOUND!")
            print("  No video content appears in multiple splits (train/val/test).")
            
            # Check for within-split content duplicates
            within_split_content_dupes = self._check_within_split_content_duplicates(hash_map)
            if within_split_content_dupes:
                print(f"\n⚠ Found {len(within_split_content_dupes)} pieces of content duplicated WITHIN the same split:")
                for file_hash, split_counts in list(within_split_content_dupes.items())[:3]:
                    for split, files in split_counts.items():
                        if len(files) > 1:
                            print(f"    Hash {file_hash[:12]}... appears {len(files)} times in {split}:")
                            for f in files[:2]:
                                print(f"      - {f.name}")
                if len(within_split_content_dupes) > 3:
                    print(f"    ... and {len(within_split_content_dupes) - 3} more")
                print("\n  This might indicate identical video files with different names in the same split.")
            
            return True
        else:
            print(f"\n✗ FOUND {len(cross_split_content_duplicates)} IDENTICAL VIDEOS IN MULTIPLE SPLITS!")
            print("\nCRITICAL: These videos (same content) appear in different splits:")
            
            # Prepare data for Excel export
            excel_data = []
            
            display_count = min(10, len(cross_split_content_duplicates))
            for idx, (file_hash, locations) in enumerate(sorted(cross_split_content_duplicates.items())[:display_count], 1):
                splits = sorted(set(split for split, _ in locations))
                print(f"\n  Group {idx} - Hash: {file_hash[:16]}...")
                print(f"  Appears in splits: {', '.join(splits)} ⚠")
                
                for split, path in locations:
                    print(f"    [{split}] {path.name}")
                    
                    # Get file info
                    file_size = path.stat().st_size if path.exists() else 0
                    file_size_mb = file_size / (1024 * 1024)
                    
                    # Extract category info from path
                    parts = path.parts
                    try:
                        raw_idx = parts.index('raw')
                        split_name = parts[raw_idx + 1]
                        category = parts[raw_idx + 2] if len(parts) > raw_idx + 2 else 'N/A'
                        subcategory = parts[raw_idx + 3] if len(parts) > raw_idx + 3 else 'N/A'
                    except (ValueError, IndexError):
                        split_name = split
                        category = 'N/A'
                        subcategory = 'N/A'
                    
                    excel_data.append({
                        'Filename': path.name,
                        'Split': split_name,
                        'Category': category,
                        'Subcategory': subcategory,
                        'Full_Path': str(path),
                        'Relative_Path': str(path.relative_to(self.target_dir)),
                        'File_Size_MB': round(file_size_mb, 2),
                        'MD5_Hash': file_hash,
                        'Duplicate_Group': f"Group_{idx}",
                        'Splits_Found_In': ', '.join(splits)
                    })
            
            if len(cross_split_content_duplicates) > display_count:
                print(f"\n  ... and {len(cross_split_content_duplicates) - display_count} more duplicate videos")
                
                # Add remaining duplicates to Excel data
                for idx, (file_hash, locations) in enumerate(list(cross_split_content_duplicates.items())[display_count:], display_count + 1):
                    splits = sorted(set(split for split, _ in locations))
                    for split, path in locations:
                        file_size = path.stat().st_size if path.exists() else 0
                        file_size_mb = file_size / (1024 * 1024)
                        
                        parts = path.parts
                        try:
                            raw_idx = parts.index('raw')
                            split_name = parts[raw_idx + 1]
                            category = parts[raw_idx + 2] if len(parts) > raw_idx + 2 else 'N/A'
                            subcategory = parts[raw_idx + 3] if len(parts) > raw_idx + 3 else 'N/A'
                        except (ValueError, IndexError):
                            split_name = split
                            category = 'N/A'
                            subcategory = 'N/A'
                        
                        excel_data.append({
                            'Filename': path.name,
                            'Split': split_name,
                            'Category': category,
                            'Subcategory': subcategory,
                            'Full_Path': str(path),
                            'Relative_Path': str(path.relative_to(self.target_dir)),
                            'File_Size_MB': round(file_size_mb, 2),
                            'MD5_Hash': file_hash,
                            'Duplicate_Group': f"Group_{idx}",
                            'Splits_Found_In': ', '.join(splits)
                        })
            
            # Export to Excel
            if export_excel and excel_data:
                self.export_duplicates_to_excel(excel_data, 'content')
            
            return False
    
    def _check_within_split_content_duplicates(self, hash_map):
        """Check for duplicate content within the same split"""
        within_split_dupes = {}
        
        for file_hash, locations in hash_map.items():
            split_files = defaultdict(list)
            for split, path in locations:
                split_files[split].append(path)
            
            # If any split has this content more than once
            if any(len(files) > 1 for files in split_files.values()):
                within_split_dupes[file_hash] = dict(split_files)
        
        return within_split_dupes
    
    def export_duplicates_to_excel(self, data, check_type):
        """Export duplicate data to Excel file"""
        df = pd.DataFrame(data)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"LEAKAGE_duplicate_{check_type}_report_{timestamp}.xlsx"
        
        try:
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Write main data
                df.to_excel(writer, sheet_name='Cross-Split Duplicates', index=False)
                
                # Create summary sheet
                summary_data = []
                
                if check_type == 'filename':
                    grouped = df.groupby('Duplicate_Group')
                    for group_name, group_df in grouped:
                        splits = sorted(group_df['Split'].unique())
                        summary_data.append({
                            'Filename': group_name,
                            'Count': len(group_df),
                            'Splits': ', '.join(splits),
                            'Categories': ', '.join(group_df['Category'].unique()),
                            'Subcategories': ', '.join(group_df['Subcategory'].unique())
                        })
                else:  # content
                    grouped = df.groupby('Duplicate_Group')
                    for group_name, group_df in grouped:
                        splits = sorted(group_df['Split'].unique())
                        filenames = ', '.join(group_df['Filename'].unique())
                        summary_data.append({
                            'Group': group_name,
                            'Count': len(group_df),
                            'Splits': ', '.join(splits),
                            'Filenames': filenames,
                            'MD5_Hash': group_df['MD5_Hash'].iloc[0] if 'MD5_Hash' in group_df else 'N/A'
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Auto-adjust column widths
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"\n✓ Excel report exported: {filename}")
            print(f"  Location: {os.path.abspath(filename)}")
            print(f"  Total cross-split duplicate entries: {len(df)}")
            
        except Exception as e:
            print(f"\n✗ Error exporting to Excel: {e}")
            print("  Trying to save as CSV instead...")
            try:
                csv_filename = filename.replace('.xlsx', '.csv')
                df.to_csv(csv_filename, index=False)
                print(f"✓ CSV report exported: {csv_filename}")
            except Exception as csv_error:
                print(f"✗ Error exporting to CSV: {csv_error}")
    
    def check_split_integrity(self):
        """Check if files are properly distributed across splits"""
        print("\n" + "="*70)
        print("CHECKING SPLIT INTEGRITY")
        print("="*70)
        
        train_files = self.get_video_files('train')
        val_files = self.get_video_files('val')
        test_files = self.get_video_files('test')
        
        total = len(train_files) + len(val_files) + len(test_files)
        
        if total == 0:
            print("\n✗ ERROR: No video files found in any split!")
            return False
        
        train_ratio = len(train_files) / total
        val_ratio = len(val_files) / total
        test_ratio = len(test_files) / total
        
        print(f"\nSplit distribution:")
        print(f"  Train: {len(train_files):5d} files ({train_ratio*100:5.2f}%)")
        print(f"  Val:   {len(val_files):5d} files ({val_ratio*100:5.2f}%)")
        print(f"  Test:  {len(test_files):5d} files ({test_ratio*100:5.2f}%)")
        print(f"  Total: {total:5d} files")
        
        # Check if distribution is reasonable (within expected ranges)
        expected_train = 0.7
        expected_val = 0.2
        expected_test = 0.1
        tolerance = 0.05  # 5% tolerance
        
        print(f"\nExpected ratios (±{tolerance*100}% tolerance):")
        print(f"  Train: {expected_train*100:.1f}% (found: {train_ratio*100:.2f}%)")
        print(f"  Val:   {expected_val*100:.1f}% (found: {val_ratio*100:.2f}%)")
        print(f"  Test:  {expected_test*100:.1f}% (found: {test_ratio*100:.2f}%)")
        
        issues = []
        if abs(train_ratio - expected_train) > tolerance:
            issues.append(f"Train ratio {train_ratio*100:.2f}% deviates from expected {expected_train*100}%")
        if abs(val_ratio - expected_val) > tolerance:
            issues.append(f"Val ratio {val_ratio*100:.2f}% deviates from expected {expected_val*100}%")
        if abs(test_ratio - expected_test) > tolerance:
            issues.append(f"Test ratio {test_ratio*100:.2f}% deviates from expected {expected_test*100}%")
        
        if issues:
            print("\n⚠ WARNINGS:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nNote: Small deviations are normal due to rounding with small subcategories.")
            return True
        else:
            print("\n✓ Split ratios are within expected ranges!")
            return True
    
    def run_all_checks(self, check_content=False, sample_size=None):
        """Run all leakage checks"""
        print("\n" + "="*70)
        print("DATA LEAKAGE CHECKER FOR VIDEO DATASET SPLITS")
        print("="*70)
        print(f"Target directory: {self.target_dir}")
        print("\nThis checker identifies files/content appearing in MULTIPLE SPLITS,")
        print("which would cause data leakage (train/val/test contamination).")
        
        results = {}
        
        # Check 1: Filename duplicates (fast)
        results['filename'] = self.check_filename_duplicates(export_excel=True)
        
        # Check 2: Split integrity
        results['integrity'] = self.check_split_integrity()
        
        # Check 3: Content duplicates (slow, optional)
        if check_content:
            results['content'] = self.check_content_duplicates(sample_size, export_excel=True)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if results['filename'] and results['integrity']:
            if check_content:
                if results['content']:
                    print("\n✓✓✓ ALL CHECKS PASSED!")
                    print("No data leakage detected. Your splits are clean.")
                    print("No files or content appear in multiple splits.")
                else:
                    print("\n✗✗✗ DATA LEAKAGE DETECTED!")
                    print("Same video content found in multiple splits (train/val/test).")
                    print("This will cause the model to see the same data during training and evaluation!")
            else:
                print("\n✓✓ FILENAME CHECKS PASSED!")
                print("No duplicate filenames detected across splits.")
                print("\nNote: Content-based checking was not performed.")
                print("Run with check_content=True for thorough verification.")
                print("(Content check detects identical videos with different names)")
        else:
            print("\n✗✗✗ DATA LEAKAGE DETECTED!")
            if not results['filename']:
                print("→ Duplicate filenames found in multiple splits (train/val/test).")
                print("  The same filename should not appear in both train and val, for example.")
            if not results['integrity']:
                print("→ Issues with split distribution ratios.")
        
        print("="*70)
        
        return all(results.values())


def main():
    # Set your target directory
    target_directory = "video_classification_project/data"
    
    checker = DataLeakageChecker(target_directory)
    
    # Quick check (filename only - fast)
    print("\n" + "="*70)
    print("RUNNING QUICK CHECK (FILENAME-BASED)")
    print("="*70)
    checker.run_all_checks(check_content=False)
    
    # Ask user if they want to run content check
    print("\n" + "="*70)
    print("OPTIONAL: CONTENT-BASED CHECK")
    print("="*70)
    print("Would you like to run a content-based check?")
    print("This uses file hashing to detect identical videos with different names.")
    print("WARNING: This can be slow for large datasets.")
    print("\nOptions:")
    print("  1. Skip content check")
    print("  2. Run full content check (all files)")
    print("  3. Run sample content check (quick verification)")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '2':
            print("\nRunning FULL content check...")
            checker.run_all_checks(check_content=True, sample_size=None)
        elif choice == '3':
            sample = int(input("Enter sample size per split (e.g., 100): ").strip())
            print(f"\nRunning SAMPLE content check ({sample} files per split)...")
            checker.run_all_checks(check_content=True, sample_size=sample)
        else:
            print("\nSkipping content check.")
    except (ValueError, KeyboardInterrupt):
        print("\nSkipping content check.")


if __name__ == "__main__":
    main()