"""
Family Speaker Clustering using PyAnnote Audio Embeddings

This script performs speaker clustering on family recordings (infant, mother, father)
using pre-trained speaker embeddings from pyannote.audio. It processes Audacity
label files and outputs speaker assignments in the same format.

Usage:
    python speaker_clustering.py --audio audio.wav --labels labels.txt --token YOUR_TOKEN
    python speaker_clustering.py --audio audio.wav --labels labels.txt --token YOUR_TOKEN --min-duration 1.0
    python speaker_clustering.py --audio audio.wav --labels labels.txt --token YOUR_TOKEN --num-speakers 2
"""

import torch
import librosa
import numpy as np
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

class FamilySpeakerClustering:
    """
    Speaker clustering system for family recordings using pyannote.audio embeddings.

    This class performs speaker diarization on family recordings by:
    1. Loading utterance segments from Audacity label files
    2. Extracting speaker embeddings using pyannote.audio
    3. Clustering embeddings to identify different speakers
    4. Assigning speaker types (child='c', female='f', male='m') based on heuristics

    Attributes:
        inference: PyAnnote inference model for speaker embedding extraction
        min_duration: Minimum duration for audio segments (shorter ones get extended)
        audio_duration: Total duration of the audio file (cached)
    """

    def __init__(self, hf_token=None, min_duration=0.5):
        """
        Initialize the family speaker clustering system.

        Args:
            hf_token (str, optional): HuggingFace authentication token for pyannote.audio
            min_duration (float, optional): Minimum duration in seconds for audio segments.
                                          Shorter segments will be extended by adding context.
                                          Defaults to 0.5 seconds.
        """
        # Set the HuggingFace token
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Load pre-trained speaker embedding model with authentication
        self.inference = Inference("pyannote/embedding",
                                 window="whole",
                                 use_auth_token=hf_token)
        print("Successfully loaded pyannote/embedding model")

        # Set minimum duration for segment extension
        self.min_duration = min_duration
        print(f"Using minimum segment duration: {min_duration}s")

        # Get total audio duration for boundary checking
        self.audio_duration = None

    def load_audacity_labels(self, label_file):
        """
        Load utterance segments from Audacity label format file.

        Args:
            label_file (str): Path to Audacity label file with format:
                             start_time\tend_time\tcontent

        Returns:
            list: List of tuples (start_time, end_time, content)
        """
        utterances = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    content = parts[2] if len(parts) > 2 else ""
                    utterances.append((start, end, content))
        return utterances

    def get_audio_duration(self, audio_file):
        """
        Get total duration of audio file (cached).

        Args:
            audio_file (str): Path to audio file

        Returns:
            float: Duration in seconds
        """
        if self.audio_duration is None:
            self.audio_duration = librosa.get_duration(filename=audio_file)
        return self.audio_duration

    def extend_short_segment(self, start, end, audio_file):
        """
        Extend short audio segments by adding context from surrounding audio.

        For segments shorter than min_duration, this function extends the time window
        by adding equal amounts of context before and after the original segment,
        while respecting audio file boundaries.

        Args:
            start (float): Original start time in seconds
            end (float): Original end time in seconds
            audio_file (str): Path to audio file for boundary checking

        Returns:
            tuple: (extended_start, extended_end, extended_duration)
        """
        original_duration = end - start

        if original_duration >= self.min_duration:
            # Segment is long enough, use as-is
            return start, end, original_duration

        # Calculate how much to extend
        extension_needed = self.min_duration - original_duration
        left_extension = extension_needed / 2
        right_extension = extension_needed / 2

        # Get audio file duration for boundary checking
        audio_duration = self.get_audio_duration(audio_file)

        # Calculate new boundaries
        new_start = max(0, start - left_extension)
        new_end = min(audio_duration, end + right_extension)

        # If we hit a boundary, extend more on the other side
        actual_duration = new_end - new_start
        if actual_duration < self.min_duration:
            if new_start == 0:
                # Hit left boundary, extend more to the right
                new_end = min(audio_duration, new_start + self.min_duration)
            elif new_end == audio_duration:
                # Hit right boundary, extend more to the left
                new_start = max(0, new_end - self.min_duration)

        extended_duration = new_end - new_start

        if extended_duration > original_duration:
            print(f"Extended segment {start:.3f}-{end:.3f} ({original_duration:.3f}s) -> {new_start:.3f}-{new_end:.3f} ({extended_duration:.3f}s)")

        return new_start, new_end, extended_duration

    def extract_embeddings(self, audio_file, utterances):
        """
        Extract speaker embeddings for each utterance using pyannote.audio.

        Short utterances are automatically extended by adding context from the
        surrounding audio to meet minimum duration requirements.

        Args:
            audio_file (str): Path to audio file
            utterances (list): List of (start, end, content) tuples

        Returns:
            tuple: (embeddings_array, valid_utterances_list)
                  - embeddings_array: numpy array of speaker embeddings
                  - valid_utterances_list: utterances that were successfully processed
        """
        embeddings = []
        valid_utterances = []
        extension_stats = {'extended': 0, 'original': 0}

        print(f"Processing {len(utterances)} utterances with smart extension...")

        for i, (start, end, content) in enumerate(utterances):
            if (i + 1) % 50 == 0:
                print(f"Processing utterance {i+1}/{len(utterances)}")

            original_duration = end - start

            try:
                # Extend short segments by expanding time window
                extended_start, extended_end, extended_duration = self.extend_short_segment(
                    start, end, audio_file)

                if extended_duration > original_duration:
                    extension_stats['extended'] += 1
                else:
                    extension_stats['original'] += 1

                # Load the extended audio segment
                audio, sr = librosa.load(audio_file,
                                       sr=16000,
                                       offset=extended_start,
                                       duration=extended_duration)

                # Ensure we have some audio
                if len(audio) < 1600:  # Less than 0.1 second
                    print(f"Warning: Very short audio segment ({len(audio)} samples) for {start}-{end}")

                # Extract embedding using pyannote
                waveform = torch.tensor(audio).unsqueeze(0)
                embedding = self.inference({"waveform": waveform,
                                          "sample_rate": sr})

                embeddings.append(embedding)
                # Keep original utterance times, not extended times
                valid_utterances.append((start, end, content))

            except Exception as e:
                print(f"Error processing segment {start}-{end} (duration: {original_duration:.3f}s): {str(e)[:100]}...")
                continue

        print(f"Successfully processed {len(embeddings)} utterances")
        print(f"  Extended {extension_stats['extended']} short segments")
        print(f"  Used original length for {extension_stats['original']} segments")

        return np.array(embeddings), valid_utterances

    def cluster_speakers(self, embeddings, num_speakers=None, max_speakers=3):
        """
        Cluster speaker embeddings using agglomerative clustering.

        Args:
            embeddings (np.ndarray): Array of speaker embeddings
            num_speakers (int, optional): Fixed number of speakers to detect.
                                        If None, automatically determines best number.
            max_speakers (int, optional): Maximum number of speakers when auto-detecting.
                                        Defaults to 3 (child + 2 parents).

        Returns:
            list: Cluster labels for each embedding
        """
        if len(embeddings) < 2:
            return [0] * len(embeddings)

        print(f"Clustering {len(embeddings)} embeddings...")

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        if num_speakers is not None:
            # Fixed number of speakers
            print(f"Using fixed number of speakers: {num_speakers}")

            if num_speakers == 1:
                return [0] * len(embeddings)

            n_clusters = min(num_speakers, len(embeddings))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )

            labels = clustering.fit_predict(distance_matrix)
            print(f"Created {n_clusters} clusters")
            return labels

        else:
            # Automatic number of speakers - try different numbers and select best
            print("Automatically determining number of speakers...")

            best_labels = None
            best_score = -1
            best_n_clusters = 1

            for n_clusters in range(1, min(max_speakers + 1, len(embeddings) + 1)):
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )

                labels = clustering.fit_predict(distance_matrix)

                # Compute clustering quality score
                if n_clusters > 1:
                    score = self.compute_clustering_score(distance_matrix, labels)
                    print(f"  {n_clusters} clusters: score = {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_n_clusters = n_clusters
                else:
                    best_labels = labels
                    best_n_clusters = 1

            print(f"Selected {best_n_clusters} clusters (automatic)")
            return best_labels if best_labels is not None else [0] * len(embeddings)

    def compute_clustering_score(self, distance_matrix, labels):
        """
        Compute clustering quality score based on intra vs inter cluster distances.

        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix between embeddings
            labels (list): Cluster assignments for each embedding

        Returns:
            float: Clustering score (higher is better)
        """
        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            return 0

        intra_distances = []
        inter_distances = []

        for i in range(len(labels)):
            same_cluster = [j for j in range(len(labels))
                           if labels[j] == labels[i] and j != i]
            if same_cluster:
                intra_dist = np.mean([distance_matrix[i][j] for j in same_cluster])
                intra_distances.append(intra_dist)

            diff_cluster = [j for j in range(len(labels))
                           if labels[j] != labels[i]]
            if diff_cluster:
                inter_dist = np.min([distance_matrix[i][j] for j in diff_cluster])
                inter_distances.append(inter_dist)

        if intra_distances and inter_distances:
            return np.mean(inter_distances) - np.mean(intra_distances)
        return 0

    def assign_speaker_types(self, embeddings, labels, utterances):
        """
        Assign speaker types (child, female, male) based on clustering results.

        Uses heuristics specific to family recordings:
        - Child typically has most total speech time
        - Assigns remaining speakers as female/male parents

        Args:
            embeddings (np.ndarray): Speaker embeddings
            labels (list): Cluster assignments
            utterances (list): List of (start, end, content) tuples

        Returns:
            list: Speaker type assignments ('c'=child, 'f'=female, 'm'=male)
        """
        unique_labels = sorted(set(labels))
        n_speakers = len(unique_labels)

        print(f"Assigning speaker types for {n_speakers} speakers...")

        # Calculate cluster characteristics
        cluster_stats = {}
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_durations = []

            for i in cluster_indices:
                start, end, content = utterances[i]
                cluster_durations.append(end - start)

            cluster_stats[label] = {
                'size': len(cluster_indices),
                'total_duration': sum(cluster_durations),
                'avg_duration': np.mean(cluster_durations) if cluster_durations else 0
            }

        # Sort clusters by total speech duration (better for family recordings)
        sorted_clusters = sorted(cluster_stats.items(),
                               key=lambda x: x[1]['total_duration'],
                               reverse=True)

        # Assignment logic based on family structure
        speaker_mapping = {}

        if n_speakers == 1:
            speaker_mapping[unique_labels[0]] = 'c'
            print("  Single speaker -> child (c)")

        elif n_speakers == 2:
            # Child typically has more total speech time
            speaker_mapping[sorted_clusters[0][0]] = 'c'
            speaker_mapping[sorted_clusters[1][0]] = 'f'  # Default to female parent

            print(f"  Two speakers:")
            print(f"    Cluster 0: {sorted_clusters[0][1]['size']} utterances ({sorted_clusters[0][1]['total_duration']:.1f}s) -> child (c)")
            print(f"    Cluster 1: {sorted_clusters[1][1]['size']} utterances ({sorted_clusters[1][1]['total_duration']:.1f}s) -> female parent (f)")

        elif n_speakers >= 3:
            # Three speakers: child + both parents
            speaker_mapping[sorted_clusters[0][0]] = 'c'  # Most speech = child
            speaker_mapping[sorted_clusters[1][0]] = 'f'  # Second = female
            speaker_mapping[sorted_clusters[2][0]] = 'm'  # Third = male

            print(f"  Three+ speakers:")
            for i in range(min(3, len(sorted_clusters))):
                speaker_type = ['c', 'f', 'm'][i]
                stats = sorted_clusters[i][1]
                print(f"    Cluster {i}: {stats['size']} utterances ({stats['total_duration']:.1f}s) -> {speaker_type}")

            # Additional clusters assigned as child
            for i in range(3, len(sorted_clusters)):
                speaker_mapping[sorted_clusters[i][0]] = 'c'
                stats = sorted_clusters[i][1]
                print(f"    Cluster {i}: {stats['size']} utterances ({stats['total_duration']:.1f}s) -> child (c)")

        # Convert cluster labels to speaker types
        speaker_types = [speaker_mapping[label] for label in labels]
        return speaker_types

    def process_audio(self, audio_file, label_file, output_file, num_speakers=None, max_speakers=3):
        """
        Complete processing pipeline from audio file and labels to speaker assignments.

        Args:
            audio_file (str): Path to audio file (.wav, .mp3, etc.)
            label_file (str): Path to Audacity label file
            output_file (str): Path for output file with speaker assignments (can be None for auto-generation)
            num_speakers (int, optional): Fixed number of speakers. If None, auto-detect.
            max_speakers (int, optional): Maximum speakers when auto-detecting. Default: 3.

        Returns:
            int: Actual number of speakers detected
        """
        print(f"Processing {audio_file}...")

        # Load utterances
        utterances = self.load_audacity_labels(label_file)
        print(f"Loaded {len(utterances)} utterances")

        # Show duration statistics
        durations = [end - start for start, end, content in utterances]
        short_count = sum(1 for d in durations if d < self.min_duration)
        very_short_count = sum(1 for d in durations if d < 0.2)

        print(f"Duration stats:")
        print(f"  Min: {min(durations):.3f}s, Max: {max(durations):.3f}s, Avg: {np.mean(durations):.3f}s")
        print(f"  Short segments (<{self.min_duration}s): {short_count}/{len(durations)} ({100*short_count/len(durations):.1f}%)")
        print(f"  Very short segments (<0.2s): {very_short_count}/{len(durations)} ({100*very_short_count/len(durations):.1f}%)")

        # Extract embeddings with smart extension
        print("Extracting speaker embeddings with context extension...")
        embeddings, valid_utterances = self.extract_embeddings(audio_file, utterances)

        if len(embeddings) == 0:
            print("No valid utterances found!")
            return 0

        # Cluster speakers
        print("Clustering speakers...")
        cluster_labels = self.cluster_speakers(embeddings, num_speakers=num_speakers, max_speakers=max_speakers)

        # Assign speaker types
        print("Assigning speaker types...")
        speaker_types = self.assign_speaker_types(embeddings, cluster_labels, valid_utterances)

        # Get actual number of speakers
        actual_num_speakers = len(set(cluster_labels))

        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.basename(label_file).replace('.txt', '')
            output_dir = os.path.dirname(label_file)
            output_file = os.path.join(output_dir, f"{base_name}_speakers_{actual_num_speakers}_pyannote.txt")

        # Generate output
        self.save_results(valid_utterances, speaker_types, output_file)

        # Print final statistics
        speaker_counts = {}
        for speaker in speaker_types:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        print(f"\nFinal Results:")
        print(f"  Total processed utterances: {len(valid_utterances)}")
        print(f"  Number of speakers detected: {actual_num_speakers}")
        print(f"  Speaker distribution: {speaker_counts}")
        print(f"  Results saved to: {output_file}")

        # Check coverage
        if len(valid_utterances) < len(utterances):
            missing = len(utterances) - len(valid_utterances)
            print(f"  Warning: {missing} utterances could not be processed")

        return actual_num_speakers

    def save_results(self, utterances, speaker_types, output_file):
        """
        Save speaker assignments in Audacity label format.

        Args:
            utterances (list): List of (start, end, content) tuples
            speaker_types (list): Speaker assignments ('c', 'f', 'm')
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            for (start, end, content), speaker in zip(utterances, speaker_types):
                f.write(f"{start:.6f}\t{end:.6f}\t{speaker}\n")

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Family Speaker Clustering using PyAnnote Audio Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speaker_clustering.py -a audio.wav -l labels.txt -t YOUR_TOKEN
  python speaker_clustering.py -a audio.wav -l labels.txt -t YOUR_TOKEN --num-speakers 2
  python speaker_clustering.py -a audio.wav -l labels.txt -t YOUR_TOKEN -m 1.0 -n 3
        """)

    parser.add_argument('--audio', '-a', type=str,
                       default="exp/sandbox/speaker/sk019_7.wav",
                       help='Path to audio file (default: exp/sandbox/speaker/sk019_7.wav)')

    parser.add_argument('--labels', '-l', type=str,
                       default="exp/sandbox/speaker/sk019_7.txt",
                       help='Path to Audacity label file (default: exp/sandbox/speaker/sk019_7.txt)')

    parser.add_argument('--token', '-t', type=str,
                       default="",
                       help='HuggingFace authentication token (default: your token)')

    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: BASENAME_speakers_NUMOFSPEAKERS_pyannote.txt)')

    parser.add_argument('--min-duration', '-m', type=float, default=0.5,
                       help='Minimum duration in seconds for audio segments (default: 0.5)')

    parser.add_argument('--num-speakers', '-n', type=int, default=None,
                       help='Fixed number of speakers (1, 2, or 3). Default: None (automatic)')

    parser.add_argument('--max-speakers', '-s', type=int, default=3,
                       help='Maximum speakers when auto-detecting (default: 3)')

    return parser.parse_args()

def main():
    """Main function to run the speaker clustering pipeline."""
    args = parse_arguments()

    # Initialize clustering system
    clustering = FamilySpeakerClustering(
        hf_token=args.token,
        min_duration=args.min_duration
    )

    print(f"Input audio: {args.audio}")
    print(f"Input labels: {args.labels}")
    print(f"Min duration: {args.min_duration}s")
    if args.num_speakers:
        print(f"Fixed speakers: {args.num_speakers}")
    else:
        print(f"Auto-detect speakers (max: {args.max_speakers})")
    print()

    # Process the audio and get the actual number of speakers
    actual_num_speakers = clustering.process_audio(args.audio, args.labels, args.output,
                                                  num_speakers=args.num_speakers,
                                                  max_speakers=args.max_speakers)

if __name__ == "__main__":
    main()
