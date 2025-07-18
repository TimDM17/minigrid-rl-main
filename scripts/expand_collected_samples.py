import pickle
import argparse
import os

def expand_progressive_training_examples(input_path, output_path, max_len=None):
    # Load original dataset
    with open(input_path, "rb") as f:
        raw_data = pickle.load(f)

    expanded_data = []

    # Expand only the first 100 samples
    num_expand = 400
    for episode_idx, sample in enumerate(raw_data[:num_expand]):
        image_seq = sample["image_seq"]
        action_seq = sample["action_seq"]
        state_labels = sample["env_symbolic_state"]

        total_steps = len(image_seq)
        max_t = total_steps if max_len is None else min(max_len, total_steps)

        for t in range(2, max_t + 1):
            expanded_sample = {
                f"image_seq": image_seq[:t],
                f"action_seq": action_seq[:t],
                f"env_symbolic_state": state_labels[:t]
            }
            expanded_data.append(expanded_sample)

    # Keep the remaining samples unchanged
    remaining_data = raw_data[num_expand:]
    expanded_data.extend(remaining_data)

    print(f"\n✅ Expanded dataset from {len(raw_data)} → {len(expanded_data)} samples (expanded {num_expand} samples, kept {len(remaining_data)} unchanged).")
    print(f"💾 Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(expanded_data, f)
    print("🎉 Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="symbolic_dataset2.pkl", help="Input dataset file")
    parser.add_argument("--output", type=str, default="expanded_symbolic_dataset2.pkl", help="Output expanded file")
    parser.add_argument("--max_len", type=int, default=None, help="Max progressive length per episode (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
    else:
        expand_progressive_training_examples(args.input, args.output, args.max_len)
