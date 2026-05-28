import yaml
import os
from src.pseudo_label import generate_pseudo_labels, merge_pseudo_with_labeled

def main():
    if not os.path.exists('data/features_pseudo.pkl'):
        with open('config.yaml', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        print("Generating pseudo labels...")
        res = generate_pseudo_labels(cfg)
        print("Result:", res)
        
        print("Merging pseudo labels with labeled data...")
        save_path = merge_pseudo_with_labeled(cfg)
        print("Augmented data saved to:", save_path)
    else:
        print("data/features_pseudo.pkl already exists. Skipping generation.")

if __name__ == '__main__':
    main()
