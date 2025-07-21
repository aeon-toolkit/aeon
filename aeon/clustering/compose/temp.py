import os

if __name__ == "__main__":

    root_dir = "/Users/chrisholder/Downloads/soft barycentre averaging"

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            print(rel_path)
