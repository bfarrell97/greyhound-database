
import os

file_path = r"src/gui/app.py"

print(f"Fixing null bytes in {file_path}...")
try:
    with open(file_path, "rb") as f:
        content = f.read()

    original_len = len(content)
    # Remove null bytes
    clean_content = content.replace(b'\x00', b'')
    new_len = len(clean_content)

    print(f"Read {original_len} bytes. New size {new_len} bytes.")
    print(f"Removed {original_len - new_len} null bytes.")

    with open(file_path, "wb") as f:
        f.write(clean_content)

    print("File saved successfully.")

except Exception as e:
    print(f"Error: {e}")
