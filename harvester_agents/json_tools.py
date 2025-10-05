import json
import os
import sys

def json_write(file_path, data_dict):
    """
    Writes a Python dictionary to a JSON file using json.dump for proper serialization.

    Args:
        file_path (str): Path to the JSON file.
        data_dict (dict): Python dictionary to serialize.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(f"Successfully wrote JSON to {file_path}")
    except Exception as e:
        print(f"Error writing JSON to {file_path}: {e}")

def json_update(file_path, update_code):
    """
    Reads a JSON file, executes update_code on the data dict, and writes back.

    Args:
        file_path (str): Path to the JSON file.
        update_code (str): Python code string to execute, where 'data' is the dict.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON from {file_path}: {e}")
        return

    try:
        exec(update_code, {'data': data})
    except Exception as e:
        print(f"Error executing update code: {e}")
        return

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully updated JSON in {file_path}")
    except Exception as e:
        print(f"Error writing updated JSON to {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_tools.py <command> [args...]")
        print("Commands:")
        print("  write <file_path> <json_str>")
        print("  update <file_path> <update_code>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "write":
        if len(sys.argv) != 4:
            print("Usage: python json_tools.py write <file_path> <json_str>")
            sys.exit(1)
        file_path = sys.argv[2]
        json_str = sys.argv[3]
        try:
            data_dict = json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON string: {e}")
            sys.exit(1)
        json_write(file_path, data_dict)

    elif command == "update":
        if len(sys.argv) != 4:
            print("Usage: python json_tools.py update <file_path> <update_code>")
            sys.exit(1)
        file_path = sys.argv[2]
        update_code = sys.argv[3]
        json_update(file_path, update_code)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)